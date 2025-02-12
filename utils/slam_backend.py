import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.cam_intrinsics = None
        # self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.q_back2front = None
        self.q_front2back = None
        self.live_mode = False

        # self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        # self.monocular = True  # config["Training"]["monocular"]
        self.nr_iters = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False  # not self.monocular
        self.keyframe_optimizers = None

        Log("Backend initialized", tag="Backend")

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        # self.single_thread = True

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=1.0, depth_map=None):
        #
        Log(f"Adding next keyframe {frame_idx}, init: {init}, scale: {scale}")

        if self.cam_intrinsics is None:
            raise Exception("Camera intrinsics are not initialized")

        self.gaussians.extend_from_pcd_seq(
            viewpoint,
            self.cam_intrinsics,
            kf_id=frame_idx,
            init=init,
            scale=scale,
            depthmap=depth_map,
        )

    def reset(self):
        #
        Log("Resetting", tag="Backend")
        self.nr_iters = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False  # not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        Log("Removing all gaussians", tag="Backend")
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

        # remove everything from the queues
        Log("Clearing the queues", tag="Backend")
        while not self.q_front2back.empty():
            self.q_front2back.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        #
        Log("Initializing the map", tag="Backend")
        for mapping_iteration in tqdm(range(self.init_itr_num)):
            self.nr_iters += 1
            render_pkg = render(
                viewpoint,
                self.cam_intrinsics,
                self.gaussians,
                # self.pipeline_params,
                self.background,
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.nr_iters == self.init_gaussian_reset or (
                    self.nr_iters == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map", tag="Backend")

        return render_pkg

    def optimize_map(self, current_window, prune=False, iters=1):

        # if the window is empty, return
        if len(current_window) == 0:
            Log("Empty window", tag="Backend")
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        # disable_pbar = iters == 1
        nr_mapping_iters = 0
        pbar = tqdm(range(iters), desc="Mapping", ncols=100, disable=True)
        for _ in pbar:

            self.nr_iters += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            pbar_window = tqdm(
                range(len(current_window)), desc="Window", ncols=100, disable=True
            )
            for cam_idx in pbar_window:

                # sample camera from window
                viewpoint = viewpoint_stack[cam_idx]
                # keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint,
                    self.cam_intrinsics,
                    self.gaussians,
                    # self.pipeline_params,
                    self.background,
                )
                # extract render results
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                
                nr_mapping_iters += 1

            # Randomly sample two additional viewpoints
            cam_idxs = torch.randperm(len(random_viewpoint_stack))[:2]
            pbar_random = tqdm(cam_idxs, desc="Random", ncols=100, disable=True)
            for cam_idx in pbar_random:

                # sample camera from random
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint,
                    self.cam_intrinsics,
                    self.gaussians,
                    # self.pipeline_params,
                    self.background,
                )
                # extract render results
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                
                nr_mapping_iters += 1

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()

            gaussian_split = False

            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    Log("Pruning Gaussians", tag="Backend")
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None:  # and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM", tag="Backend")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.nr_iters % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                # Opacity reset
                if (self.nr_iters % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians", tag="Backend")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                #
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.nr_iters)
                # 
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    # do not update the first frame
                    if viewpoint.frame_idx == 0:
                        continue
                    update_pose(viewpoint)
            
        Log(f"Optimized map for {nr_mapping_iters} iters", tag="Backend")

        return gaussian_split

    def color_refinement(self):
        Log("Color refinement", tag="Backend")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam,
                self.cam_intrinsics,
                self.gaussians,
                # self.pipeline_params,
                self.background,
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.rgb.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        
    def push_to_frontend(self, tag):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        msg = [
            tag,
            clone_obj(self.gaussians),
            self.occ_aware_visibility,
            keyframes,
            clone_obj(self.cam_intrinsics),
        ]
        self.q_back2front.put(msg)

    def run(self):

        Log("Backend started", tag="Backend")

        while True:

            # if backend queue is empty, sleep for a while
            if self.q_front2back.empty():
                
                # Log("Queue empty", tag="Backend")
                
                # if self.pause:
                #     time.sleep(0.01)
                #     continue

                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                # if self.single_thread:
                # time.sleep(0.01)
                # continue

                # TODO: should activate?
                # optimize map if frontend queue is empty
                # self.optimize_map(self.current_window, prune=False, iters=1)
                
                # TODO: needed?
                # if self.last_sent >= 10:
                #     self.optimize_map(self.current_window, prune=True, iters=10)
                #     self.push_to_frontend("sync_backend")

            else:

                data = self.q_front2back.get()

                if data[0] == "stop":
                    Log("Stopping", tag="Backend")
                    break

                # elif data[0] == "pause":
                #     self.pause = True
                #     Log("Paused the backend", tag="Backend")

                # elif data[0] == "unpause":
                #     self.pause = False
                #     Log("Unpaused the backend", tag="Backend")

                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend("sync_backend")

                elif data[0] == "init":

                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":

                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num  # if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    # iterate over the keyframes in the window
                    for cam_idx in range(len(self.current_window)):
                        
                        # do not optimize the first frame
                        if self.current_window[cam_idx] == 0:
                            continue

                        # get viewpoint
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.frame_idx),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.frame_idx),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.frame_idx),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.frame_idx),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.optimize_map(
                        self.current_window, prune=False, iters=iter_per_kf
                    )
                    self.optimize_map(self.current_window, prune=True, iters=1)
                    self.push_to_frontend("keyframe")

                else:

                    raise Exception("Unprocessed data", data)

        self.push_to_frontend("stop")

        # empty the queues
        while not self.q_front2back.empty():
            self.q_front2back.get()

        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()  # Helps with shared memory cleanup

        Log("Backend halted", tag="Backend")

        return
