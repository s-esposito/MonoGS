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


class Mapper(mp.Process):
    def __init__(
        self,
        state,
        config,
        gaussians,
        cam_intrinsics,
        opt_params,
        background,
        q_map2track,
        q_track2map,
        window_size,
        device,
    ):
        super().__init__()
        self.state = state
        self.config = config
        self.gaussians = gaussians
        self.cam_intrinsics = cam_intrinsics
        # self.pipeline_params = None
        self.opt_params = opt_params
        self.background = background
        self.q_map2track = q_map2track
        self.q_track2map = q_track2map

        #
        # self.live_mode = False

        # self.pause = False
        self.device = device
        # self.dtype = torch.float32
        # self.monocular = True  # config["Training"]["monocular"]
        self.nr_iters = 0
        # self.last_sent = 0
        self.occ_aware_visibility_dict = {}
        self.viewpoints_dict = {}
        self.cur_kf_list = []
        # self.first_time_pruned = False  # not self.monocular
        self.keyframe_optimizers = None
        self.window_size = window_size

        Log("Created", tag="Mapper")

        self.init()

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = 1050  # self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = (
            100  # self.config["Training"]["init_gaussian_update"]
        )
        self.init_gaussian_reset = 500  # self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = 0.005  # self.config["Training"]["init_gaussian_th"]

        self.cameras_extent = 1.0  # 6.0
        self.init_gaussian_extent = self.cameras_extent * 30
        # (
        #     self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        # )
        self.mapping_itr_num = 150  # self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = (
            150  # self.config["Training"]["gaussian_update_every"]
        )
        self.gaussian_update_offset = (
            50  # self.config["Training"]["gaussian_update_offset"]
        )
        self.gaussian_th = 0.7  # self.config["Training"]["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * 1.0
        # (
        #     self.cameras_extent * self.config["Training"]["gaussian_extent"]
        # )
        self.gaussian_reset = 2001  # self.config["Training"]["gaussian_reset"]
        self.size_threshold = 20  # self.config["Training"]["size_threshold"]
        # self.single_thread = True

    def add_next_kf(self, frame_idx, cur_viewpoint, init=False):
        #
        Log(f"Adding keyframe {frame_idx}, init: {init}", tag="Mapper")

        # check if the gaussians / camera intrinsics are initialized
        assert self.gaussians is not None, "Gaussians are not initialized"
        assert self.cam_intrinsics is not None, "Camera intrinsics are not initialized"
        # assert params are given
        # assert cur_depth is not None, "Depth map is not given"
        # assert cur_segmentation is not None, "Segmentation is not given"
        # assert cur_depth and cur_segmentation are torch tensors and on the same device
        # assert isinstance(cur_depth, torch.Tensor), "Depth map is not a torch tensor"
        # assert isinstance(
        #     cur_segmentation, torch.Tensor
        # ), "Segmentation is not a torch tensor"
        # assert cur_depth.device == self.device, f"Depth map device ({cur_depth.device}) is not {self.device}"
        # assert cur_segmentation.device == self.device, f"Segmentation device ({cur_segmentation.device}) is not {self.device}"

        # render cur_viewpoint to get the render_image, render_depth,
        if not init:
            with torch.no_grad():
                render_pkg = render(
                    cur_viewpoint,
                    self.cam_intrinsics,
                    self.gaussians.get_xyz,
                    self.gaussians.get_rotation,
                    self.gaussians.get_scaling,
                    self.gaussians.get_opacity,
                    self.gaussians.get_features,
                    # self.gaussians.active_sh_degree,
                    # self.pipeline_params,
                    self.background,
                )
                if render_pkg is None:
                    raise ValueError("Render package is None")
                (
                    render_depth,
                    render_opacity,
                ) = (
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
        else:
            render_depth = None
            render_opacity = None

        self.gaussians.extend_from_pcd_seq(
            viewpoint=cur_viewpoint,
            cam_intrinsics=self.cam_intrinsics,
            render_depth=render_depth,
            render_opacity=render_opacity,
            kf_idx=frame_idx,
            init=init,
            # scale=scale,
        )
        
        if init:
            with torch.no_grad():
                # print shapes
                print("xyz", self.gaussians.get_xyz.shape)
                print("features_dc", self.gaussians.get_features.shape)
                # print("features_rest", new_features_rest.shape)
                print("scaling", self.gaussians.get_scaling.shape)
                print("rotation", self.gaussians.get_rotation.shape)
                print("opacity", self.gaussians.get_opacity.shape)
                print("obj_prob", self.gaussians.get_obj_prob.shape)

    def init(self):
        #
        Log("Initializing", tag="Mapper")
        self.nr_iters = 0
        self.occ_aware_visibility_dict = {}
        self.viewpoints_dict = {}
        self.cur_kf_list = []
        self.first_time_pruned = False  # not self.monocular
        self.keyframe_optimizers = None

    def initialize_map(self, cur_frame_idx, cur_viewpoint):
        #
        Log("Initializing the map", tag="Mapper")
        for mapping_iteration in tqdm(range(self.init_itr_num)):
            self.nr_iters += 1
            render_pkg = render(
                cur_viewpoint,
                self.cam_intrinsics,
                self.gaussians.get_xyz,
                self.gaussians.get_rotation,
                self.gaussians.get_scaling,
                self.gaussians.get_opacity,
                self.gaussians.get_features,
                # self.gaussians.active_sh_degree,
                # self.pipeline_params,
                self.background,
            )
            if render_pkg is None:
                raise ValueError("Render package is None")
            (
                render_image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                render_depth,
                render_opacity,
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
                render_image,
                render_depth,
                # render_opacity,
                cur_viewpoint,
                init=True,
                invert_depth=False,
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii_2d[visibility_filter] = torch.max(
                    self.gaussians.max_radii_2d[visibility_filter],
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

        self.occ_aware_visibility_dict[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map", tag="Mapper")

        return render_pkg

    def optimize_map(self, cur_kf_list, prune=False, iters=1):

        # if the window is empty, return
        if len(cur_kf_list) == 0:
            Log("Empty window", tag="Mapper")
            return

        # Keyframes viewpoints
        viewpoint_stack = [self.viewpoints_dict[kf_idx] for kf_idx in cur_kf_list]

        # frames_to_optimize = self.config["Training"]["pose_window"]

        # disable_pbar = iters == 1
        nr_mapping_iters = 0
        pbar = tqdm(range(iters), desc="Mapping", ncols=100, disable=True)
        for _ in pbar:

            self.nr_iters += 1
            # self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            pbar_window = tqdm(
                range(len(cur_kf_list)), desc="Window", ncols=100, disable=True
            )
            for cam_idx in pbar_window:

                # sample camera from window
                cur_viewpoint = viewpoint_stack[cam_idx]
                # keyframes_opt.append(cur_viewpoint)
                render_pkg = render(
                    cur_viewpoint,
                    self.cam_intrinsics,
                    self.gaussians.get_xyz,
                    self.gaussians.get_rotation,
                    self.gaussians.get_scaling,
                    self.gaussians.get_opacity,
                    self.gaussians.get_features,
                    # self.gaussians.active_sh_degree,
                    # self.pipeline_params,
                    self.background,
                )
                if render_pkg is None:
                    raise ValueError("Render package is None")
                # extract render results
                (
                    render_image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    render_depth,
                    render_opacity,
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
                    render_image,
                    render_depth,
                    # render_opacity,
                    cur_viewpoint,
                    init=False,
                    invert_depth=False,
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                nr_mapping_iters += 1

            if False:
                # Map from the two additional random viewpoints

                # Randomly sample two additional viewpoints
                random_viewpoint_stack = []
                current_window_set = set(cur_kf_list)
                for cam_idx, cur_viewpoint in self.viewpoints_dict.items():
                    if cam_idx in current_window_set:
                        continue
                    random_viewpoint_stack.append(cur_viewpoint)

                cam_idxs = torch.randperm(len(random_viewpoint_stack))[:2]
                pbar_random = tqdm(cam_idxs, desc="Random", ncols=100, disable=True)
                for cam_idx in pbar_random:

                    # sample camera from random
                    cur_viewpoint = random_viewpoint_stack[cam_idx]
                    render_pkg = render(
                        cur_viewpoint,
                        self.cam_intrinsics,
                        self.gaussians.get_xyz,
                        self.gaussians.get_rotation,
                        self.gaussians.get_scaling,
                        self.gaussians.get_opacity,
                        self.gaussians.get_features,
                        # self.gaussians.active_sh_degree,
                        # self.pipeline_params,
                        self.background,
                    )
                    if render_pkg is None:
                        raise ValueError("Render package is None")
                    # extract render results
                    (
                        render_image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        render_depth,
                        render_opacity,
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
                        render_image,
                        render_depth,
                        # render_opacity,
                        cur_viewpoint,
                        init=False,
                        invert_depth=False,
                    )
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)

                    nr_mapping_iters += 1

            # TODO: reactivate if isotropic is False
            # scaling = self.gaussians.get_scaling
            # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            # loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()

            gaussian_split = False

            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility_dict = {}
                for idx in range((len(cur_kf_list))):
                    kf_idx = cur_kf_list[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility_dict[kf_idx] = (n_touched > 0).long()

                # compute the visibility of the gaussians
                # Only prune on the last iteration and when we have full window
                if prune:
                    Log("Pruning Gaussians", tag="Mapper")
                    # only prune if we have a full window
                    if (
                        len(cur_kf_list) == self.window_size
                    ):  # self.config["Training"]["window_size"]:
                        # prune_mode = self.config["Training"]["prune_mode"]  # slam
                        prune_coviz = 3
                        self.gaussians.nr_obs.fill_(0)
                        # count the number of observations
                        for (
                            window_idx,
                            visibility,
                        ) in self.occ_aware_visibility_dict.items():
                            self.gaussians.nr_obs += visibility.cpu()
                        # to_prune = None
                        # if prune_mode == "odometry":
                        #     to_prune = self.gaussians.nr_obs < 3
                        #     # make sure we don't split the gaussians, break here.
                        # if prune_mode == "slam":
                        if not self.first_time_pruned:
                            # first time visibility pruning
                            kf_mask = self.gaussians.kf_idx >= 0
                            self.first_time_pruned = True
                        else:
                            # only prune keyframes which are relatively new
                            # keyframes idx in the window are in descending order
                            sorted_window = sorted(cur_kf_list, reverse=True)
                            # mask of gaussians whose kf is in the last 3 keyframes
                            kf_mask = self.gaussians.kf_idx >= sorted_window[2]
                        # mask of guassians that are observed less than 3 times
                        obs_mask = self.gaussians.nr_obs <= prune_coviz
                        # join the masks
                        to_prune = torch.logical_and(obs_mask, kf_mask)
                        # if to_prune is not None:  # and self.monocular:
                        self.gaussians.prune_points(to_prune.cuda())
                        for i in range((len(cur_kf_list))):
                            kf_idx = cur_kf_list[i]
                            self.occ_aware_visibility_dict[kf_idx] = (
                                self.occ_aware_visibility_dict[kf_idx][~to_prune]
                            )

                    # returns false because Gaussians have not been split
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii_2d[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii_2d[visibility_filter_acm[idx]],
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
                if (self.nr_iters % self.gaussian_reset) == 0 and (not update_gaussian):
                    Log("Resetting the opacity of non-visible Gaussians", tag="Mapper")
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
                # for cam_idx in range(min(frames_to_optimize, len(cur_kf_list))):
                for cam_idx in range(len(cur_kf_list)):
                    cur_viewpoint = viewpoint_stack[cam_idx]
                    # do not update the first frame
                    if cur_viewpoint.frame_idx == 0:
                        continue
                    update_pose(cur_viewpoint)

        Log(f"Optimized map for {nr_mapping_iters} iters", tag="Mapper")

        return gaussian_split

    def refinement(self, on_all_frames=False):

        # TODO: refine ALL camera poses too

        Log("Refinement", tag="Mapper")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints_dict.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints_dict[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam,
                self.cam_intrinsics,
                self.gaussians.get_xyz,
                self.gaussians.get_rotation,
                self.gaussians.get_scaling,
                self.gaussians.get_opacity,
                self.gaussians.get_features,
                # self.gaussians.active_sh_degree,
                # self.pipeline_params,
                self.background,
            )
            if render_pkg is None:
                raise ValueError("Render package is None")
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.rgb.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_ssim) * (
                Ll1
            ) + self.opt_params.lambda_ssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii_2d[visibility_filter] = torch.max(
                    self.gaussians.max_radii_2d[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

    def push_to_frontend(self, tag):
        # torch.cuda.synchronize()
        # self.last_sent = 0
        keyframes = []
        for kf_idx in self.cur_kf_list:
            kf = self.viewpoints_dict[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        msg = [
            tag,
            clone_obj(self.gaussians),
            self.occ_aware_visibility_dict,
            keyframes,
            clone_obj(self.cam_intrinsics),
        ]
        self.q_map2track.put(msg)

    def run(self):

        Log("Started", tag="Mapper")

        while True:

            # if backend queue is empty, sleep for a while
            if self.q_track2map.empty():

                # Log("Queue empty", tag="Mapper")

                if self.state.pause:
                    time.sleep(0.01)
                    continue

                if len(self.cur_kf_list) == 0:
                    time.sleep(0.01)
                    continue

                # if self.single_thread:
                # time.sleep(0.01)
                # continue

                # optimize map if frontend queue is empty
                # self.optimize_map(self.cur_kf_list, prune=False, iters=1)

                # TODO: needed?
                # if self.last_sent >= 10:
                #     self.optimize_map(self.cur_kf_list, prune=True, iters=10)
                #     self.push_to_frontend("sync_backend")

            else:

                data = self.q_track2map.get()

                if data[0] == "stop":
                    Log("Stopping", tag="Mapper")
                    break

                elif data[0] == "pause":
                    self.state.pause = True
                    Log("Paused the backend", tag="Mapper")

                elif data[0] == "unpause":
                    self.state.pause = False
                    Log("Unpaused the backend", tag="Mapper")

                elif data[0] == "refinement":
                    # run color refinement
                    self.refinement()
                    # push results to frontend
                    self.push_to_frontend("sync_backend")

                elif data[0] == "init":
                    # initialize the map (first frame)
                    cur_frame_idx = data[1]
                    cur_viewpoint = data[2]

                    # add the first keyframe
                    self.viewpoints_dict[cur_frame_idx] = cur_viewpoint
                    #
                    self.add_next_kf(
                        frame_idx=cur_frame_idx,
                        cur_viewpoint=cur_viewpoint,
                        init=True,
                    )
                    #
                    self.initialize_map(
                        cur_frame_idx=cur_frame_idx, cur_viewpoint=cur_viewpoint
                    )
                    # push results to frontend
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    # add a new keyframe to expand the map
                    cur_frame_idx = data[1]
                    cur_viewpoint = data[2]
                    cur_kf_list = data[3]

                    # add the keyframe
                    self.viewpoints_dict[cur_frame_idx] = cur_viewpoint
                    self.cur_kf_list = cur_kf_list

                    # TODO:

                    #
                    self.add_next_kf(
                        cur_frame_idx,
                        cur_viewpoint,
                        init=False,
                    )

                    opt_params = []
                    # frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = (
                        300  # self.mapping_itr_num  # if self.single_thread else 10
                    )
                    # if not self.first_time_pruned:
                    #     if (
                    #         len(self.cur_kf_list)
                    #         == self.config["Training"]["window_size"]
                    #     ):
                    #         frames_to_optimize = (
                    #             self.config["Training"]["window_size"] - 1
                    #         )
                    #         # iter_per_kf = 50 if self.live_mode else 300
                    #         iter_per_kf = 300
                    #         Log("Mapping", tag="Mapper")
                    #     else:
                    # iter_per_kf = self.mapping_itr_num

                    # iterate over the keyframes in the window
                    for cam_idx in range(len(self.cur_kf_list)):

                        # do not optimize the first frame
                        if self.cur_kf_list[cam_idx] == 0:
                            continue

                        # get cur_viewpoint
                        cur_viewpoint = self.viewpoints_dict[cur_kf_list[cam_idx]]
                        # if cam_idx < frames_to_optimize:
                        opt_params.append(
                            {
                                "params": [cur_viewpoint.cam_rot_delta],
                                "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                * 0.5,
                                "name": "rot_{}".format(cur_viewpoint.frame_idx),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [cur_viewpoint.cam_trans_delta],
                                "lr": self.config["Training"]["lr"]["cam_trans_delta"]
                                * 0.5,
                                "name": "trans_{}".format(cur_viewpoint.frame_idx),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [cur_viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(cur_viewpoint.frame_idx),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [cur_viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(cur_viewpoint.frame_idx),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.optimize_map(self.cur_kf_list, prune=False, iters=iter_per_kf)
                    self.optimize_map(self.cur_kf_list, prune=True, iters=1)
                    # push results to frontend
                    self.push_to_frontend("keyframe")

                else:

                    raise Exception("Unprocessed data", data)

        # push results to frontend
        self.push_to_frontend("stop")
        time.sleep(0.1)

        Log("Finished", tag="Mapper")

        return
