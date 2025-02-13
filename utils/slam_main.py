import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix, getWorld2View
from utils.camera_utils import CameraExtrinsics, CameraIntrinsics
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from viewer.viewer_packet import MainToViewerPacket


class Main(mp.Process):
    def __init__(
        self,
        config,
        dataset,
        background,
        q_back2main,
        q_main2back,
        q_main2vis,
        q_vis2main,
    ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.background = background
        # self.pipeline_params = None
        self.q_back2main = q_back2main
        self.q_main2back = q_main2back
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main

        self.initialized = False
        self.kf_indices = []
        # self.monocular = config["Training"]["monocular"]
        self.nr_iters = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        # self.reset = True
        self.requested_init = False
        self.requested_keyframe = False
        self.requested_stop = False
        # self.use_every_n_frames = 1

        self.gaussians = None  # GaussianModel
        self.cameras = dict()  # dict of CameraExtrinsics
        self.cam_intrinsics = None  # CameraIntrinsics
        self.device = "cuda:0"
        self.pause = False

        Log("Created", tag="Main")

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        #  self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        cur_viewpoint = self.cameras[cur_frame_idx]
        gt_img = cur_viewpoint.rgb.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        
        if depth is None:
            initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
            initial_depth += torch.randn_like(initial_depth) * 0.3
        else:
            assert opacity is not None, "Opacity has to be given"
            depth = depth.detach().clone()
            opacity = opacity.detach()
            # use_inv_depth = False
            # if use_inv_depth:
            #     inv_depth = 1.0 / depth
            #     inv_median_depth, inv_std, valid_mask = get_median_depth(
            #         inv_depth, opacity, mask=valid_rgb, return_std=True
            #     )
            #     invalid_depth_mask = torch.logical_or(
            #         inv_depth > inv_median_depth + inv_std,
            #         inv_depth < inv_median_depth - inv_std,
            #     )
            #     invalid_depth_mask = torch.logical_or(
            #         invalid_depth_mask, ~valid_mask
            #     )
            #     inv_depth[invalid_depth_mask] = inv_median_depth
            #     inv_initial_depth = inv_depth + torch.randn_like(
            #         inv_depth
            #     ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
            #     initial_depth = 1.0 / inv_initial_depth
            # else:
            median_depth, std, valid_mask = get_median_depth(
                depth, opacity, mask=valid_rgb, return_std=True
            )
            invalid_depth_mask = torch.logical_or(
                depth > median_depth + std, depth < median_depth - std
            )
            invalid_depth_mask = torch.logical_or(
                invalid_depth_mask, ~valid_mask
            )
            depth[invalid_depth_mask] = median_depth
            initial_depth = depth + torch.randn_like(depth) * torch.where(
                invalid_depth_mask, std * 0.5, std * 0.2
            )

            initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels

        return initial_depth.cpu().numpy()[0]
        
        # # use the observed depth
        # initial_depth = torch.from_numpy(cur_viewpoint.depth).unsqueeze(0)
        # initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        # return initial_depth[0].numpy()

    def init(self):
        Log("Initializing", tag="Main")
        self.initialized = False  # not self.monocular
        self.kf_indices = []
        self.nr_iters = 0
        self.occ_aware_visibility = {}
        self.current_window = []

    def tracking(self, cur_frame_idx, cur_viewpoint):

        # get previous camera pose
        prev = self.cameras[cur_frame_idx - 1]  # self.use_every_n_frames]

        # TODO: try constant velocity forward projection
        cur_viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [cur_viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(cur_viewpoint.frame_idx),
            }
        )
        opt_params.append(
            {
                "params": [cur_viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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
        pose_optimizer = torch.optim.Adam(opt_params)

        nr_tracking_iters = 0
        pbar = tqdm(range(self.tracking_itr_num), desc=f"Tracking {cur_frame_idx}", disable=True)
        for tracking_itr in pbar:
            
            render_pkg = render(
                cur_viewpoint,
                self.cam_intrinsics,
                self.gaussians,
                # self.pipeline_params,
                self.background,
            )
            
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            
            pose_optimizer.zero_grad()
            
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, cur_viewpoint
            )
            
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(cur_viewpoint, converged_threshold=1e-4)

            if tracking_itr % 10 == 0:
                gt_depth = cur_viewpoint.depth
                if cur_viewpoint.depth is None:
                    gt_depth = np.zeros(
                        (self.cam_intrinsics.height, self.cam_intrinsics.width)
                    )
                self.q_main2vis.put(
                    MainToViewerPacket(
                        current_frame=cur_viewpoint,
                        current_frame_idx=cur_frame_idx,
                        gt_rgb=cur_viewpoint.rgb,
                        gt_depth=gt_depth,
                    )
                )

            #
            nr_tracking_iters += 1
            
            if converged:
                break
        Log(f"Frame: {cur_frame_idx} tracked for {nr_tracking_iters} iters", tag="Main")

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_stop(self):
        Log("Requesting backend stop", tag="Main")
        self.q_main2back.put(["stop"])
    
    def request_keyframe(self, cur_frame_idx, cur_viewpoint, depthmap):
        Log("Requesting keyframe", tag="Main")
        msg = ["keyframe", cur_frame_idx, cur_viewpoint, self.current_window, depthmap]
        self.q_main2back.put(msg)
        self.requested_keyframe = True

    def request_init(self, cur_frame_idx, cur_viewpoint, depth_map):
        Log("Requesting initialization", tag="Main")
        msg = ["init", cur_frame_idx, cur_viewpoint, depth_map]
        self.q_main2back.put(msg)
        self.requested_init = True

    def sync_from_backend(self, data):
        # torch.cuda.synchronize()
        Log(f"Unpacking {data[0]} msg from backend", tag="Main")
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.cam_intrinsics = data[4]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):

        Log("Started", tag="Main")
        
        self.init()

        # tic = torch.cuda.Event(enable_timing=True)
        # toc = torch.cuda.Event(enable_timing=True)

        # prec_frame_idx = -1
        cur_frame_idx = 0
        while True:
            
            if cur_frame_idx >= len(self.dataset):
                if not self.requested_stop:
                    self.request_stop()
                    self.requested_stop = True
            
            # 
            if self.q_vis2main is not None:
                if self.q_vis2main.empty():
                    if self.pause:
                        continue
                else:
                    # get the data
                    data_vis2main = self.q_vis2main.get()
                    self.pause = data_vis2main.paused
                    Log(f"Paused: {self.pause}", tag="Main")
                    if self.pause:
                        self.q_main2back.put(["pause"])
                        continue
                    else:
                        self.q_main2back.put(["unpause"])

            # if the frontend queue is empty, do stuff
            if self.q_back2main.empty():
                
                # Log("Queue empty", tag="Main")
                
                # tic.record()

                if self.requested_init:
                    # waiting for the backend to init
                    time.sleep(0.01)
                    continue

                if self.requested_keyframe:
                    # waiting for the backend to add the keyframe
                    time.sleep(0.01)
                    continue
                
                if self.requested_stop:
                    # waiting for the backend to stop
                    time.sleep(0.01)
                    continue

                cur_viewpoint = CameraExtrinsics.init_from_dataset(
                    self.dataset,
                    cur_frame_idx,
                )
                #
                cur_viewpoint.compute_grad_mask(self.config)

                #
                self.cameras[cur_frame_idx] = cur_viewpoint

                # check if first frame
                if cur_frame_idx == 0:
                    
                    # Initialise the frame at the ground truth pose
                    cur_viewpoint.update_RT(cur_viewpoint.R_gt, cur_viewpoint.T_gt)
                    
                    depth_map = self.add_new_keyframe(cur_frame_idx)
                    self.request_init(cur_frame_idx, cur_viewpoint, depth_map)
                    
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # self.initialized = self.initialized or (
                #     len(self.current_window) == self.window_size
                # )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, cur_viewpoint)

                #
                # current_window_dict = {}
                # current_window_dict[self.current_window[0]] = self.current_window[1:]
                # keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                
                # create a dict with viewpoints in current windows
                viewpoints = {kf_idx: self.cameras[kf_idx] for kf_idx in self.current_window}
                # TODO: unefficent way to add viewpoints ... slows viewer down 
                # max_added_viewpoints = 100
                # one_every = int(np.ceil(len(self.dataset) / max_added_viewpoints))
                # for i in range(0, len(self.cameras), one_every):
                #     if i not in viewpoints.keys():
                #         viewpoints[i] = self.cameras[i]
                
                # add a new packet to the visualization queue
                if self.q_main2vis is not None:
                    self.q_main2vis.put(
                        MainToViewerPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=cur_viewpoint,
                            cam_intrinsics=clone_obj(self.cam_intrinsics),
                            viewpoints=viewpoints,
                            # keyframes=keyframes,
                            kf_window=self.current_window,  # current_window_dict,
                        )
                    )

                # #
                # if self.requested_keyframe:
                #     self.cleanup(cur_frame_idx)
                #     cur_frame_idx += 1
                #     continue

                #
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                
                # if self.single_thread:
                create_kf = check_time and create_kf
                
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    # if self.monocular and not self.initialized and removed is not None:
                    #     self.reset = True
                    #     Log(
                    #         "Keyframes lacks sufficient overlap to init the map, resetting."
                    #     )
                    #     continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"]
                    )
                    self.request_keyframe(cur_frame_idx, cur_viewpoint, depth_map)
                
                else:
                    self.cleanup(cur_frame_idx)
                
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    # skip this if the dataset does not have gt poses
                    if self.dataset.has_traj:
                        Log("Evaluating ATE at frame: ", cur_frame_idx, tag="Eval")
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            cur_frame_idx,
                            monocular=True,  # self.monocular,
                        )
                
                # toc.record()
                
            else:

                # if frontend queue is not empty, sync with backend

                data = self.q_back2main.get()
                
                if data[0] == "sync_backend":
                    self.sync_from_backend(data)

                elif data[0] == "keyframe":
                    self.sync_from_backend(data)
                    self.requested_keyframe = False

                elif data[0] == "init":
                    self.sync_from_backend(data)
                    self.requested_init = False
                    
                elif data[0] == "stop":
                    self.sync_from_backend(data)
                    self.requested_stop = False
                    break
        
        # 
        Log("Finished", tag="Main")
        
        if self.save_results:
            
            # skip this if the dataset does not have gt poses
            if self.dataset.has_traj:
                Log("Evaluating ATE at frame: ", cur_frame_idx, tag="Eval")
                eval_ate(
                    self.cameras,
                    self.kf_indices,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=True  # self.monocular,
                )
            
            # save the final gaussians
            save_gaussians(
                self.gaussians, self.save_dir, "final", final=True
            )
