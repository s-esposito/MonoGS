import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix, getWorld2View
from utils.camera_utils import CameraExtrinsics, CameraIntrinsics
from utils.eval_utils import eval_traj_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from viewer.viewer_packet import MainToViewerPacket


class Tracker(mp.Process):
    def __init__(
        self,
        state,
        config,
        dataset,
        background,
        q_map2track,
        q_track2map,
        q_main2vis,
        q_vis2main,
        window_size,
        device
    ):
        super().__init__()
        self.state = state
        self.config = config
        self.dataset = dataset
        self.background = background
        # self.pipeline_params = None
        self.q_map2track = q_map2track
        self.q_track2map = q_track2map
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main

        self.is_window_full = False
        self.nr_iters = 0
        self.occ_aware_visibility_dict = {}
        self.cur_kf_list = []

        self.requested_init = False
        self.requested_keyframe = False
        self.requested_stop = False

        self.gaussians = None  # GaussianModel
        self.cameras = dict()  # dict of CameraExtrinsics
        self.cam_intrinsics = None  # CameraIntrinsics
        self.window_size = window_size
        self.device = device

        Log("Created", tag="Tracker")

        self.init()
        
    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = True  # self.config["Results"]["save_results"]
        self.save_trj = True  # self.config["Results"]["save_trj"]
        self.save_trj_every = 10  # self.config["Results"]["save_trj_kf_intv"]
        self.send_gui_every = 1

        self.tracking_itr_num = 100  # self.config["Training"]["tracking_itr_num"]
        self.kf_interval = 1  # 5  # self.config["Training"]["kf_interval"]
        self.check_viewpoints_overlap = False
        #  self.single_thread = self.config["Training"]["single_thread"]

    def init(self):
        Log("Initializing", tag="Tracker")
        self.is_window_full = False  # not self.monocular
        # self.kf_indices = []
        self.nr_iters = 0
        self.occ_aware_visibility_dict = {}
        self.cur_kf_list = []

    def tracking(self, cur_frame_idx, cur_viewpoint):

        # get previous camera pose
        if cur_frame_idx == 0:
            raise ValueError("Cannot track first frame")
        elif cur_frame_idx >= 1:
            prev = self.cameras[cur_frame_idx - 1]  # self.use_every_n_frames]]
            cur_viewpoint.update_RT(prev.R, prev.T)
        # TODO: eval this better; seems to be worse
        # elif cur_frame_idx > 1:
        #     # Et+1 = Et + (Et − Et-1)
        #     prev = self.cameras[cur_frame_idx - 1]
        #     prev2 = self.cameras[cur_frame_idx - 2]
        #     # Compute the relative rotation
        #     delta_R = prev.R @ torch.linalg.inv(prev2.R)  # ΔR = R_t-1 * R_t-2⁻¹
        #     # Extrapolate the new rotation
        #     newR = delta_R @ prev.R  # R_t+1 = ΔR * R_t
        #     newT = prev.T + (prev.T - prev2.T)
        #     cur_viewpoint.update_RT(newR, newT)

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
        pbar = tqdm(
            range(self.tracking_itr_num), desc=f"Tracking {cur_frame_idx}", disable=True
        )
        for tracking_itr in pbar:
            
            pose_optimizer.zero_grad()

            render_pkg = render(
                cur_viewpoint,
                self.cam_intrinsics,
                self.gaussians.get_xyz,
                self.gaussians.get_rotation,
                self.gaussians.get_scaling,
                self.gaussians.get_opacity,
                self.gaussians.get_features,
                self.gaussians.active_sh_degree,
                # self.pipeline_params,
                self.background,
            )

            render_image, render_depth, render_opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            loss_tracking = get_loss_tracking(
                render_image,
                render_depth,
                render_opacity,
                cur_viewpoint,
                invert_depth=False
            )

            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(cur_viewpoint, converged_threshold=1e-4)
                
            if self.q_main2vis is not None and tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    MainToViewerPacket(
                        cur_viewpoint=cur_viewpoint,
                        cur_frame_idx=cur_frame_idx,
                    )
                )

            #
            nr_tracking_iters += 1

            if converged:
                break

        # Log(f"Frame: {cur_frame_idx} tracked for {nr_tracking_iters} iters", tag="Tracker")

        self.median_depth = get_median_depth(render_depth, render_opacity)
        return render_pkg

    def should_add_as_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_visibility_mask,
    ):
        kf_translation = 0.08  # self.config["Training"]["kf_translation"]
        kf_min_translation = 0.05  # self.config["Training"]["kf_min_translation"]
        kf_overlap = 0.9  # self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_visibility_mask, self.occ_aware_visibility_dict[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_visibility_mask, self.occ_aware_visibility_dict[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_visibility_mask
    ):
        N_dont_touch = 2
        self.cur_kf_list = [cur_frame_idx] + self.cur_kf_list
        # remove frames which has little overlap with the current frame

        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(self.cur_kf_list)):
            kf_idx = self.cur_kf_list[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_visibility_mask, self.occ_aware_visibility_dict[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_visibility_mask.count_nonzero(),
                self.occ_aware_visibility_dict[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.is_window_full:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            self.cur_kf_list.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        
        curr_frame = self.cameras[cur_frame_idx]
        kf_0_WC = torch.linalg.inv(getWorld2View(curr_frame.R, curr_frame.T))

        if len(self.cur_kf_list) > self.window_size:  # self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(self.cur_kf_list)):
                inv_dists = []
                kf_i_idx = self.cur_kf_list[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(self.cur_kf_list)):
                    if i == j:
                        continue
                    kf_j_idx = self.cur_kf_list[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = self.cur_kf_list[N_dont_touch + idx]
            self.cur_kf_list.remove(removed_frame)

        return removed_frame

    def request_stop(self):
        Log("Requesting backend stop", tag="Tracker")
        self.q_track2map.put(["stop"])

    def request_keyframe(self, cur_frame_idx, cur_viewpoint):
        Log("Requesting keyframe", tag="Tracker")
        msg = [
            "keyframe",
            cur_frame_idx,
            cur_viewpoint,
            self.cur_kf_list
        ]
        self.q_track2map.put(msg)
        self.requested_keyframe = True

    def request_init(self, cur_frame_idx, cur_viewpoint):
        Log("Requesting initialization", tag="Tracker")
        msg = ["init", cur_frame_idx, cur_viewpoint]
        self.q_track2map.put(msg)
        self.requested_init = True

    def sync_from_backend(self, data):
        # torch.cuda.synchronize()
        Log(f"Unpacking {data[0]} msg from backend", tag="Tracker")
        self.gaussians = data[1]
        occ_aware_visibility_dict = data[2]
        keyframes = data[3]
        self.cam_intrinsics = data[4]
        self.occ_aware_visibility_dict = occ_aware_visibility_dict

        for kf_idx, kf_R, kf_T in keyframes:
            self.cameras[kf_idx].update_RT(kf_R.clone(), kf_T.clone())
            
        # if not self.is_window_full:
        #     # check if window is full
        #     self.is_window_full = len(self.cur_kf_list) == self.window_size
        
    def run(self):

        Log("Started", tag="Tracker")

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
                    if self.state.pause:
                        continue
                else:
                    # get the data
                    data_vis2main = self.q_vis2main.get()
                    self.state.pause = data_vis2main.paused
                    Log(f"Paused: {self.state.pause}", tag="Tracker")
                    if self.state.pause:
                        self.q_track2map.put(["pause"])
                        continue
                    else:
                        self.q_track2map.put(["unpause"])

            # if the frontend queue is empty, do stuff
            if self.q_map2track.empty():

                # Log("Queue empty", tag="Tracker")

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

                Log(
                    f"Processing frame: {cur_frame_idx}, window {self.cur_kf_list}",
                    tag="Tracker",
                )

                cur_viewpoint = CameraExtrinsics.init_from_dataset(
                    self.dataset,
                    cur_frame_idx,
                )
                #
                cur_viewpoint.compute_grad_mask(self.config)
                
                #
                self.cameras[cur_frame_idx] = cur_viewpoint
                
                if self.q_main2vis is not None:
                    self.q_main2vis.put(
                        MainToViewerPacket(
                            cur_viewpoint=cur_viewpoint,
                            cur_frame_idx=cur_frame_idx,
                            unpack_buffers=True,
                        )
                    )

                # check if first frame
                if cur_frame_idx == 0:
                    
                    # Initialise the frame at the ground truth pose
                    cur_viewpoint.update_RT(cur_viewpoint.R_gt, cur_viewpoint.T_gt)

                    self.request_init(cur_frame_idx, cur_viewpoint)

                    self.cur_kf_list.append(cur_frame_idx)
                    
                    cur_frame_idx += 1
                    continue
                    
                # track new frame
                render_pkg = self.tracking(cur_frame_idx, cur_viewpoint)

                # check if keyframe
                last_keyframe_idx = self.cur_kf_list[0]
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                
                #
                create_kf = False
                if self.check_viewpoints_overlap:
                    # if the window is not full
                    if len(self.cur_kf_list) < self.window_size:
                        # check overlap with the last keyframe
                        union = torch.logical_or(
                            curr_visibility, self.occ_aware_visibility_dict[last_keyframe_idx]
                        ).count_nonzero()
                        intersection = torch.logical_and(
                            curr_visibility, self.occ_aware_visibility_dict[last_keyframe_idx]
                        ).count_nonzero()
                        point_ratio = intersection / union
                        create_kf = point_ratio < 0.9  # self.config["Training"]["kf_overlap"]
                    else:
                        create_kf = self.should_add_as_keyframe(
                            cur_frame_idx,
                            last_keyframe_idx,
                            curr_visibility,
                        )
                else:
                    # always add new keyframes
                    create_kf = True

                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval

                if check_time and create_kf:

                    removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility
                    )
                    if removed is not None:
                        Log("Removed frame: ", removed, tag="Tracker")
                    
                    self.request_keyframe(
                        cur_frame_idx, cur_viewpoint
                    )

                cur_frame_idx += 1

                # gui update
                if (
                    self.q_main2vis is not None
                    and cur_frame_idx % self.send_gui_every == 0
                ):
                
                    # create a dict with viewpoints in current windows
                    viewpoints = {
                        kf_idx: self.cameras[kf_idx] for kf_idx in self.cur_kf_list
                    }
                    # # TODO: unefficent way to add viewpoints ... slows viewer down
                    # # max_added_viewpoints = 100
                    # # one_every = int(np.ceil(len(self.dataset) / max_added_viewpoints))
                    # # for i in range(0, len(self.cameras), one_every):
                    # #     if i not in viewpoints.keys():
                    # #         viewpoints[i] = self.cameras[i]

                    # add a new packet to the visualization queue
                    # Log("Sending to viewer", tag="Tracker")
                    self.q_main2vis.put(
                        MainToViewerPacket(
                            gaussians=clone_obj(self.gaussians),
                            # cur_viewpoint=cur_viewpoint,
                            # cam_intrinsics=clone_obj(self.cam_intrinsics),
                            viewpoints=viewpoints,
                            cur_kf_list=self.cur_kf_list,
                        )
                    )
                
                # traj evaluation
                if (
                    self.dataset.has_traj
                    and self.save_results
                    and self.save_trj
                    and cur_frame_idx % self.save_trj_every == 0
                ):
                    # skip this if the dataset does not have gt poses
                    Log("Evaluating ATE at frame: ", cur_frame_idx, tag="Eval")
                    eval_traj_ate(
                        frames=self.cameras,
                        kf_idxs=None,  # self.kf_indices,
                        save_dir=self.save_dir,
                        latest_frame_idx=cur_frame_idx,
                        final=False,
                        # correct_scale=False
                    )

            else:

                # if frontend queue is not empty, sync with backend

                data = self.q_map2track.get()

                if data[0] == "sync_backend":
                    self.sync_from_backend(data)

                elif data[0] == "keyframe":
                    self.sync_from_backend(data)
                    self.requested_keyframe = False

                elif data[0] == "init":
                    self.sync_from_backend(data)
                    self.requested_init = False
                    
                    if self.q_main2vis is not None:
                        self.q_main2vis.put(
                            MainToViewerPacket(
                                cur_viewpoint=cur_viewpoint,
                                cur_frame_idx=cur_frame_idx,
                            )
                        )

                elif data[0] == "stop":
                    self.sync_from_backend(data)
                    self.requested_stop = False
                    break

        #
        Log("Finished", tag="Tracker")

        if self.save_results:

            # skip this if the dataset does not have gt poses
            if self.dataset.has_traj:
                Log("Evaluating ATE at frame: ", cur_frame_idx, tag="Eval")
                eval_traj_ate(
                    frames=self.cameras,
                    kf_idxs=None,  # self.kf_indices,
                    save_dir=self.save_dir,
                    latest_frame_idx=len(self.dataset) - 1,
                    final=True,
                    # correct_scale=True
                )

            # save the final gaussians
            save_gaussians(self.gaussians, self.save_dir, "final", final=True)
