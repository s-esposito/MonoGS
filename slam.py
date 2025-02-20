import os

# import sys
import time

# from argparse import ArgumentParser
from datetime import datetime
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import tyro
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify
import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.camera_utils import CameraExtrinsics, CameraIntrinsics
from gaussian_splatting.utils.system_utils import mkdir_p
from viewer import slam_viewer
from utils.multiprocessing_utils import clone_obj

# from viewer.gui_utils import ParamsGUI
from viewer.viewer_packet import MainToViewerPacket
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_traj_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log

# from utils.multiprocessing_utils import FakeQueue
from utils.slam_mapper import Mapper
from utils.slam_tracker import Tracker


@dataclass
class State:
    pause: bool = False  # written by the GUI

    # shared
    # cur_kf_list: list = dataclasses.field(default_factory=list)

    # tracker

    # mapper
    first_time_pruned: bool = False


class SLAM:
    def __init__(self, args, config, save_dir=None):

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        # pipeline_params = munchify(config["pipeline_params"])
        # self.model_params, self.opt_params, self.pipeline_params = (
        #     model_params,
        #     opt_params,
        #     pipeline_params,
        # )
        self.model_params = model_params
        self.opt_params = opt_params

        self.use_gui = args.gui.active
        self.use_threading = True
        # self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.config["Results"]["save_dir"] = save_dir

        # model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        #
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        self.window_size = 30

        nr_objects = len(self.dataset.static_objects_idxs) + len(
            self.dataset.dynamic_objects_idxs
        )

        #
        self.gaussians = GaussianModel(
            config=args.gaussians,
            nr_objects=nr_objects,
            device=args.system.device,
        )
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(opt_params)

        #
        self.cam_intrinsics = CameraIntrinsics.init_from_dataset(self.dataset)

        bg_color = [0, 0, 0]
        self.background = torch.tensor(
            bg_color, dtype=torch.float32, device=args.system.device
        )

        # create state
        self.state = State()

        # create queues for the frontend and backend
        self.q_map2track = mp.Queue() if self.use_threading else None
        self.q_track2map = mp.Queue() if self.use_threading else None

        # create queues for the gui
        # main to visualization queue
        self.q_main2vis = mp.Queue() if self.use_gui else None
        # visualization to main queue
        self.q_vis2main = mp.Queue() if self.use_gui else None

        # start gui process
        if self.use_gui:
            params_gui = {
                "nr_objects": nr_objects,
                "background": self.background,
                "cam_intrinsics": self.cam_intrinsics,
                "q_main2vis": self.q_main2vis,
                "q_vis2main": self.q_vis2main,
            }
            gui_process = mp.Process(target=slam_viewer.run, args=(params_gui,))
            gui_process.start()
            time.sleep(5)
        else:
            gui_process = None

        # instantiate cuda events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # record the start time
        torch.cuda.synchronize()
        start.record()

        # start frontend process
        self.tracker = Tracker(
            state=self.state,
            config=self.config,
            dataset=self.dataset,
            background=self.background,
            q_map2track=self.q_map2track,
            q_track2map=self.q_track2map,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
            window_size=self.window_size,
            device=args.system.device,
        )
        # self.tracker.pipeline_params = self.pipeline_params
        self.tracker.set_hyperparams()

        # start backend process
        self.mapper = Mapper(
            state=self.state,
            config=self.config,
            gaussians=self.gaussians,
            cam_intrinsics=self.cam_intrinsics,
            opt_params=self.opt_params,
            background=self.background,
            q_map2track=self.q_map2track,
            q_track2map=self.q_track2map,
            window_size=self.window_size,
            device=args.system.device,
        )
        self.mapper.set_hyperparams()

        if self.use_threading:
            # WITH THREADING

            # start the backend process
            backend_process = mp.Process(target=self.mapper.run)  # separate process
            backend_process.start()

            self.tracker.run()

            # # start the frontend process
            # frontend_process = mp.Process(target=self.tracker.run)  # separate process
            # frontend_process.start()

            # join the backend process
            backend_process.join()
            Log("Mapper joined the main thread")

            # # join the frontend process
            # frontend_process.join()
            # Log("Frontend joined the main thread")
        else:
            # WITHOUT THREADING

            self.run()

        # record the end time
        torch.cuda.synchronize()
        end.record()

        # N_frames = len(self.tracker.cameras)
        # FPS = N_frames / (start.elapsed_time(end) * 0.001)
        # Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        # Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        # TODO: reactivate this
        if False:
            # if self.eval_rendering:
            #
            self.gaussians = self.tracker.gaussians
            # kf_indices = self.tracker.kf_indices

            # skip this if the dataset does not have gt poses
            if self.dataset.has_traj:
                ATE = eval_traj_ate(
                    frames=self.tracker.cameras,
                    kf_idxs=None,  # self.tracker.kf_indices,
                    save_dir=self.save_dir,
                    latest_frame_idx=len(self.tracker.cameras) - 1,
                    final=True,
                    # correct_scale=False,
                )
            else:
                ATE = 0.0

            #
            rendering_result = eval_rendering(
                self.tracker.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                # self.pipeline_params,
                self.background,
                kf_indices=None,  # kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                # FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not self.q_map2track.empty():
                self.q_map2track.get()
            self.q_track2map.put(["refinement"])
            while True:
                if self.q_map2track.empty():
                    time.sleep(0.01)
                    continue
                data = self.q_map2track.get()
                if data[0] == "sync_backend" and self.q_map2track.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.tracker.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                # self.pipeline_params,
                self.background,
                kf_indices=None,  # kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        if self.use_gui:
            self.q_main2vis.put(MainToViewerPacket(finish=True))
            # wait for user to close the GUI
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

        Log("Closing all queues")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # empty all queues
        if self.q_map2track is not None:
            while not self.q_map2track.empty():
                self.q_map2track.get()
            self.q_map2track.close()

        if self.q_track2map is not None:
            while not self.q_track2map.empty():
                self.q_track2map.get()
            self.q_track2map.close()

        if self.q_main2vis is not None:
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            self.q_main2vis.close()

        if self.q_vis2main is not None:
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main.close()

        Log("SLAM finished")

    def run(self):

        # TODO: finish implementation
        # WITHOUT THREADING

        Log("Started")

        cur_frame_idx = 0
        while True:

            if cur_frame_idx >= len(self.dataset):
                break

            #
            if self.q_vis2main is not None:
                if self.q_vis2main.empty():
                    if self.state.pause:
                        continue
                else:
                    # get the data
                    data_vis2main = self.q_vis2main.get()
                    self.state.pause = data_vis2main.paused
                    Log(f"Paused: {self.state.pause}")

            #
            Log(
                f"Processing frame: {cur_frame_idx}, lenght window {len(self.tracker.cur_kf_list)}",
            )

            # get new frame

            cur_viewpoint = CameraExtrinsics.init_from_dataset(
                self.dataset,
                cur_frame_idx,
            )
            #
            cur_viewpoint.compute_grad_mask(self.config)

            # update cameras
            self.tracker.cameras[cur_frame_idx] = cur_viewpoint

            # check if first frame
            if cur_frame_idx == 0:

                # Initialise the frame at the ground truth pose
                cur_viewpoint.update_RT(cur_viewpoint.R_gt, cur_viewpoint.T_gt)

                # self.tracker.kf_indices.append(cur_frame_idx)

                # get the depth and segmentation
                cur_depth, cur_segmentation = (
                    self.tracker.get_viewpoint_depth_and_segmentation(cur_viewpoint)
                )

                # add the first keyframe to viewpoints dict
                self.mapper.viewpoints_dict[cur_frame_idx] = cur_viewpoint

                # first frmae is a keyframe
                self.mapper.add_next_kf(
                    frame_idx=cur_frame_idx,
                    cur_viewpoint=cur_viewpoint,
                    cur_depth=cur_depth,
                    cur_segmentation=cur_segmentation,
                    init=True,
                )

                #
                self.mapper.initialize_map(
                    cur_frame_idx=cur_frame_idx, cur_viewpoint=cur_viewpoint
                )

                # exchange info
                self.tracker.gaussians = self.mapper.gaussians
                self.tracker.cam_intrinsics = self.mapper.cam_intrinsics
                self.tracker.occ_aware_visibility_dict = (
                    self.mapper.occ_aware_visibility_dict
                )
                for kf_idx in self.mapper.cur_kf_list:
                    kf_mapper = self.mapper.viewpoints_dict[kf_idx]
                    self.tracker.cameras[kf_idx].update_RT(kf_mapper.R, kf_mapper.T)

                self.tracker.cur_kf_list.append(cur_frame_idx)
                if not self.tracker.is_window_full:
                    # check if window is full
                    self.tracker.is_window_full = (
                        len(self.tracker.cur_kf_list) == self.window_size
                    )

                cur_frame_idx += 1
                continue

            else:

                # Track new frame
                render_pkg = self.tracker.tracking(cur_frame_idx, cur_viewpoint)

                # # create a dict with viewpoints in current windows
                # viewpoints = {
                #     kf_idx: self.tracker.cameras[kf_idx] for kf_idx in self.tracker.cur_kf_list
                # }
                # # add a new packet to the visualization queue
                # if self.q_main2vis is not None:
                #     # Log("Sending to viewer", tag="Tracker")
                #     self.q_main2vis.put(
                #         MainToViewerPacket(
                #             gaussians=clone_obj(self.gaussians),
                #             cur_viewpoint=cur_viewpoint,
                #             cam_intrinsics=clone_obj(self.cam_intrinsics),
                #             viewpoints=viewpoints,
                #             # keyframes=keyframes,
                #             cur_kf_list=self.cur_kf_list,  # current_window_dict,
                #         )
                #     )

                #
                last_keyframe_idx = self.tracker.cur_kf_list[0]
                check_time = (
                    cur_frame_idx - last_keyframe_idx
                ) >= self.tracker.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.tracker.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.tracker.occ_aware_visibility_dict,
                )

                # check if window is full
                if len(self.tracker.cur_kf_list) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility,
                        self.tracker.occ_aware_visibility_dict[last_keyframe_idx],
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility,
                        self.tracker.occ_aware_visibility_dict[last_keyframe_idx],
                    ).count_nonzero()
                    point_ratio = intersection / union
                    # condition to create a keyframe
                    create_kf = (
                        check_time
                        and point_ratio < 0.9  # self.config["Training"]["kf_overlap"]
                    )

                create_kf = check_time and create_kf

                if create_kf:

                    self.tracker.cur_kf_list, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.tracker.occ_aware_visibility_dict,
                        self.tracker.cur_kf_list,
                    )

                    if removed is not None:
                        print("removed from window", removed)

                    depth, segmentation = self.get_viewpoint_depth_and_segmentation(
                        cur_frame_idx, cur_viewpoint
                    )

                    self.request_keyframe(
                        cur_frame_idx, cur_viewpoint, depth, segmentation
                    )

                else:
                    self.cleanup(cur_frame_idx)

                cur_frame_idx += 1

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

            # TODO: remove
            break


@dataclass
class System:
    device: str = "cuda:0"
    seed: int = 42

    # set the seed
    def __post_init__(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Guassians:
    sh_degree: int = 0
    isotropic: bool = True


@dataclass
class GUI:
    active: bool = True


@dataclass
class Results:
    save: bool = True
    eval_rendering: bool = False
    wandb: bool = False
    path: Path = Path("results")


@dataclass
class Dataset:
    path: Path = Path("datasets")


@dataclass
class Args:
    #
    config_path: Path
    run_eval: bool = False
    #
    dataset: Dataset = dataclasses.field(default_factory=Dataset)
    #
    gaussians: Guassians = dataclasses.field(default_factory=Guassians)
    #
    results: Results = dataclasses.field(default_factory=Results)
    #
    gui: GUI = dataclasses.field(default_factory=GUI)
    #
    system: System = dataclasses.field(default_factory=System)


if __name__ == "__main__":

    # mp.set_start_method("spawn", force=True)
    mp.set_start_method("spawn")

    # Parse command line arguments with tyro
    args = tyro.cli(Args, description="SLAM parameters")

    # # Set up command line argument parser
    # parser = ArgumentParser(description="Training parameters")
    # parser.add_argument("--config", type=str)
    # parser.add_argument("--eval", action="store_true")

    # args = parser.parse_args(sys.argv[1:])

    print("Loading config from", args.config_path)
    with open(args.config_path, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config_path)
    save_dir = None

    # TODO: reactivate this
    # if args.run_eval:
    #     Log("Running MonoGS in Evaluation Mode")
    #     Log("Following config will be overriden")
    #     Log("\tsave_results=True")
    #     config["Results"]["save_results"] = True
    #     Log("\tuse_gui=False")
    #     config["Results"]["use_gui"] = False
    #     Log("\teval_rendering=True")
    #     config["Results"]["eval_rendering"] = True
    #     Log("\tuse_wandb=True")
    #     config["Results"]["use_wandb"] = True

    if args.results.save:
        # mkdir_p(args.results.path)
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.dataset.path = Path(config["Dataset"]["dataset_path"])
        path_split = str(args.dataset.path).split("/")
        dataset_name = path_split[-3]
        print("dataset_name", dataset_name)
        scene_name = path_split[-2]
        print("scene_name", scene_name)
        save_dir = os.path.join(
            args.results.path, dataset_name + "_" + scene_name, current_datetime
        )
        tmp = str(args.config_path).split(".")[0]
        args.results.path = save_dir
        mkdir_p(save_dir)

        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)

        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if args.results.wandb else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(args, config, save_dir=save_dir)

    # slam.run()
    wandb.finish()

    # All done
    Log("Done.")
