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
from utils.camera_utils import CameraIntrinsics
from gaussian_splatting.utils.system_utils import mkdir_p
from viewer import slam_viewer
from viewer.gui_utils import ParamsGUI
from viewer.viewer_packet import MainToViewerPacket
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
# from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_main import Main


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

        self.live_mode = False  # self.config["Dataset"]["type"] == "realsense"
        self.monocular = True  # self.config["Dataset"]["sensor_type"] == "monocular"
        # self.use_spherical_harmonics = False  # self.config["Training"]["spherical_harmonics"]
        self.use_gui = True  # self.config["Results"]["use_gui"]
        # if self.live_mode:
        #     self.use_gui = True
        # self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.config["Results"]["save_dir"] = save_dir
        # self.config["Training"]["monocular"] = self.monocular

        # model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        #
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        
        #
        self.gaussians = GaussianModel(config=args.gaussians)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(opt_params)

        #
        self.cam_intrinsics = CameraIntrinsics.init_from_dataset(self.dataset)

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # create queues for the frontend and backend
        q_back2main = mp.Queue()
        q_main2back = mp.Queue()

        # create queues for the gui
        # main to visualization queue
        q_main2vis = mp.Queue() if self.use_gui else None  # FakeQueue()
        # visualization to main queue
        q_vis2main = mp.Queue() if self.use_gui else None  # FakeQueue()

        # start gui process
        if self.use_gui:
            params_gui = ParamsGUI(
                # pipe=self.pipeline_params,
                nr_objects=self.dataset.nr_objects,
                background=self.background,
                gaussians=self.gaussians,
                cam_intrinsics=self.cam_intrinsics,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
                width_data=self.dataset.width,
                height_data=self.dataset.height,
            )
            gui_process = mp.Process(target=slam_viewer.run, args=(params_gui,))
            gui_process.start()
            time.sleep(5)

        # instantiate cuda events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # record the start time
        torch.cuda.synchronize()
        start.record()

        # start frontend process
        self.frontend = Main(
            config=self.config,
            dataset=self.dataset,
            background=self.background,
            q_back2main=q_back2main,
            q_main2back=q_main2back,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        # self.frontend.pipeline_params = self.pipeline_params
        self.frontend.set_hyperparams()
        
        # start backend process
        self.backend = BackEnd(
            config=self.config,
            gaussians=self.gaussians,
            cam_intrinsics=self.cam_intrinsics,
            opt_params=self.opt_params,
            background=self.background,
            cameras_extent=6.0,
            q_back2main=q_back2main,
            q_main2back=q_main2back,
        )
        # self.backend.live_mode = self.live_mode
        self.backend.set_hyperparams()
        
        # start the backend process
        backend_process = mp.Process(target=self.backend.run)  # separate process
        backend_process.start()
        
        # start the frontend process
        self.frontend.run()  # same process
        
        # join the backend process
        backend_process.join()
        Log("Backend joined the main thread")
                
        # record the end time
        torch.cuda.synchronize()
        end.record()
        
        # N_frames = len(self.frontend.cameras)
        # FPS = N_frames / (start.elapsed_time(end) * 0.001)
        # Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        # Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        # TODO: reactivate this
        if False:
            # if self.eval_rendering:
            #
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices

            # skip this if the dataset does not have gt poses
            if self.dataset.has_traj:
                ATE = eval_ate(
                    self.frontend.cameras,
                    self.frontend.kf_indices,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )
            else:
                ATE = 0.0

            #
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                # self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
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
            while not q_back2main.empty():
                q_back2main.get()
            q_main2back.put(["color_refinement"])
            while True:
                if q_back2main.empty():
                    time.sleep(0.01)
                    continue
                data = q_back2main.get()
                if data[0] == "sync_backend" and q_back2main.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                # self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
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
            q_main2vis.put(MainToViewerPacket(finish=True))
            # wait for user to close the GUI
            gui_process.join()
            Log("GUI Stopped and joined the main thread")
        
        Log("Closing all queues")
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # empty all queues
        while not q_back2main.empty():
            q_back2main.get()
        q_back2main.close()
        
        while not q_main2back.empty():
            q_main2back.get()
        q_main2back.close()
        
        if q_main2vis is not None:
            while not q_main2vis.empty():
                q_main2vis.get()
            q_main2vis.close()
            
        if q_vis2main is not None:
            while not q_vis2main.empty():
                q_vis2main.get()
            q_vis2main.close()    
        
        Log("SLAM finished")

@dataclass
class Guassians:
    sh_degree: int = 0
    isotropic: bool = False

@dataclass
class GUI:
    active: bool = True

@dataclass
class Results:
    save: bool = True
    eval_rendering: bool = False
    wandb: bool = False
    save_path: Path = Path("results")

@dataclass
class Args:
    #
    config_path: Path
    run_eval: bool = False
    #
    gaussians: Guassians = dataclasses.field(default_factory=Guassians)
    #
    results: Results = dataclasses.field(default_factory=Results)
    #
    gui: GUI = dataclasses.field(default_factory=GUI)


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

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = str(args.config_path).split(".")[0]
        config["Results"]["save_dir"] = save_dir
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
