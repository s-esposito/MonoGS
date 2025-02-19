import csv
import glob
import os
import json
import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm
from PIL import Image
from pyquaternion import Quaternion
import itertools
from io import StringIO


from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass


class KubricParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=24)
        self.n_img = len(self.color_paths)

        # this dataset has no ground truth trajectory info

    def load_poses(self, datapath, frame_rate=-1):

        # list all poses
        self.poses = []

        # open datapath/metadata.json as dict
        json_path = os.path.join(datapath, "metadata.json")
        with open(json_path, "r") as f:
            metadata = json.load(f)

        metadata = metadata["camera"]
        positions = metadata["positions"]
        quaternions = metadata["quaternions"]
        for position, quat in zip(positions, quaternions):
            quat = Quaternion(quat)
            rot = quat.rotation_matrix
            # flip y and z axis
            # rot = np.array([rot[0], -rot[2], -rot[1]])
            # Construct the transformation matrix (c2w)
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = position
            local_transform = np.eye(4)
            local_transform[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

            T = T @ local_transform

            # Convert to w2c
            T = np.linalg.inv(T)
            # flip y and z axis
            local_transform = np.array
            self.poses += [T]

        # list all files in datapath/rgb
        self.color_paths = sorted(glob.glob(f"{datapath}/rgba/*.png"))
        # sort
        self.color_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        # list all files in datapath/depth (depth_00000.tiff)
        self.depth_paths = sorted(glob.glob(f"{datapath}/depth/*.tiff"))
        # sort
        self.depth_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        # list all files in datapath/segmentation
        self.segmentation_paths = sorted(glob.glob(f"{datapath}/segmentation/*.png"))
        # sort
        self.segmentation_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))


class DavisParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=24)
        self.n_img = len(self.color_paths)

    def load_poses(self, datapath, frame_rate=-1):

        # list all poses
        self.poses = []  # this dataset has no ground truth trajectory info

        # list all files in datapath/depth
        self.depth_paths = []

        # list all files in datapath/rgb
        self.color_paths = sorted(glob.glob(f"{datapath}/rgb/*.jpg"))
        # sort
        self.color_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        # list all files in datapath/segmentation
        self.segmentation_paths = sorted(glob.glob(f"{datapath}/segmentation/*.png"))
        # sort
        self.segmentation_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))


class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        # frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            # frame = {
            #     "file_path": self.color_paths[i],
            #     "depth_path": self.depth_paths[i],
            #     "transform_matrix": pose.tolist(),
            # }

            # frames.append(frame)
        # self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        # data = np.loadtxt(filepath, delimiter=" ", dtype=str, skiprows=skiprows)
        data = np.genfromtxt(
            filepath, delimiter=" ", dtype=str, skip_header=skiprows, filling_values=""
        )
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths = []
        self.poses = []
        self.depth_paths = []
        # self.frames = []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            # frame = {
            #     "file_path": str(os.path.join(datapath, image_data[i, 1])),
            #     "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
            #     "transform_matrix": (np.linalg.inv(T)).tolist(),
            # }

            # self.frames.append(frame)


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        # frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]

            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)

            self.poses += [np.linalg.inv(T_w_c)]

            # frame = {
            #     "file_path": self.color_paths[i],
            #     "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            # }

            # frames.append(frame)
        # self.frames = frames


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999
        # objects
        self.static_objects_idxs = []
        self.dynamic_objects_idxs = []
        self.masked_objects_idxs = []

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        objects = config["Dataset"].get("Objects", None)

        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        self.use_depth = calibration.get("use_depth", False)
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # rgb
        self.color_paths = []
        # segmentation masks
        self.has_segmentation = False
        self.segmentation_paths = []
        # objects (idx and names of segments)
        if objects is not None:
            self.static_objects_idxs = objects["static"]
            self.dynamic_objects_idxs = objects["dynamic"]
            self.masked_objects_idxs = objects["masked"]

        # gt poses
        self.has_traj = True
        self.poses = []
        # depth parameters
        self.has_depth = False
        self.depth_paths = []
        self.depth_scale = calibration.get("depth_scale", None)

        # # Default scene scale
        # nerf_normalization_radius = 5
        # self.scene_info = {
        #     "nerf_normalization": {
        #         "radius": nerf_normalization_radius,
        #         "translation": np.zeros(3),
        #     },
        # }
        
        self.preload = False
        self.color_imgs = []
        self.depth_imgs = []
        self.segmentation_imgs = []
            
            
    def load_data(self):
        
        self.preload = True
        
        for color_path in tqdm(self.color_paths, desc="Loading RGB"):
            # remove alpha channel
            self.color_imgs.append(np.array(Image.open(color_path))[..., :3])

        if self.has_depth and self.use_depth:
            for depth_path in tqdm(self.depth_paths, desc="Loading Depth"):
                self.depth_imgs.append(np.array(Image.open(depth_path)) / self.depth_scale)
                
        if self.has_segmentation:
            for segmentation_path in tqdm(self.segmentation_paths, desc="Loading Segmentation"):
                self.segmentation_imgs.append(np.array(Image.open(segmentation_path)))
            
    # def start_gt_traj_from_identity(self):
    #     # get first pose
    #     first_pose = self.poses[0]

    #     # compute all other poses as relative to the first pose
    #     relative_poses = []
    #     for pose in self.poses[1:]:
    #         relative_poses.append(np.dot(np.linalg.inv(first_pose), pose))

    #     # recompute absolute poses from relative poses, now setting the first pose to identity
    #     self.poses = [np.eye(4)]
    #     for pose in relative_poses:
    #         self.poses.append(pose)

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]

        # gt pose if available
        if self.has_traj:
            pose = self.poses[idx]
        else:
            pose = None

        # rgb
        if self.preload:
            image = self.color_imgs[idx]
        else:
            image = np.array(Image.open(color_path))[..., :3]  # remove alpha channel

        # depth
        if self.has_depth and self.use_depth:
            if self.preload:
                depth = self.depth_imgs[idx]
            else:
                depth_path = self.depth_paths[idx]
                depth = np.array(Image.open(depth_path)) / self.depth_scale
        else:
            depth = None

        # segments to mask
        if self.has_segmentation:
            if self.preload:
                segmentation = self.segmentation_imgs[idx]
            else:
                segmentation_path = self.segmentation_paths[idx]
                segmentation = np.array(Image.open(segmentation_path))
        else:
            segmentation = None

        if self.has_segmentation:
            # mask image
            mask = np.ones_like(image[..., 0], dtype=np.bool)
            for obj_idx in self.masked_objects_idxs:
                mask[segmentation == obj_idx] = False

        # undistort image
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        # convert to tensor
        
        if pose is not None:
            pose = torch.from_numpy(pose).to(device=self.device)
        
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        if depth is not None:
            depth = torch.from_numpy(depth).to(device=self.device, dtype=self.dtype)

        if segmentation is not None:
            segmentation = torch.from_numpy(segmentation).to(
                device=self.device, dtype=torch.long
            )

        if mask is not None:
            mask = torch.from_numpy(mask).to(device=self.device, dtype=torch.bool)

        # print(f"Image shape: {image.shape}, min: {image.min()}, max: {image.max()}, dtype: {image.dtype}")

        # if depth is not None:
        #     print(f"Depth shape: {depth.shape}, min: {depth.min()}, max: {depth.max()}, dtype: {depth.dtype}")
        # else:
        #     print("Depth shape: None")

        # if segmentation is not None:
        #     print(f"Segmentation shape: {segmentation.shape}, unique classes: {torch.unique(segmentation)}, dtype: {segmentation.dtype}")
        # else:
        #     print("Segmentation shape: None")

        # if mask is not None:
        #     print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        # else:
        #     print("Mask shape: None")

        # if pose is not None:
        #     print(f"Pose shape: {pose.shape}, dtype: {pose.dtype}")
        # else:
        #     print("Pose shape: None")

        data = {
            "rgb": image,
            "depth": depth,
            "mask": mask,
            "segmentation": segmentation,
            "pose": pose,
        }

        return data


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        data = {
            "rgb": image,
            "depth": depth,
            "pose": pose,
        }

        return data


class KubricDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = KubricParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.segmentation_paths = parser.segmentation_paths
        self.poses = parser.poses
        # self.start_gt_traj_from_identity()
        #
        self.has_segmentation = True
        self.has_depth = True
        self.has_traj = True
        #
        print(f"Color paths lenght: {len(self.color_paths)}")
        print(f"Depth paths lenght: {len(self.depth_paths)}")
        print(f"Segmentation paths lenght: {len(self.segmentation_paths)}")
        print(f"Poses lenght: {len(self.poses)}")
        #
        print("Static objects IDs", self.static_objects_idxs)
        print("Dynamic objects IDs", self.dynamic_objects_idxs)
        print("Nr dyn objects", len(self.dynamic_objects_idxs))
        # 
        self.load_data()


class DavisDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = DavisParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.segmentation_paths = parser.segmentation_paths
        self.poses = parser.poses
        # self.start_gt_traj_from_identity()
        #
        self.has_segmentation = True
        self.has_depth = False
        self.has_traj = False
        #
        print(f"Color paths lenght: {len(self.color_paths)}")
        print(f"Depth paths lenght: {len(self.depth_paths)}")
        print(f"Segmentation paths lenght: {len(self.segmentation_paths)}")
        print(f"Poses lenght: {len(self.poses)}")


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        if len(self.depth_paths) > 0:
            self.has_depth = True
        self.poses = parser.poses
        # self.start_gt_traj_from_identity()
        print(f"Color paths lenght: {len(self.color_paths)}")
        print(f"Depth paths lenght: {len(self.depth_paths)}")
        print(f"Poses lenght: {len(self.poses)}")


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        if len(self.depth_paths) > 0:
            self.has_depth = True
        self.poses = parser.poses
        # self.start_gt_traj_from_identity()


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280

        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True
        else:
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.w, self.h, rs.format.bgr8, 30
        )
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()

        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()

    def __getitem__(self, idx):

        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data()) * self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        data = {
            "rgb": image,
            "depth": depth,
            "pose": pose,
        }

        return data


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    elif config["Dataset"]["type"] == "davis":
        return DavisDataset(args, path, config)
    elif config["Dataset"]["type"] == "kubric":
        return KubricDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
