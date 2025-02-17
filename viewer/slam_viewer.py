import pathlib
import threading
import time
from datetime import datetime
from copy import deepcopy
import cv2
import glfw
import imgviz
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import torch.nn.functional as F
from OpenGL import GL as gl
import matplotlib.pyplot as plt
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View
from viewer.gl_render.util import CameraGL
from viewer.gl_render.util_gau import GaussianData
from viewer.gl_render.render_ogl import GaussiansRenderGL
from viewer.viewer_packet import MainToViewerPacket
from viewer.gui_utils import (
    ViewerToMainPacket,
    create_frustum,
    cv_gl,
    get_latest_queue,
)
from utils.camera_utils import CameraExtrinsics, CameraIntrinsics
from utils.logging_utils import Log

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class Viewer:
    def __init__(
        self,
        nr_objects,
        background,
        cam_intrinsics,
        q_main2vis,
        q_vis2main,
    ):
        # app = o3d.visualization.gui.Application.instance
        # app.initialize()

        self.window_w, self.window_h = 1920, 1080

        self.window = gui.Application.instance.create_window(
            "MonoGS", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        #
        self.selected_shader = "rgb"

        self.step = 0
        self.nr_frames = 0
        self.nr_objects = 0
        self.cur_frame_idx = 0
        self.process_finished = False
        self.device = "cuda"

        self.current_frame_frustum = None
        self.current_frame_frustum_gt = None
        self.frustum_dict = {}
        self.model_dict = {}

        self.gaussians = None

        # self.init = False
        self.kf_window = None
        self.render_img = None

        self.width_3d = self.window_w
        self.height_3d = self.window_h

        self.nr_objects = nr_objects
        self.background = background
        self.cam_intrinsics_cur = cam_intrinsics
        # self.init = True
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        # self.pipe = pipe

        # generate nr objects random colors
        self.segments_colors = torch.rand(
            (self.nr_objects, 3), dtype=torch.float32, device="cuda"
        )
        # set first 3 to RGB
        self.segments_colors[0] = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device="cuda"
        )
        self.segments_colors[1] = torch.tensor(
            [0.0, 1.0, 0.0], dtype=torch.float32, device="cuda"
        )
        self.segments_colors[2] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device="cuda"
        )
        
        Log(f"Viewer resolution {self.window_w}x{self.window_h}", tag="GUI")
        Log(f"Data resolution {self.width_3d}x{self.height_3d}", tag="GUI")

        self.init_widget()

        self.camera_gl = CameraGL(self.window_h, self.window_w)
        self.window_gl = self.init_glfw()
        self.renderer_gl = GaussiansRenderGL(self.camera_gl.w, self.camera_gl.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = None

        self.save_path = "."
        self.save_path = pathlib.Path(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # app.run()

        # self.run()
        threading.Thread(target=self._update_thread).start()

    def init_widget(self):

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)

        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"

        self.specular_geo = rendering.MaterialRecord()
        self.specular_geo.shader = "defaultLit"

        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())

        # GUI panel

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        self.button = gui.ToggleSwitch("Resume/Pause")
        self.button.is_on = True
        self.button.set_on_clicked(self._on_button)
        self.panel.add_child(self.button)

        self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))

        # Check boxes
        vp_subtile1.add_child(gui.Label("Camera follow options"))
        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.followcam_chbox = gui.Checkbox("Follow Camera")
        self.followcam_chbox.checked = True
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("From Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)
        vp_subtile1.add_child(chbox_tile)

        # Combo panels
        combo_tile = gui.Vert(0.5 * em, gui.Margins(margin))
        combo_tile.add_child(gui.Label("Viewpoints"))
        # Jump to the camera viewpoint
        self.combo_kf = gui.Combobox()
        self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        combo_tile.add_child(self.combo_kf)

        vp_subtile2.add_child(combo_tile)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))
        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.axis_chbox = gui.Checkbox("Axis")
        self.axis_chbox.checked = False
        self.axis_chbox.set_on_checked(self._on_axis_chbox)
        chbox_tile_3dobj.add_child(self.axis_chbox)

        self.panel.add_child(chbox_tile_3dobj)

        # Rendering options

        self.panel.add_child(gui.Label("Rendering options"))
        # chbox_tile_geometry = gui.Horiz(0.5 * em, gui.Margins(margin))

        # SHADERS
        combo_tile_shaders = gui.Vert(0.5 * em, gui.Margins(margin))
        combo_tile_shaders.add_child(gui.Label("Shaders"))
        self.combo_shaders = gui.Combobox()
        self.combo_shaders.set_on_selection_changed(self._on_combo_shaders)
        # add items
        self.combo_shaders.add_item("rgb")
        self.combo_shaders.add_item("depth")
        self.combo_shaders.add_item("time")
        self.combo_shaders.add_item("elipsoids")
        self.combo_shaders.add_item("segmentation")
        combo_tile_shaders.add_child(self.combo_shaders)
        self.panel.add_child(combo_tile_shaders)

        # Scaling slider

        slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        slider_label = gui.Label("Gaussian Scale (0-1)")
        self.scaling_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scaling_slider.set_limits(0.001, 1.0)
        self.scaling_slider.double_value = 1.0
        slider_tile.add_child(slider_label)
        slider_tile.add_child(self.scaling_slider)
        self.panel.add_child(slider_tile)

        # Screenshot buttom

        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        self.panel.add_child(self.screenshot_btn)

        # Rendering Tab
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        tab_info = gui.Vert(0, tab_margins)

        # current frame idx
        self.frame_idx_info = gui.Label("Current Frame Index: ")
        tab_info.add_child(self.frame_idx_info)

        # nr gaussians
        self.output_info = gui.Label("Number of Gaussians: ")
        tab_info.add_child(self.output_info)

        # input color/depth
        tab_info.add_child(gui.Label("Inputs (RGB/D/Segmentation"))
        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        self.in_segments_widget = gui.ImageWidget()
        tab_info.add_child(self.in_rgb_widget)
        tab_info.add_child(self.in_depth_widget)
        tab_info.add_child(self.in_segments_widget)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)
        self.window.add_child(self.panel)

    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window

    def update_activated_renderer_state(self, gaus):
        self.renderer_gl.update_gaussian_data(gaus)
        self.renderer_gl.sort_and_update(self.camera_gl)
        self.renderer_gl.set_scale_modifier(self.scaling_slider.double_value)
        self.renderer_gl.set_render_mod(-4)
        self.renderer_gl.update_camera_pose(self.camera_gl)
        self.renderer_gl.update_camera_intrin(self.camera_gl)
        self.renderer_gl.set_render_reso(self.camera_gl.w, self.camera_gl.h)

    def add_camera(self, c2w, name, color=[0, 1, 0], size=1.0):
        # check if frustum already exists
        # if name not in self.frustum_dict.keys():
        # get intrinsics from camera
        fx = self.cam_intrinsics_cur.fx.detach().cpu().numpy()
        fy = self.cam_intrinsics_cur.fy.detach().cpu().numpy()
        cx = self.cam_intrinsics_cur.cx
        cy = self.cam_intrinsics_cur.cy
        H = self.cam_intrinsics_cur.height
        W = self.cam_intrinsics_cur.width
        frustum = create_frustum(
            pose=c2w, H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, color=color, size=size
        )
        # add to scene
        self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
        self.update_camera(name, frustum, c2w, color=color)
        return frustum

    def update_camera(self, name, frustum, c2w, color=[0, 1, 0]):

        # update camera pose
        frustum.update_pose(c2w)
        self.widget3d.scene.set_geometry_transform(name, c2w.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        # update color
        colors = [color for i in range(len(frustum.line_set.lines))]
        frustum.line_set.colors = o3d.utility.Vector3dVector(colors)

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 0.7
        self.widget3d_width = int(
            # self.window.size.width * self.widget3d_width_ratio
            self.width_3d
            * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _on_combo_kf(self, name, new_idx):
        if name == "current":
            frustum = self.current_frame_frustum
        elif name == "current_gt":
            frustum = self.current_frame_frustum_gt
        else:
            frustum = self.frustum_dict[name]
        viewpoint = frustum.view_dir

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_combo_shaders(self, name, new_idx):
        self.selected_shader = name

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_axis_chbox(self, is_checked):
        name = "axis"
        if is_checked:
            self.widget3d.scene.remove_geometry(name)
            self.widget3d.scene.add_geometry(name, self.axis, self.lit_geo)
        else:
            self.widget3d.scene.remove_geometry(name)

    def _on_button(self, is_on):
        packet = ViewerToMainPacket()
        packet.paused = not self.button.is_on
        self.q_vis2main.put(packet)

    def _on_screenshot_btn(self):
        if self.render_img is None:
            return
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_path / "screenshots" / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        # create the filename
        filename = save_dir / "screenshot"
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}-gui.png", img)
        img = np.asarray(self.render_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}.png", img)

    @staticmethod
    def resize_img(img, width):
        height = int(width * img.shape[0] / img.shape[1])
        return cv2.resize(img, (width, height))

    def receive_data(self, q):

        if q is None:
            return

        packet = get_latest_queue(q)
        if packet is None:
            return

        # intrinsics
        if packet.cam_intrinsics is not None:
            # Log("Received new intrinsics", tag="GUI")
            self.cam_intrinsics_cur = packet.cam_intrinsics

        # gaussians
        if packet.gaussians is not None:
            # Log("Received new gaussians", tag="GUI")
            self.gaussians = packet.gaussians
            self.output_info.text = "Number of Gaussians: {}".format(
                self.gaussians["means"].shape[0]
            )

        # current frame idx
        if packet.current_frame_idx is not None:
            # Log("Received new frame idx", tag="GUI")
            self.cur_frame_idx = packet.current_frame_idx
            self.frame_idx_info.text = "Current Frame Index: {}".format(
                self.cur_frame_idx
            )

        # current frame
        if packet.current_frame is not None:

            # Log("Received current viewpoint", tag="GUI")

            camera = packet.current_frame

            # add current camera
            w2c = getWorld2View(camera.R, camera.T).cpu().numpy()
            c2w = np.linalg.inv(w2c)
            name = "current"
            if self.current_frame_frustum is None:
                frustum = self.add_camera(
                    c2w,
                    name=name,
                    color=[0, 1, 0],
                )
                self.current_frame_frustum = frustum
                # add new camera frustum to list
                self.combo_kf.add_item(name)
            else:
                # update pose
                self.current_frame_frustum.update_pose(c2w)
                self.widget3d.scene.set_geometry_transform(name, c2w.astype(np.float64))

            # if follow camera is checked, update viewpoint
            if self.followcam_chbox.checked:
                viewpoint = (
                    self.current_frame_frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else self.current_frame_frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

            # add current camera gt (if available)
            if camera.R_gt is not None and camera.T_gt is not None:
                w2c = getWorld2View(camera.R_gt, camera.T_gt).cpu().numpy()
                c2w = np.linalg.inv(w2c)
                name = "current_gt"
                if self.current_frame_frustum_gt is None:
                    frustum = self.add_camera(
                        c2w,
                        name="current_gt",
                        color=[1, 0, 0],
                    )
                    self.current_frame_frustum_gt = frustum
                    # add new camera frustum to list
                    self.combo_kf.add_item(name)
                else:
                    # update pose
                    self.current_frame_frustum_gt.update_pose(c2w)
                    self.widget3d.scene.set_geometry_transform(
                        name, c2w.astype(np.float64)
                    )

        # whole sequence viewpoints subset
        if packet.viewpoints is not None:

            # Log("Received additional viewpoints", tag="GUI")

            for _, viewpoint in packet.viewpoints.items():

                w2c = getWorld2View(viewpoint.R, viewpoint.T).cpu().numpy()
                c2w = np.linalg.inv(w2c)
                name = "viewpoint_{}".format(viewpoint.frame_idx)

                color = [0, 0, 1]
                if packet.kf_window is not None:
                    if viewpoint.frame_idx in packet.kf_window:
                        # viewpoint is a keyframe
                        color = [1, 1, 0]

                # check if frustum already exists
                if name not in self.frustum_dict.keys():

                    frustum = self.add_camera(
                        c2w,
                        name=name,
                        color=color,
                    )

                    # add to dictionary
                    self.frustum_dict[name] = frustum
                    # add new camera frustum to list
                    self.combo_kf.add_item(name)

                else:

                    # update color and position
                    frustum = self.frustum_dict[name]
                    self.update_camera(name, frustum, c2w, color=color)

        if packet.gt_rgb is not None:
            # Log("Received ground truth image", tag="GUI")
            rgb = packet.gt_rgb  # 3xHxW
            rgb = torch.clamp(rgb, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0)  # HxWx3
            rgb = rgb.contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if packet.gt_depth is not None:
            # Log("Received ground truth depth", tag="GUI")
            depth = packet.gt_depth.squeeze(0)  # torch, float32, HxW
            min_value = 0
            max_value = depth.max().item()
            # normalize depth
            if max_value != min_value:
                depth = (depth - min_value) / (max_value - min_value)
            # convert to numpy
            depth = depth.cpu().numpy()
            # get the color map
            color_map = plt.colormaps.get_cmap("jet")
            # apply the colormap
            depth = color_map(depth)[..., :3] * 255
            # convert to uint8
            depth = depth.astype(np.uint8)
            # convert to C-contiguous memory layout
            depth = np.ascontiguousarray(depth)
            # convert to o3d image
            depth = o3d.geometry.Image(depth)
            # update widget
            self.in_depth_widget.update_image(depth)
            
        if packet.gt_segments is not None:
            # Log("Received ground truth segmentation", tag="GUI")
            segments = packet.gt_segments.squeeze(0)  # torch, int64, HxW
            # get colors from self.segments_colors
            rgb = self.segments_colors[segments] * 255 # HxWx3
            # convert to uint8
            rgb = rgb.byte()
            # convert to C-contiguous memory layout
            rgb = rgb.contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_segments_widget.update_image(rgb)

        if packet.finish:
            Log("Received terminate signal", tag="GUI")
            self.process_finished = True

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    def render_o3d_image(self, results, current_cam, cam_intrinsics):

        if self.selected_shader == "rgb":

            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            render_img = o3d.geometry.Image(rgb)

        elif self.selected_shader == "depth":

            depth = results["depth"][0].detach()  # torch, float32, HxW
            min_value = 0
            max_value = depth.max().item()
            # normalize depth
            if max_value != min_value:
                depth = (depth - min_value) / (max_value - min_value)
            # convert to numpy
            depth = depth.cpu().numpy()
            # get the color map
            color_map = plt.colormaps.get_cmap("jet")
            # apply the colormap
            depth = color_map(depth)[..., :3] * 255
            # convert to uint8
            depth = depth.astype(np.uint8)
            # convert to C-contiguous memory layout
            depth = np.ascontiguousarray(depth)
            # convert to o3d image
            render_img = o3d.geometry.Image(depth)

        elif self.selected_shader == "time":

            opacity = results["opacity"][0].detach()  # torch, float32, 1xHxW
            # convert to numpy
            opacity = opacity.cpu().numpy()
            # get the color map
            color_map = plt.colormaps.get_cmap("jet")
            # apply the colormap
            opacity = color_map(opacity)[..., :3] * 255
            # convert to uint8
            opacity = opacity.astype(np.uint8)
            # convert to C-contiguous memory layout
            opacity = np.ascontiguousarray(opacity)
            # convert to o3d image
            render_img = o3d.geometry.Image(opacity)

        elif self.selected_shader == "elipsoids":

            glfw.poll_events()
            gl.glClearColor(0, 0, 0, 1.0)
            gl.glClear(
                gl.GL_COLOR_BUFFER_BIT
                | gl.GL_DEPTH_BUFFER_BIT
                | gl.GL_STENCIL_BUFFER_BIT
            )

            w2c = getWorld2View(current_cam.R, current_cam.T)

            w2c = w2c.cpu().numpy()
            c2w = np.linalg.inv(w2c)
            # get intrinsics from camera
            fx = cam_intrinsics.fx.detach().cpu().numpy()
            fy = cam_intrinsics.fy.detach().cpu().numpy()
            cx = cam_intrinsics.cx
            cy = cam_intrinsics.cy
            H = cam_intrinsics.height
            W = cam_intrinsics.width
            frustum = create_frustum(c2w, H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy)

            glfw.set_window_size(self.window_gl, W, H)
            self.camera_gl.fovy = cam_intrinsics.FoVy
            self.camera_gl.update_resolution(H, W)
            self.renderer_gl.set_render_reso(W, H)

            self.camera_gl.position = frustum.eye.astype(np.float32)
            self.camera_gl.target = frustum.center.astype(np.float32)
            self.camera_gl.up = frustum.up.astype(np.float32)

            self.gaussians_gl = GaussianData(
                self.gaussians["means"].cpu().numpy(),
                self.gaussians["rotations"].cpu().numpy(),
                self.gaussians["scales"].cpu().numpy(),
                self.gaussians["opacity"].cpu().numpy(),
                self.gaussians["features"].cpu().numpy()[:, 0, :],
            )
            self.update_activated_renderer_state(self.gaussians_gl)
            self.renderer_gl.sort_and_update(self.camera_gl)
            width, height = glfw.get_framebuffer_size(self.window_gl)
            self.renderer_gl.draw()
            bufferdata = gl.glReadPixels(
                0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
            )
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            img = cv2.flip(img, 0)
            render_img = o3d.geometry.Image(img)
            glfw.swap_buffers(self.window_gl)

        elif self.selected_shader == "segmentation":

            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            render_img = o3d.geometry.Image(rgb)

        else:

            raise Exception("Unknown shader selected")

        return render_img

    def rasterise(self, current_cam, cam_intrinsics):
        features = self.gaussians["features"]

        if self.selected_shader == "segmentation":
            # replace gaussians color for object color
            features = self.segments_colors[self.gaussians["ids"]]
        elif self.selected_shader == "time":
            kf_ids = self.gaussians["unique_kfIDs"].float()
            rgb_kf = imgviz.depth2rgb(
                kf_ids.view(-1, 1).cpu().numpy(), colormap="jet", dtype=np.float32
            )
            alpha = 0.1
            features = alpha * features + (1 - alpha) * torch.from_numpy(rgb_kf).to(
                features.device
            )
            
        # print("features", features.shape, features.min(), features.max())

        # render the scene
        rendering_data = render(
            current_cam,
            cam_intrinsics,
            # gaussians,
            # self.pipe,
            self.gaussians["means"],
            self.gaussians["rotations"],
            self.gaussians["scales"],
            self.gaussians["opacity"],
            features,
            self.gaussians["active_sh_degree"],
            self.background,
            self.scaling_slider.double_value,
            override_color=None,
        )

        # if (
        #     self.selected_shader == "time"
        #     and self.gaussians is not None
        #     # and type(self.gaussians) == MainToViewerPacket
        # ):
        #     self.gaussians["features"] = features

        # # undo color replacement
        # if selected_shader == "segmentation":
        #     gaussians.get_features = features

        return rendering_data

    def render_gui(self):

        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()
        T = torch.from_numpy(w2c)
        current_cam = CameraExtrinsics.init_from_gui(frame_idx=-1, T=T)
        # TODO: needed?
        current_cam.update_RT(T[0:3, 0:3], T[0:3, 3])

        # get intrinsics from camera
        height = int(self.window.size.height)
        width = int(self.widget3d_width)

        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(vfov_deg, height, width)
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, width)
        fy = fov2focal(FoVy, height)
        cx = width // 2
        cy = height // 2
        cam_intrinsics = CameraIntrinsics.init_from_gui(fx, fy, cx, cy, height, width)

        if self.gaussians is None:
            return

        # render gaussians
        results = self.rasterise(current_cam, cam_intrinsics)

        if results is None:
            # no data to render
            return

        self.render_img = self.render_o3d_image(results, current_cam, cam_intrinsics)
        self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)

    def scene_update(self):
        self.receive_data(self.q_main2vis)
        self.render_gui()

    def _update_thread(self):
        while True:

            time.sleep(0.01)

            # received terminate signal
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                Log("Closing Visualization", tag="GUI")
                break

            def update():
                if self.step % 3 == 0:
                    self.scene_update()

                if self.step >= 1e9:
                    self.step = 0

            # self.scene_update()
            gui.Application.instance.post_to_main_thread(self.window, update)


def run(params_gui):
    Log("Running GUI", tag="GUI")
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = Viewer(**params_gui)
    app.run()
