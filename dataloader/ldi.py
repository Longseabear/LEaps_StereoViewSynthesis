import numpy as np
import networkx as netx
from vispy import scene, io
from vispy.scene import visuals
from vispy.visuals.filters import Alpha
import networkx
import utils.geometry as geometry_helper
import utils.mesh as mesh_helper
import utils.camera as camera_helper
import transform3d
from importlib import import_module
from utils.utils import *

class VirtualCamera():
    def __init__(self, size, K, R, T):
        h,w = size[:2]
        self.size = size

        self.canvas = scene.SceneCanvas(bgcolor='black', size=(w, h))
        self.view = self.canvas.central_widget.add_view()

        self.view.camera = 'perspective'
        self.transform = self.view.camera.transform
        self.view.camera.fov = camera_helper.calculate_fov(K, self.size)

        self.mesh = visuals.Mesh(shading=None)
        self.view.add(self.mesh)

        # camera intrinsic parameter & extrinsic parameter
        self.K = K
        self.R = R
        self.T = T

        self.focal_length_pix = self.K[0,0]

    def view_changed(self):
        return self.view.camera.view_changed

    def get_image(self):
        return self.canvas.render()

    def set_fov(self, fov):
        self.view.camera.fov = fov

    def set_fov_from_int_mat(self, int_mat):
        self.view.camera.fov = camera_helper.calculate_fov(int_mat, self.size)

    def rotate(self, axis=[1,0,0], angle=0):
        self.transform.rotate(axis=axis, angle=angle)

    def translate(self, trans=[0,0,0]):
        self.transform.translate(trans)

class LDI():
    @staticmethod
    def merge_two_ldi(config, lid_1, lid_2):
        ldi = LDI.make_LDI_from_config(config)
        ldi.mesh = networkx.compose(lid_1.mesh, lid_2.mesh)
        return ldi

    @staticmethod
    def make_LDI_from_config(config):
        ldi = LDI()
        ldi.config = config
        size = config.size
        rendering_size = config.rendering_size

        ldi.mesh = netx.Graph(H=size[0], W=size[1])
        _, ldi.dataset_camera = get_instance_from_name(config.camera.camera_module,
                                                    config.camera.camera_class,
                                                    config.camera)
        ldi.virtual_camera = VirtualCamera(rendering_size, K=ldi.dataset_camera.intrinsic_parameter, R=np.eye(3), T=np.array([[0], [0], [0]]))
        ldi.pixel_layer_nums = np.zeros(size).astype(np.long)
        return ldi

    def __init__(self):
        self.config = None
        self.mesh = None
        self.info_pixel = {}
        self.pixel_layer_nums = None
        self.dataset_camera = None
        self.virtual_camera = None

    def merge_mesh_from_image(self, img, disp, threshold):
       self.mesh = mesh_helper.merge_mesh(self,img,disp,threshold)

    def disunite_discontinuities(self, threshold):
        self.mesh = mesh_helper.disunite_discontinuities(self, threshold)

    def set_mesh_from_image(self, image, disp):
        self.mesh = mesh_helper.init_mesh(self, image, disp, self.dataset_camera.baseline_mm)

    def get_inv_K(self):
        return np.linalg.inv(self.virtual_camera.K)

    def get_K(self):
        return self.virtual_camera.K

    def set_render_infos(self, mesh_b=None):
        """
        Set vertices, faces and colors.
        """

        if mesh_b is not None:
            vertices, colors, faces = mesh_helper.mesh_to_render_points_for_stereo(self.mesh, mesh_b, self.get_K(),
                                 self.get_inv_K())
        else:
            vertices, colors, faces = mesh_helper.mesh_to_render_points(self.mesh, self.get_K(), self.get_inv_K())

        vertice = np.array(vertices).astype(np.float32)
        colors = np.array(colors).astype(np.float32)
        faces = np.array(faces).astype(np.long)

        mesh = visuals.Mesh(shading=None)
        mesh.set_data(vertices=vertice, faces=faces, vertex_colors=colors)
        mesh.attach(Alpha(1.0))
        self.virtual_camera.view.add(mesh)

    def render(self, R, T):
        axis, angle = geometry_helper.extrinsic_matrix_to_axis_angle(R)
        self.virtual_camera.rotate(axis, angle)
        self.virtual_camera.translate(T.transpose())
        self.virtual_camera.view_changed()

        return self.virtual_camera.canvas.render()
