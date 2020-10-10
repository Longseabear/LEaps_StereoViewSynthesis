import matplotlib.pyplot as plt
import torch
from dataloader.sceneflow_dataloader import *
from dataloader.ldi import *
from config.config_utils import *
import utils.geometry as geometry_helper
import numpy as np

left, right, left_disp, right_disp = sceneflow_dataloader.read_stereo("source/sceneflow/image/left/0006.png"
                                                          ,"source/sceneflow/image/right/0006.png"
                                                          , "source/sceneflow/depth/left/0006.pfm"
                                                          , "source/sceneflow/depth/right/0006.pfm")

config = Config.from_yaml('config/train.yaml')

R, T = geometry_helper.get_identity_rotation(), geometry_helper.get_identity_transform()

H,W = left.shape[:2]
layered_depth_image = LDI.make_LDI_from_config(config.LDI)
layered_depth_image.set_mesh_from_image(right, right_disp)
layered_depth_image.set_render_infos(disp_threshold=config.disp_threshold)
layered_depth_image.merge_mesh_from_image(right, right_disp)

plt.title('left origin')
plt.imshow(left)
plt.show()

img = layered_depth_image.render(R, T)
plt.title('virtualView identity')
plt.imshow(img)
plt.show()

img = layered_depth_image.render(R, geometry_helper.get_transform_matrix_from_list([-1,0,0]))
plt.title('virtualView transform 1')
plt.imshow(img)
plt.show()

plt.title('right origin')
plt.imshow(right)
plt.show()
