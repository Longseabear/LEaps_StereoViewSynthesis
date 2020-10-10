import matplotlib.pyplot as plt
import torch
from dataloader.sceneflow_dataset import *
from dataloader.ldi import *
import numpy as np

left, right, left_disp = sceneflow_dataset.read_stereo("source/sceneflow/image/left/0006.png"
                              ,"source/sceneflow/image/right/0006.png"
                              ,"source/sceneflow/depth/0006.pfm")

sceneflow_cam = sceneflow_dataset([],1)
R,T = sceneflow_cam.left_camera.get_Rt()

H,W = left.shape[:2]
layered_depth_image = LDI((H,W), sceneflow_cam.intrinsic_parameter, R, T, left, left_disp)
layered_depth_image.set_render_infos()
img = layered_depth_image.render(sceneflow_cam.left_camera.R, sceneflow_cam.left_camera.T)
plt.imshow(img)
plt.show()
print(sceneflow_cam.left_camera.T)
img = layered_depth_image.render(sceneflow_cam.left_camera.R, sceneflow_cam.left_camera.T)
plt.imshow(img)
plt.show()
plt.imshow(left)
plt.show()
