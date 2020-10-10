import matplotlib.pyplot as plt
import torch
from dataloader.sceneflow_dataloader import *
from dataloader.ldi import *
import numpy as np
import utils.geometry as geometry_helper

p = [1,102,3]
left, right, left_disp = sceneflow_dataloader.read_stereo("source/sceneflow/image/left/0006.png"
                                                          ,"source/sceneflow/image/right/0006.png"
                                                          , "source/sceneflow/depth/left/0006.pfm")
sceneflow_cam = sceneflow_dataloader([], 1)

print(sceneflow_cam.intrinsic_parameter)
print(sceneflow_cam.intrinsic_parameter_inv)
cam_coor = geometry_helper.pix2cam(p, sceneflow_cam.intrinsic_parameter_inv)
origin = geometry_helper.cam2pix(cam_coor, sceneflow_cam.intrinsic_parameter)
print(p)
print(cam_coor)
print(origin)
