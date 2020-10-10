import numpy as np
import transforms3d

def Rt_matrix(R, t):
    assert R.shape[1] == t.shape[0], "{} and {} must to same".format(R.shape[1],t.shape[0])
    return np.concatenate((R,t),axis=1)

def extrinsic_matrix_to_axis_angle(extrinsic):
    axis, angle = transforms3d.axangles.mat2axangle(extrinsic)
    return axis, angle


def cam2pix(point, K):
    print('cam2pix in geometry error')
    x,y,z = point
    return [((x/abs(z)) * K[0,0] + K[0,2]), ((y/abs(z)) * -K[1,1] + K[1,2]), 1]

def pix2cam(point, K_inv):
    x, y, z = point
    return [abs(z) * (x * K_inv[0, 0] + K_inv[0, 2]), abs(z) * (y * -K_inv[1, 1] + -K_inv[1, 2]), z]

def pix2cam_backup(point, K_inv):
    x,y,z = point
    return [abs(z) * (x * K_inv[0,0] + K_inv[0,2]), abs(z) * (y * -K_inv[1,1] + -K_inv[1,2]), z]
