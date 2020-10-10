import numpy as np
import utils.geometry as geometry_helper

class Camera():
    def __init__(self, name, R, T, scale=1):
        self.name = name

        self.R = R
        self.T = T
        self.extrinsic_param = geometry_helper.Rt_matrix(R,T)

    def get_Rt(self):
        return self.R, self.T


def get_translate_matrix(vec):
    """
    :param array:
    :return: 3x1 transform matrix
    """
    return np.array(vec).transpose()


def calculate_fov(int_mat, size):
    """

    @param int_mat: camera intrinsic parameter
    @param size: resolution
    @return:
    """
    h, w = size
    #print(w, int_mat[0,0])
    vfov = 2 * np.arctan((1. / 2.) * (w / int_mat[0, 0]))
    hfov = 2 * np.arctan((1. / 2.) * (h / int_mat[1, 1]))
    fov_in_rad = hfov
    fov = (fov_in_rad * 180) / np.pi
    return fov
