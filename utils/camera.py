import numpy as np

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
#    print((hfov * 180) / np.pi)
#    fov_in_rad = max(hfov, vfov)
    fov_in_rad = hfov
    fov = (fov_in_rad * 180) / np.pi
    return fov
