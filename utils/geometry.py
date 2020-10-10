import numpy as np
import transforms3d
import torch

def Rt_matrix(R, t):
    assert R.shape[1] == t.shape[0], "{} and {} must to same".format(R.shape[1],t.shape[0])
    return np.concatenate((R,t),axis=1)

def extrinsic_matrix_to_axis_angle(extrinsic):
    axis, angle = transforms3d.axangles.mat2axangle(extrinsic)
    return axis, angle

def get_identity_transform():
    return np.array([[0], [0], [0]])

def get_transform_matrix_from_list(a):
    return np.array([[a[0]], [a[1]], [a[2]]])

def get_identity_rotation():
    return np.eye(3)

def cam2pix(point, K):
    x,y,z = point
    return [((x/abs(z)) * K[0,0] + K[0,2]), ((y/abs(z)) * -K[1,1] + K[1,2]), 1]

def pix2cam(point, K_inv):
    x, y, z = point
    return [abs(z) * (x * K_inv[0, 0] + K_inv[0, 2]), abs(z) * (y * -K_inv[1, 1] + -K_inv[1, 2]), z]

def pix2cam_backup(point, K_inv):
    x,y,z = point
    return [abs(z) * (x * K_inv[0,0] + K_inv[0,2]), abs(z) * (y * -K_inv[1,1] + -K_inv[1,2]), z]

def transform_points(T, A):
    """
    linear transform T*A.
    if rank of A is 3, then we changed
    @param transform matrix T[out_c, P]
    @param input points A[in_c, P]     
    :return: Output Matrix[out_c, P
    
    example T[3,4](extrinsic) * A[4,P](points) = 3xP matrix
    """
    assert len(T.size())==2
    rank = len(A.size())
    if rank==2:
        return torch.matmul(T,A)
    elif rank==3:
        # FEATURE X Y
        if isinstance(A, np.ndarray):
            feature_num = A.size()[-1]
            A = A.transpose((2,0,1))
            A = A.view(feature_num, -1)
        else:
            feature_num = A.size()[0]
            A = A.view(feature_num, -1)

        return torch.matmul(T,A)
