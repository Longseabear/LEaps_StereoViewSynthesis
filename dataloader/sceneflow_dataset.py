from PIL import Image, ImageOps
from skimage.transform import resize
from torchgeometry import core

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
import re
import numpy as np
import sys
import chardet
import glob
import cv2
import matplotlib.pyplot as plt
import utils.geometry as geometry_helper
import utils.mesh as mesh_helper
import utils.camera as camera_helper

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)
    header = header.decode(encode_type['encoding'])
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode(encode_type['encoding']))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return plt.imread(path)

def disparity_loader(path):
    return readPFM(path)


class sceneflow_camera():
    def __init__(self, name, R, T, scale=1):
        self.name = name

        self.R = R
        self.T = T
        self.extrinsic_param = geometry_helper.Rt_matrix(R,T)
        self.focal_length_mm = 35.0
        self.pixel_mm = 0.03333333
        self.sensor_size_mm = (32.00,18.00) # width, height

    def get_Rt(self):
        return self.R, self.T


class sceneflow_dataset():
    def __init__(self, data_list, scale):
        self.intrinsic_parameter = np.array([[1050.0, 0, 479.5], [0, 1050.0, 269.5], [0, 0, 1]]).astype(np.float32)
        self.intrinsic_parameter_inv = np.linalg.inv(self.intrinsic_parameter)

        self.left_camera = sceneflow_camera("Camera:Left", np.eye(3), np.array([[0,0,0]]).transpose(), scale)
        self.right_camera = sceneflow_camera("Camera:Right", np.eye(3), np.array([[1,0,0]]).transpose(), scale)
        self.data_list = data_list

    @staticmethod
    def read_stereo(left_path, right_path, disp_path):
        return default_loader(left_path), default_loader(right_path), sceneflow_dataset.read_disparity(disp_path)

    @staticmethod
    def read_disparity(path):
        dataL, scaleL = disparity_loader(path)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        return dataL

    @staticmethod
    def read_depth(disp_fi, disp_rescale=10., h=None, w=None):
        disp = sceneflow_dataset.read_disparity(disp_fi)
        disp = disp - disp.min()  # 0~disp(max)
        disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
        disp = (disp / disp.max()) * disp_rescale  # 0~disp_rescsale
        if h is not None and w is not None:
            disp = resize(disp / disp.max(), (h, w), order=1) * disp.max()
        depth = 1. / np.maximum(disp, 0.05)  # disparity to depth(baseline 1)
        return depth
