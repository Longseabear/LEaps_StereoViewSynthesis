from importlib import import_module
import math
from utils.geometry import get_transform_matrix_from_list
def get_instance_from_name(module_path, class_name, config):
    m = import_module(module_path, class_name)
    return m, getattr(m, class_name)(config)

import cv2
import cv2
import numpy as np
import glob

class VideoWriter():
    def __init__(self, output_path=None, fps=5, is_color=True, format="XVID"):
        self.output_path = output_path
        self.fourcc = cv2.VideoWriter_fourcc(*format)
        self.fps = fps
        self.is_color = is_color
        self.out = None

    def write_image(self, image, rgb=True):
        if image.max() <= 1:
            image = image * 255

        if rgb:
            image = (image[:,:,::-1]).astype(np.int8)

        if self.out is None:
            size = image.shape[1], image.shape[0]
            self.out = cv2.VideoWriter(self.output_path, self.fourcc, float(self.fps), size, self.is_color)
        self.out.write(image)

    def finish(self):
        self.out.release()

def make_sacaddes_movement(move_info, max_disparity, baseline):
    """
    :param R:
    :param T:
    :param move_info:
    :param frames: ceil(max_disparity * move_mm/base_line))
    :return:
    """
    start_pos = (0,0)
    R = np.eye(3)
    for info in move_info:
        max_move_mm = max(abs(info[0]-start_pos[0]), abs(info[1]-start_pos[1]))
        dist = (info[0]-start_pos[0], info[1]-start_pos[1])
        frames = int(math.ceil(max_disparity * max_move_mm / baseline))
        for i in range(frames-1):
            yield R, get_transform_matrix_from_list([dist[0]/(frames-1), dist[1]/(frames-1), 0])
        start_pos = info
