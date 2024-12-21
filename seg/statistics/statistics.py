import os
from os.path import join as opj
from seg.utils.io import IMAGE_POSTFIX, async_execute, map_execute
from functools import partial
import cv2
import numpy as np


def dropout_zero(mat):
    vector = mat.flatten()
    vector = vector[vector > 0]
    return vector


def calc_mean_std_without_zero(image):
    channel_list = [image] if len(image.shape) == 2 else cv2.split(image)
    vector_list = [dropout_zero(mat) for mat in channel_list]
    mean_std = [[v.mean() / 255, v.std() / 255] if v.size > 0 else [v, v] for v in vector_list]
    return mean_std


def calc_means_stds_without_zeros(image_path):
    # image_list = [cv2.imread(ip) for ip in image_path]
    image = cv2.imread(image_path)
    if image is None:
        return None
    return calc_mean_std_without_zero(image)


class ClsStatistics:
    def __init__(self, root, logger):
        self.image_path = [opj(root, i) for i in os.listdir(root) if i.split('.')[-1].upper() in IMAGE_POSTFIX]
        logger.info(f'Calculate mean and stds for train dataset {len(self.image_path)} images')
        pfunc = partial(calc_means_stds_without_zeros)
        mean_std = map_execute(pfunc, (self.image_path,))
        mean_std = np.array(mean_std, dtype=np.float64)
        self.mean = [round(float(i), 3) for i in np.mean(mean_std[:, :, 0], axis=0)]
        self.std = [round(float(i), 3) for i in np.mean(mean_std[:, :, 1], axis=0)]
        logger.info(f'Calculate mean and stds done. mean: {self.mean}, std: {self.std}')