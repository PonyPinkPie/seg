import argparse
import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from seg.runners import TRTRunner
from seg.utils import file_to_config, save_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/workspace/mycode/03-seg/seg/config/train.json')
    # parser.add_argument('--cfg', type=str, default='C:\mycode\mycode\seg\config\\train_local.json')
    parser.add_argument('--image_path', type=str, default='/workspace/mycode/03-seg/seg/local/04fa7bde-8a9b-4ce6-a644-7e287a56a8f4.jpg')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.cfg
    image_path = args.image_path
    vis_path = image_path.replace('.jpg', '-vis.jpg')
    jp = image_path.replace('.jpg', '.json')
    cfg = file_to_config(cfg_path)
    trt_cfg = cfg['trt']
    trt_runner = TRTRunner(trt_cfg)
    image = cv2.imread(image_path)
    images = [image] * 2
    result = trt_runner(images)
    save_json(result, jp)
    info_dict = result[0]
    for cls, item in info_dict.items():
        for idx, points in enumerate(item.items()):
            contours = np.array(points, dtype=np.int32)[:, None,:]
            cv2.drawContours(image, [contours], 0, (0, 0, 255), thickness=2)
            cv2.putText(image, f"")


if __name__ == '__main__':
    main()