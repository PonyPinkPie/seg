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
    parser.add_argument('--cfg', type=str, default='/workspace/mycode/03-seg/seg/workdir/test/20241216_151647/20241216_151647_trt.json')
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
    trt_runner = TRTRunner(cfg)
    image = cv2.imread(image_path)
    h, w, c = image.shape
    images = [image]
    result = trt_runner(images)
    image_copy = image.copy()
    save_json(result, jp)
    info_dict = result[0]
    for cls, item in info_dict.items():
        for idx, item_info in item.items():
            contours, area, score = item_info['contour'], item_info['area'], item_info['score']

            text = f"{cls}_{area}_{score}"
            org = (int(contours[0][0]), int(contours[0][1]))
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)  # BGR格式，白色
            thickness = 1

            contours = np.array(contours, dtype=np.int32)[:, None, :]
            cv2.drawContours(image_copy, [contours], 0, (0, 0, 255), thickness=2)
            cv2.putText(image_copy, text, org, fontFace, fontScale, color, thickness)

    cv2.imwrite(vis_path, np.hstack([image, np.ones((h, 10, c))*255, image_copy]))


if __name__ == '__main__':
    main()