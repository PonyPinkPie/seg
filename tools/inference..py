import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from seg.runners import InferenceRunner
from seg.utils.config import file_to_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='C:\mycode\mycode\seg\config\inference.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.cfg
    cfg = file_to_config(cfg_path)
    common_cfg = cfg['common']
    inference_cfg = cfg['inference']
    inference_runner = InferenceRunner(inference_cfg, common_cfg)
    ip = '/workspace/mycode/03-seg/seg/local/000.png'
    gp = '/workspace/mycode/03-seg/seg/local/000_mask.png'
    image = cv2.imread(ip)
    mask = cv2.imread(gp)
    inference_runner(image, mask)

if __name__ == '__main__':
    main()