import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from seg.runners import TrainRunner
from seg.utils.config import file_to_config


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='/workspace/mycode/03-seg/seg/config/train.json')
    parser.add_argument('--cfg', type=str, default='C:\mycode\mycode\seg\config\\train_local.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.cfg
    cfg = file_to_config(cfg_path)
    common_cfg = cfg['common']
    inference_cfg = cfg['inference']
    train_cfg = cfg['data']
    train_runner = TrainRunner(train_cfg, inference_cfg, common_cfg)
    train_runner()

if __name__ == '__main__':
    main()