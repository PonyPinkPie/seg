import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from seg.runners import TrainRunner
from seg.utils.config import file_to_config
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='/workspace/mycode/03-seg/seg/config/train.json')
    # parser.add_argument('--cfg', type=str, default='/workspace/mycode/03-seg/seg/config/train-v1.json')
    # parser.add_argument('--cfg', type=str, default='/workspace/mycode/03-seg/seg/config/transform-test.json')
    parser.add_argument('--cfg', type=str, default='./config/labelme.json')
    # parser.add_argument('--cfg', type=str, default='C:\mycode\mycode\seg\config\\train_local.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.cfg
    try:
        cfg = file_to_config(cfg_path)
    except Exception as e:
        raise print(f"load json error, please check {cfg_path} format, especially like redundant ',' at the end. {e} \n{traceback.format_exc()}")
    common_cfg = cfg['common']
    train_cfg = cfg['data']
    export_cfg = cfg['export']
    train_runner = TrainRunner(export_cfg, train_cfg, common_cfg)
    train_runner()


if __name__ == '__main__':
    main()
