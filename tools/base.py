import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from seg.runners import BaseRunner
from seg.utils.config import file_to_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='C:\mycode\mycode\seg\config\\base.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.cfg
    cfg = file_to_config(cfg_path)
    common_cfg = cfg['common']
    base_runner = BaseRunner(common_cfg)


if __name__ == '__main__':
    main()