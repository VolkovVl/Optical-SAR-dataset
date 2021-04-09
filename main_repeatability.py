#!/usr/bin/env python3

import os.path as osp
from tqdm import tqdm
import numpy as np

from load_functions import read_json_data, get_filenames
from kp_detection import calc_repeatability


def convert_json_kps_to_array(kps_json):
    kps_array = np.zeros((len(kps_json), 1, 2))
    for idx, kp_dict in enumerate(kps_json):
        kps_array[idx,0,0] = kp_dict['x']
        kps_array[idx,0,1] = kp_dict['y']
    return kps_array


def _main(kps_dir, theta, mode):
    filenames_list = get_filenames(osp.join(kps_dir, 'opt'))
    repeatable_kps_list = []
    repeatability_list = []
    for filename in tqdm(filenames_list):
        # read first json (opt)
        opt_path = osp.join(kps_dir, 'opt', filename)
        opt_kps_json = read_json_data(opt_path)
        opt_kps_array = convert_json_kps_to_array(opt_kps_json)
        # read second json (sar)
        sar_path = osp.join(kps_dir, 'sar', filename)
        sar_kps_json = read_json_data(sar_path)
        sar_kps_array = convert_json_kps_to_array(sar_kps_json)

        num_repeatable_kps, repeatability = calc_repeatability(opt_kps_array, sar_kps_array, thresh=theta, mode=mode)
        repeatable_kps_list.append(num_repeatable_kps)
        repeatability_list.append(repeatability)
    print('Number of repeatable keypoints = {:5.1f}; Percent of repeatable keypoints = {:5.1%}'.format(np.mean(np.asarray(repeatable_kps_list)), np.mean(np.asarray(repeatability_list))))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='repeatability')
    parser.add_argument("kps_dir", help="dir with .json keypoints")
    parser.add_argument("theta", type=float, help="distance threshold")
    parser.add_argument("--mode", "-m", type=str, default='all', help="repeatable keypoints: 'all' or 'nearest'")
    args = parser.parse_args()
    _main(**vars(args))