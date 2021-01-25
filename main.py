#!/usr/bin/env python3

import os.path as osp
from tqdm import tqdm
import numpy as np

from load_functions import read_json_data, get_opt_filenames
from kp_detection import get_matches_NN, get_mask_of_near_matches


def prepare_data(dists, thresholds, num_img):
    num_valid = []
    for thresh in thresholds:
        num_valid.append(np.count_nonzero(dists<=thresh))

    num_valid = np.array(num_valid)
    y_data1 = num_valid / num_img
    y_data2 = 100 * num_valid / len(dists)
    return y_data1, y_data2


def convert_json_kps_to_array(kps_json):
    kps_array = np.zeros((len(kps_json), 1, 2))
    for idx, kp_dict in enumerate(kps_json):
        kps_array[idx,0,0] = kp_dict['x']
        kps_array[idx,0,1] = kp_dict['y']
    return kps_array


def _main(kps_dir, theta, search_radius):
    min_dists_list = []
    opt_filenames_list = get_opt_filenames(kps_dir)
    for filename in tqdm(opt_filenames_list):
        # read first json (opt)
        opt_path = osp.join(kps_dir, 'opt', filename)
        opt_kps_json = read_json_data(opt_path)
        opt_kps_array = convert_json_kps_to_array(opt_kps_json)
        # read second json (sar)
        sar_path = osp.join(kps_dir, 'sar', filename)
        sar_kps_json = read_json_data(sar_path)
        sar_kps_array = convert_json_kps_to_array(sar_kps_json)

        matches_mask = get_mask_of_near_matches(kps_array1=opt_kps_array, kps_array2=sar_kps_array, searchRad=search_radius)
        min_dists = get_matches_NN(opt_kps_array, sar_kps_array, matches_mask=matches_mask)[1]
        min_dists_list.extend(min_dists)

    num_img = len(opt_filenames_list)
    dists = np.array(min_dists_list)
    thresholds = [theta]
    data_num, data_per = prepare_data(dists, thresholds, num_img)
    for i, (val_num, val_per) in enumerate(zip(data_num, data_per)):
        print('theta = {}: number of good matches = {}; percent of good matches = {}'.format(thresholds[i], val_num, val_per))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='repeatability')
    parser.add_argument("kps_dir", help="dir with .json keypoints")
    parser.add_argument("theta", type=float, help="distance threshold")
    parser.add_argument("--search_radius", "-r", type=int, default=np.inf, help="search radius in which keypoints are searched")
    args = parser.parse_args()
    _main(**vars(args))