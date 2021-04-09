#!/usr/bin/env python3

import os.path as osp
import numpy as np

from load_functions import get_filenames, load_matches


def calc_rmse_for_one_image(keypoints1, keypoints2):
    assert keypoints1.shape == keypoints2.shape, "keypoints shape must be the same"
    assert len(keypoints1) > 0, "expected one or more matches"
    kps1 = keypoints1.reshape(-1,1,2)
    kps2 = keypoints2.reshape(-1,1,2)
    sum_rxy = np.sum((kps1 - kps2)**2)
    rmse = (sum_rxy / kps1.shape[0])**(1./2)
    return rmse


def _main(matches_dir):
    filename_list = get_filenames(matches_dir)
    no_matches_list = []
    rmse_list = []
    for matches_filename in filename_list:
        image_filename = osp.splitext(matches_filename)[0]
        opt_kps, sar_kps = load_matches(osp.join(matches_dir, matches_filename))
        if len(opt_kps) == 0:
            no_matches_list.append(matches_filename)
            continue
        rmse_list.append(calc_rmse_for_one_image(opt_kps, sar_kps))

    if no_matches_list != []:
        print('Warning: {} files with no matches:\n{}'.format(len(no_matches_list), no_matches_list))
    print('mean rmse = {:5.2f}\n'.format(np.mean(np.asarray(rmse_list))))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='calculating matching accuracy')
    parser.add_argument("matches_dir", help="dir with keypoints matches")
    args = parser.parse_args()
    _main(**vars(args))