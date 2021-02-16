#!/usr/bin/env python3

import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2

from load_functions import load_matches, get_filenames


def check_homography(H_mat, img_size, thresh):
    width = img_size[0]
    height = img_size[1]
    corners = np.array([[[0,0]],
                        [[0,height]],
                        [[width, height]],
                        [[width, 0]]])
    dst_corners = apply_matrix_to_kps(kp=corners, matrix=H_mat)
    dists = np.linalg.norm((corners - dst_corners), axis=2)
    max_dist = np.max(dists)
    if np.all(dists <= thresh):
        return True, max_dist
    else:
        return False, max_dist


def apply_matrix_to_kps(kp, matrix):
    # kp -> array
    kp = np.copy(kp)
    kp = kp.reshape(-1, 2, 1)
    kp = np.hstack((kp, np.ones((kp.shape[0], 1, 1))))
    kp_new = np.zeros_like(kp)
    for i, keypoint in enumerate(kp):
        kp_new[i] = matrix.dot(keypoint)
    # Generalized coordinates:
    kp_new[:,0,:] /= kp_new[:,2,:]
    kp_new[:,1,:] /= kp_new[:,2,:]
    kp_new = kp_new[:,:2,:].reshape(-1,1,2)
    return kp_new


def _main(matches_dir, image_size, ransac_thr, dists_thr):
    filename_list = get_filenames(matches_dir)
    false_pair_list = []
    true_pair_list = []
    for matches_filename in tqdm(filename_list):
        image_filename = osp.splitext(matches_filename)[0]
        src_kps, dst_kps = load_matches(osp.join(matches_dir, matches_filename))
        if len(src_kps) == 0:
            false_pair_list.append(image_filename)
            continue

        M, mask = cv2.findHomography(src_kps, dst_kps, cv2.RANSAC, ransac_thr)
        if M is None:
            false_pair_list.append(image_filename)
            continue

        res, max_dist = check_homography(H_mat=M, img_size=image_size, thresh=dists_thr)
        if res:
            true_pair_list.append(image_filename)
        else:
            false_pair_list.append(image_filename)

    quality = len(true_pair_list) / (len(true_pair_list) + len(false_pair_list))
    print('quality = {}/{}'.format(len(true_pair_list), len(true_pair_list) + len(false_pair_list)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='repeatability')
    parser.add_argument("matches_dir", help="dir with .txt files")
    parser.add_argument("image_size", type=int, nargs=2, help="image size in format [width, height]")
    parser.add_argument("--ransac_thr", "-r", type=float, default=2, help="threshold for RANSAC")
    parser.add_argument("--dists_thr", "-t", type=float, default=5, help="threshold for successful image match")
    args = parser.parse_args()
    _main(**vars(args))