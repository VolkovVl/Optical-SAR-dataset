#!/usr/bin/env python3

import os.path as osp
import numpy as np

from load_functions import get_filenames


def check_homography(H_mat, img_size, thresh):
    width = int(img_size[0])
    height = int(img_size[1])
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


def _main(homography_dir, image_size, dists_thr):
    assert osp.isdir(homography_dir), "Not founded dir: {}".format(homography_dir)
    false_pair_list = []
    true_pair_list = []
    filename_list = get_filenames(homography_dir)
    for homography_filename in filename_list:
        image_filename = osp.splitext(homography_filename)[0]
        filepath = osp.join(homography_dir, homography_filename)
        H = np.loadtxt(filepath, delimiter='\t')
        if np.array_equal(H, np.zeros((3,3))):
            false_pair_list.append(image_filename)
            continue

        res, max_dist = check_homography(H_mat=H, img_size=image_size, thresh=dists_thr)
        if res:
            true_pair_list.append(image_filename)
        else:
            false_pair_list.append(image_filename)
    print('Successfully registered images = {}/{}'.format(len(true_pair_list), len(true_pair_list) + len(false_pair_list)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='calculating image registration accuracy')
    parser.add_argument("homography_dir", help="dir with .txt homography files")
    parser.add_argument("--image_size", "-i", type=int, nargs=2, default=["1024", "1024"], help="image size in format [width, height]")
    parser.add_argument("--dists_thr", "-t", type=float, default=10, help="threshold for successful image match")
    args = parser.parse_args()
    _main(**vars(args))