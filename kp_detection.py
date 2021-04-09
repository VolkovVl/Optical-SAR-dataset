import numpy as np


def calc_repeatability(inp_array1, inp_array2, thresh, mode='all'):
    assert thresh >= 0
    assert inp_array1.shape[1] == 1
    assert inp_array2.shape[1] == 1
    assert mode in ['all', 'nearest']

    num_points1 = inp_array1.shape[0]
    num_points2 = inp_array2.shape[0]
    # make inp_array1 as vertical vectors, inp_array2 as horizontal vectors
    array1 = inp_array1.repeat(num_points2, axis=1)
    array2 = inp_array2.reshape(1, -1, inp_array2.shape[2]).repeat(num_points1, axis=0)
    dists = np.linalg.norm((array1 - array2), axis=2)  # [idx1, idx2]: 'idx1' refer to 'inp_array1' idx, 'idx2' -- to 'inp_array2' idx

    if mode == 'all':
        mask = dists <= thresh
        num_rep_kps1 = np.count_nonzero(np.sum(mask, axis=1))
        num_rep_kps2 = np.count_nonzero(np.sum(mask, axis=0))
        num_rep_kps = num_rep_kps1 + num_rep_kps2
    elif mode == 'nearest':
        mask = calc_mask_of_nearest_points(dists, thresh)
        num_rep_kps = 2 * np.count_nonzero(mask)  # one pair of points give 2 unique repeatable points
    repeatability = num_rep_kps / (num_points1 + num_points2)
    return num_rep_kps, repeatability


def calc_mask_of_nearest_points(dists, thresh):
    assert thresh >= 0
    mask1 = np.zeros_like(dists)
    array1_idxs = np.arange(dists.shape[0])
    array2_idxs = np.argmin(dists, axis=1)
    mask1[array1_idxs, array2_idxs] = True

    mask2 = np.zeros_like(dists)
    array1_idxs = np.argmin(dists, axis=0)
    array2_idxs = np.arange(dists.shape[1])
    mask2[array1_idxs, array2_idxs] = True

    mask_dists = dists <= thresh
    mask = mask1 * mask2 * mask_dists
    return mask
