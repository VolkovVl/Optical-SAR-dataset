import numpy as np


def calc_repeatability_dists(inp_array1, inp_array2):
    assert inp_array1.shape[1] == 1
    assert inp_array2.shape[1] == 1
    # make inp_array1 as vertical vector, inp_array2 as horizontal vector
    array1 = inp_array1.repeat(inp_array2.shape[0], axis=1)
    array2 = inp_array2.reshape(1, -1, inp_array2.shape[2]).repeat(inp_array1.shape[0], axis=0)
    # calc euclidean distance
    dists = np.linalg.norm((array1 - array2), axis=2)  # [idx1, idx2]: 'idx1' refer to 'inp_array1' idx, 'idx2' refer to 'inp_array2' idx
    return dists


def get_matches_dists(dists, thresh=0):
    assert thresh >= 0
    mask1 = np.zeros_like(dists)
    array1_idxs = np.arange(dists.shape[0])
    array2_idxs = np.argmin(dists, axis=1)
    mask1[array1_idxs, array2_idxs] = True
    
    mask2 = np.zeros_like(dists)
    array1_idxs = np.argmin(dists, axis=0)
    array2_idxs = np.arange(dists.shape[1])
    mask2[array1_idxs, array2_idxs] = True
    
    if thresh == 0:
        mask = mask1 * mask2
    else:
        mask_dists = dists < thresh
        mask = mask1 * mask2 * mask_dists
    
    idx1, idx2 = np.nonzero(mask)
    matches = [(m,n) for (m,n) in zip(idx1, idx2)]
    matches_dists = dists[idx1, idx2]
    return matches, matches_dists


def get_matches_NN(des1, des2, thresh=0, matches_mask=None):
    # return: (i) indexes of matches (opt_idx, sar_idx)
    # (ii) distances between points
    dists = calc_repeatability_dists(des1, des2)
    if matches_mask is not None:
        ind_x, ind_y = np.nonzero(matches_mask == 0)
        dists[ind_x, ind_y] = np.inf
    assert len(dists.shape) == 2, 'Get {}D array, expected 2D array'.format(len(dists.shape))
    
    matches, matches_dists = get_matches_dists(dists, thresh)
    return matches, matches_dists


def get_mask_of_near_matches(kps_array1, kps_array2, searchRad):
    assert kps_array1.shape[1] == 1
    assert kps_array2.shape[1] == 1
    
    mask = np.zeros((kps_array1.shape[0], kps_array2.shape[0]))
    kps1_x = kps_array1[:,:,0].flatten() 
    kps1_y = kps_array1[:,:,1].flatten()
    kps2_x = kps_array2[:,:,0].flatten()
    kps2_y = kps_array2[:,:,1].flatten()
    for i in range(kps_array1.shape[0]):
        cur_x = kps1_x[i]
        cur_y = kps1_y[i]
        mask1 = (kps2_x >= cur_x - searchRad) * (kps2_x <= cur_x + searchRad)              
        mask2 = (kps2_y >= cur_y - searchRad) * (kps2_y <= cur_y + searchRad)
        idxs = np.flatnonzero(mask1 * mask2)
        mask[np.repeat(i, idxs.shape[0]), idxs] = 1
    return mask
