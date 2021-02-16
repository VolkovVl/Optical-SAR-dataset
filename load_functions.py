import json
import glob
import os.path as osp
import numpy as np


def read_json_data(filepath):
    with open(filepath, 'r') as json_data:
        data = json.load(json_data)
    return data


def get_filenames(data_dir, file_ext=None):
    if file_ext is None:
        filepath_list = glob.glob(data_dir + '/*')
    else:
        filepath_list = glob.glob(data_dir + '/*.' + file_ext)
    filename_list = [osp.basename(p) for p in filepath_list if osp.isfile(p)]
    filename_list.sort()
    return filename_list


def load_matches(filepath):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.loadtxt(filepath, delimiter='\t', skiprows=1).reshape(-1,4)
    if data.size == 0:
        return [], []
    opt_kps = data[:, :2].reshape(-1,1,2)
    sar_kps = data[:, 2:4].reshape(-1,1,2)
    return opt_kps, sar_kps