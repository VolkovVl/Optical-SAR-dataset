import json
import glob
import os.path as osp


def read_json_data(filepath):
    with open(filepath, 'r') as json_data:
        data = json.load(json_data)
    return data


def get_opt_filenames(data_dir, file_ext=None):
    opt_dir = osp.join(data_dir, 'opt')
    if file_ext is None:
        opt_filepath_list = glob.glob(opt_dir + '/*')
    else:
        opt_filepath_list = glob.glob(opt_dir + '/*.' + file_ext)
    opt_filename_list = [osp.basename(p) for p in opt_filepath_list if osp.isfile(p)]
    opt_filename_list.sort()
    return opt_filename_list
