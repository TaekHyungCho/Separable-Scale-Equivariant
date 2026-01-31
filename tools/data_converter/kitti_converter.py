from collections import OrderedDict
from pathlib import Path
from mmengine.fileio import dump
import mmcv
import numpy as np
import sys
import os
import argparse
sys.path.append(os.getcwd())
from kitti_data_utils import get_kitti_image_info

kitti_categories = ('Pedestrian', 'Cyclist', 'Car')

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line.split('.')[0]) for line in lines]

def create_kitti_info_file(data_path,
                           save_path,
                           pkl_prefix ='kitti',
                           relative_path=True):
    
    train_img_ids = _read_imageset_file(data_path + 'train.txt')
    val_img_ids = _read_imageset_file(data_path + 'val.txt')
    save_path = Path(save_path)

    kitti_infos_train = get_kitti_image_info(
        data_path,
        training=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train_official_m3.pkl'
    print(f'Kitti info train file is saved to {filename}')
    dump(kitti_infos_train, filename)

    kitti_infos_val = get_kitti_image_info(
        data_path,
        training=False,
        image_ids=val_img_ids,
        relative_path=relative_path)
    
    filename = save_path / f'{pkl_prefix}_infos_val_official_m3.pkl'
    print(f'Kitti info val file is saved to {filename}')
    dump(kitti_infos_val, filename)

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='KITTI Data generation parser')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Specify the path where the raw data is located')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Specify the path to save the generated pickle file')
    
    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path
    create_kitti_info_file(data_path,save_path)