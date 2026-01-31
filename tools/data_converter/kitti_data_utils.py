from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np
from PIL import Image
from skimage import io

kitti_categories = ['Pedestrian', 'Cyclist', 'Car']


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)

def get_kitti_info_path(idx,
                        prefix,
                        info_type='images',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx,use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    file_path = info_type+'/'+img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='images',
                   file_tail='.png',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)

def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='labels',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool_)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool_)
    hard_mask = np.ones((len(dims), ), dtype=np.bool_)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff

def get_kitti_image_info(path,
                        training=True,
                        label_info=True,
                        image_ids=7481,
                        num_worker=8,
                        relative_path=True,
                        with_imageshape=True):
    
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}

        image_info = {'image_idx': idx}
        annotations = None
        image_info['image_path'] = get_image_path(idx, path, training,
                                                  relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
                image_info['image_path'] = img_path
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        
        if annotations is not None:
            info['annos'] = annotations
            #add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    if training:
        image_infos = filter_info_training(list(image_infos))
    else:
        image_infos = filter_info_validation(list(image_infos))
    
    return image_infos

def filter_info_validation(image_infos):

    new_image_info = {
        'meatainfo' : {
            'classes' : ('Car','Pedestrian','Cyclist')
        },
        'data_list' : []
    }
    kitti_eval_gt = dict()
    for info in image_infos:

        data_info = dict()
        data_info['img_path'] = info['image']['image_path']
        data_info['height'] = info['image']['image_shape'][0]
        data_info['width'] = info['image']['image_shape'][1]
        instances = []
        
        key = info['image']['image_path'].split('/')[-1][:6]
        kitti_eval_gt[key]=info['annos']
        # num_gt = len(info['annos']['name'])
        # for idx in range(num_gt):
        #     instance = dict()
        #     instance['name'] = info['annos']['name'][idx]
        #     instance['truncated'] = info['annos']['truncated'][idx]
        #     instance['occluded'] = info['annos']['occluded'][idx]
        #     instance['alpha'] = info['annos']['alpha'][idx]
        #     instance['bbox'] = info['annos']['bbox'][idx]
        #     instance['dimensions'] = info['annos']['dimensions'][idx]
        #     instance['location'] = info['annos']['location'][idx]
        #     instance['rotation_y'] = info['annos']['rotation_y'][idx]
        #     instance['ignore_flag'] = 0
        #     instances.append(instance)

        data_info['instances'] = instances
        new_image_info['data_list'].append(data_info)
    mmengine.dump(kitti_eval_gt,'kitti_eval_gt.pkl')
    return new_image_info

def filter_info_training(image_infos):

    new_image_info = {
        'meatainfo' : {
            'classes' : ('Car','Pedestrian','Cyclist')
        },
        'data_list' : []
    }

    for info in image_infos:
        mask = np.isin(info['annos']['name'],kitti_categories)
        if not np.any(mask):
            continue

        def cat2id(arr):
            kitti_classes = {
                'Car': 0,
                'Pedestrian': 1,
                'Cyclist': 2,
                'Van' : 0,
                'Person_sitting' : 1
            }
            gt_label = []
            for val in arr:
                gt_label.append(kitti_classes[val])
            return np.array(gt_label)
        
        data_info = dict()
        data_info['img_path'] = info['image']['image_path']
        data_info['height'] = info['image']['image_shape'][0]
        data_info['width'] = info['image']['image_shape'][1]
        instances = []

        for bboxes,labels in zip(info['annos']['bbox'][mask],info['annos']['name'][mask]):
            bboxes = bboxes.astype(np.float32).reshape(-1,4)
            labels = cat2id([labels])
            for bbox,label in zip(bboxes,labels):
                instance = dict()
                instance['bbox'] = bbox
                instance['bbox_label'] = label
                instance['ignore_flag'] = 0
                instances.append(instance)

        data_info['instances'] = instances
        new_image_info['data_list'].append(data_info)
    return new_image_info