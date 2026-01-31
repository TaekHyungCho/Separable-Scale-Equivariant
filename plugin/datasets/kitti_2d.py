import copy
import os.path as osp
import mmengine
from typing import Any, List, Union

from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset

@DATASETS.register_module()
class Kitti2DDataset(BaseDetDataset):

    metainfo = {
        'classes' : ('Car','Pedestrian','Cyclist')
    }

    def __init__(self,
                ann_file,
                data_root,
                data_prefix,
                pipeline,
                backend_args = None,
                test_mode = False):
        
        metainfo = {
        'classes' : ('Car','Pedestrian','Cyclist')
        }
        
        super().__init__(
            ann_file = ann_file,
            metainfo = metainfo,
            data_root = data_root,
            data_prefix = data_prefix,
            pipeline = pipeline,
            test_mode = test_mode,
            backend_args= backend_args
            #lazy_init = True
        )
    
    def full_init(self):

        if self._fully_initialized:
            return
        
        self.metainfo, self.data_list = self.load_data_list()
        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def load_data_list(self):
        
        raw_data_info = mmengine.load(self.ann_file)
        metainfo = raw_data_info['meatainfo']
        data_list = raw_data_info['data_list']
        return metainfo, data_list

    def prepare_data(self, idx):

        data = self.get_data_info(idx)
        return self.pipeline(data)


    
        
