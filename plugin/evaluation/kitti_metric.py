import numpy as np
import torch
import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import print_log
from mmengine.logging import MMLogger
from mmdet.registry import METRICS
from typing import Any, Dict, List, Optional, Sequence, Union
from plugin.evaluation.kitti_eval import get_official_eval_result
from collections import OrderedDict

@METRICS.register_module()
class KittiMetric(BaseMetric):

    def __init__(self,
                img_size,
                conf_th = None,
                current_classes: List = [0,1,2],
                gt_filename: str = 'kitti_eval_gt.pkl',
                collect_device: str = 'cpu',
                prefix: Optional[str] = 'kitti'):
    
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.img_size = img_size
        self.kitti_categories = ['Car','Pedestrian','Cyclist']
        self.gts = mmengine.load(gt_filename)
        self.results = []
        self.anno_dets = []
        self.anno_gts = []
        self.conf_th = conf_th
        self.current_classes = current_classes

    def process(self, data_batch, data_samples):

        pred_label = data_samples[-1]['pred_instances']['labels'].detach().cpu().numpy()
        pred_bbox = data_samples[-1]['pred_instances']['bboxes'].detach().cpu().numpy()
        pred_score = data_samples[-1]['pred_instances']['scores'].detach().cpu().numpy()

        if self.conf_th is not None:
            score_mask = pred_score >= self.conf_th
            pred_label = pred_label[score_mask]
            pred_bbox = pred_bbox[score_mask]
            pred_score = pred_score[score_mask]

        if len(pred_score) == 0:
            pass
        else:
            if self.img_size is not None:
                factor = data_samples[-1]['scale_factor']
                ori_width = data_samples[-1]['ori_shape'][1]
                ori_height = data_samples[-1]['ori_shape'][0]
                pred_bbox[:,0] = pred_bbox[:,0] * ori_width / (self.img_size[0] / factor[0])
                pred_bbox[:,1] = pred_bbox[:,1] * ori_height / (self.img_size[1] / factor[1])
                pred_bbox[:,2] = pred_bbox[:,2] * ori_width / (self.img_size[0]  / factor[0])
                pred_bbox[:,3] = pred_bbox[:,3] * ori_height / (self.img_size[1] / factor[1])

        annotations = dict()
        annotations['name'] = np.array([self.kitti_categories[int(j)] for j in pred_label])
        annotations['bbox'] = np.array([[float(info) for info in x[0:4]] for x in pred_bbox]).reshape(-1, 4)
        annotations['score'] = np.array([float(x) for x in pred_score])
        self.anno_dets.append(annotations)

        key = data_samples[-1]['img_path'].split('/')[-1][:6]
        self.anno_gts.append(self.gts[key])
        
    def compute_metrics(self, results: list) -> dict:

        logger: MMLogger = MMLogger.get_current_instance()
        eval_results = OrderedDict()
        mAP, result_AP = get_official_eval_result(self.anno_gts,self.anno_dets,current_classes=self.current_classes)
        self.anno_dets = []
        self.anno_gts = []
        print_log(f'\n mAP : {mAP}',logger=logger)
        print_log('\n' + result_AP,logger=logger)
        eval_results['mAP'] = mAP
        return eval_results