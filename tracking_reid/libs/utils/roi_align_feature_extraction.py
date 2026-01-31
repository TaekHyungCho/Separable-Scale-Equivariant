import torch
import numpy as np
import torch.nn.functional as F
from torchvision.ops import roi_align

def extract_reid_features_at_roi(
    reid_feature_map, 
    bboxes, 
    scale_factor=8,
    output_size=(7, 7),
    sampling_ratio=2
):
   
    device = reid_feature_map.device
    B, C = reid_feature_map.size(0), reid_feature_map.size(1)
    
    rois_list = []
    
    for batch_idx in range(B):

        if isinstance(bboxes[batch_idx],torch.Tensor): # Train
            bbox_batch =bboxes[batch_idx]
        elif isinstance(bboxes[batch_idx], np.ndarray): # Track
            bbox_batch = torch.from_numpy(bboxes[batch_idx])
            
        if len(bbox_batch) == 0:
            continue
        
        bbox_scaled = (bbox_batch / scale_factor).to(device)
        
        batch_indices = torch.full(
            (len(bbox_batch), 1), 
            batch_idx, 
            dtype=torch.float32,
            device=device
        )
        
        rois = torch.cat([batch_indices, bbox_scaled], dim=1)
        rois_list.append(rois)
    
    if len(rois_list) == 0:
        return None
    
    rois = torch.cat(rois_list, dim=0)
    
    pooled_features = roi_align(
        reid_feature_map,
        rois,
        output_size=output_size,
        spatial_scale=1.0,
        sampling_ratio=sampling_ratio,
        aligned=True
    )  
    
    features = F.adaptive_avg_pool2d(pooled_features, (1, 1))

    features = features.view(-1, C)
    
    return features