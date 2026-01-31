import torch
import cv2
import numpy as np
import mmcv
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import os
from collections import defaultdict
import albumentations as A

def get_train_augmentation():

    return A.Compose([
        # 1. Horizontal Flip
        A.HorizontalFlip(p=0.5),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['global_ids', 'classes', 'visibilities'],
        min_visibility=0.25,    
        min_area=60,
    ))


def get_val_augmentation():

    return None  

class KITTITrackingReID(Dataset):
   
    def __init__(
        self,
        image_root: str,
        label_root: str,
        split : str,
        split_idxs : List,
        input_size: Tuple[int, int] = (621, 188),
        pad_size_divisor: int = 32,
        min_visibility: float = 0.25,
        filter_classes: List[str] = ['Car', 'Van', 'Pedestrian'],
        max_objects: int = 50,
        verbose: bool = False
    ):

        self.image_root = image_root
        self.label_root = label_root
        self.split = split
        self.sequences = split_idxs
        self.filter_classes = filter_classes
        self.min_visibility = min_visibility
        self.max_objects = max_objects
        
        
        # Adjust input image size
        self.input_w = input_size[0]  # 621 (resize target)
        self.input_h = input_size[1]  # 188 (resize target)
        self.padded_w = int(np.ceil(input_size[0] / 32) * 32)  # 640 (after padding)
        self.padded_h = int(np.ceil(input_size[1] / 32) * 32)  # 192 (after padding)
        self.pad_size_divisor = pad_size_divisor
        
        # BGR mean/std (MMDet ImageNet pretrained)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        self.samples = []
        self.id_mapping = {}  # (seq_id, track_id) -> global_id
        self.num_ids = 0

        if self.split == 'train':
            self.use_augmentation = True
            self.augmentation = get_train_augmentation()
        else:
            self.use_augmentation = False
            self.augmentation = get_val_augmentation()
        
        self._build_dataset()
        
        # Statistics
        total_objects = sum(len(s['annotations']) for s in self.samples)
        avg_objects = total_objects / len(self.samples) if self.samples else 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"KITTI Tracking ReID Dataset - {split.upper()}")
            print(f"{'='*60}")
            print(f"Image root: {image_root}")
            print(f"Label root: {label_root}")
            print(f"Sequences: {self.sequences}")
            print(f"Input size: ({self.input_w}, {self.input_h})")
            print(f"After padding: ({self.padded_w}, {self.padded_h})")
            print(f"Total frames: {len(self.samples)}")
            print(f"Unique IDs: {self.num_ids}")
            print(f"Total objects: {total_objects}")
            print(f"Avg objects/frame: {avg_objects:.1f}")
            print(f"Min visibility: {min_visibility}")
            print(f"Filter classes: {filter_classes}")
            print(f"{'='*60}\n")
    
    def _round_to_multiple(self, value: int, divisor: int) -> int:

        return int(np.ceil(value / divisor) * divisor)
    
    def _build_dataset(self):

        global_id = 0
        
        for seq_id in self.sequences:

            label_file = os.path.join(self.label_root, 'label_02', f'{seq_id:04d}.txt')
            
            if not os.path.exists(label_file):
                print(f"Warning: {label_file} not found, skipping sequence {seq_id}")
                continue
            

            seq_track_ids = set()
            frame_annotations = defaultdict(list)
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 17:  # KITTI format 
                        continue
                    
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    obj_type = parts[2]
                    truncated = float(parts[3])
                    occluded = int(parts[4])
                    
                    # Class filtering
                    if obj_type not in self.filter_classes:
                        continue
                    
                    # Visibility (truncation + occlusion)
                    visibility = 1.0 - truncated - occluded * 0.33
                    if visibility < self.min_visibility:
                        continue
                    
                    # Bbox: [left, top, right, bottom]
                    x1 = float(parts[6])
                    y1 = float(parts[7])
                    x2 = float(parts[8])
                    y2 = float(parts[9])
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    seq_track_ids.add(track_id)
                    
                    frame_annotations[frame_id].append({
                        'bbox': [x1, y1, x2, y2],
                        'track_id': track_id,
                        'class': obj_type,
                        'visibility': visibility,
                        'truncated': truncated,
                        'occluded': occluded,
                    })
            
            # Global ID mapping 
            for track_id in sorted(seq_track_ids):
                key = (seq_id, track_id)
                self.id_mapping[key] = global_id
                global_id += 1
            
            for frame_id in sorted(frame_annotations.keys()):
                annotations = frame_annotations[frame_id]
                
                if len(annotations) == 0:
                    continue
                
                if len(annotations) > self.max_objects:
                    annotations = sorted(annotations, 
                                       key=lambda x: x['visibility'], 
                                       reverse=True)[:self.max_objects]
                
                for ann in annotations:
                    key = (seq_id, ann['track_id'])
                    ann['global_id'] = self.id_mapping[key]
                
                image_path = os.path.join(
                    self.image_root, 
                    'image_02', 
                    f'{seq_id:04d}',
                    'images', 
                    f'{frame_id:06d}.png'
                )
                
                if not os.path.exists(image_path):
                    print(f"Warning: {image_path} not found")
                    continue
                
                self.samples.append({
                    'seq_id': seq_id,
                    'frame_id': frame_id,
                    'image_path': image_path,
                    'annotations': annotations,
                })
        
        self.num_ids = global_id
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (BGR format)
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"Failed to load: {sample['image_path']}")
        
        orig_h, orig_w = image.shape[:2]
        
        # Resize using mmcv
        image_resized = mmcv.imresize(
            image, 
            size=(self.input_w, self.input_h),  # (W, H) 
            return_scale=False,
            interpolation='bilinear',
            backend='cv2'
        )
        
        # Scale factors
        scale_x = self.input_w / orig_w
        scale_y = self.input_h / orig_h
        
        # Process annotations
        bboxes = []
        global_ids = []
        classes = []
        visibilities = []
        
        for ann in sample['annotations']:
            x1, y1, x2, y2 = ann['bbox']
            
            # Scale bbox
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y
            
            # Clip to RESIZED image bounds
            x1_clipped = np.clip(x1_scaled, 0, self.input_w - 1)
            y1_clipped = np.clip(y1_scaled, 0, self.input_h - 1)
            x2_clipped = np.clip(x2_scaled, 0, self.input_w - 1)
            y2_clipped = np.clip(y2_scaled, 0, self.input_h - 1)
            
            # Validity check
            if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
                continue
            
            bboxes.append([x1_clipped, y1_clipped, x2_clipped, y2_clipped])
            global_ids.append(ann['global_id'])
            classes.append(self._class_to_id(ann['class']))
            visibilities.append(ann['visibility'])
        
        if self.use_augmentation and self.augmentation is not None:
            # Albumentations expects RGB but we have BGR
            # Convert BGR -> RGB for augmentation
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

            try:
                transformed = self.augmentation(
                    image=image_rgb,
                    bboxes=bboxes,
                    global_ids=global_ids,
                    classes=classes,
                    visibilities=visibilities
                )
            
                # Get augmented data
                image_aug = transformed['image']
                bboxes = transformed['bboxes']
                global_ids = transformed['global_ids']
                classes = transformed['classes']
                visibilities = transformed['visibilities']
                
                # Convert RGB -> BGR
                image_resized = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
            
            except Exception as e:
                print(f"Warning: Augmentation failed for {sample['image_path']}: {e}")

        image_padded = mmcv.impad(
            image_resized,
            shape=(self.padded_h, self.padded_w),
            pad_val=0,  # BGR black padding
            padding_mode='constant'
        )
        
        # Image preprocessing
        # mmcv transforms: normalization + Convert CHW 
        image_tensor = torch.from_numpy(image_padded).float()  # (H, W, 3) BGR
        
        # Normalize: (image - mean) / std
        mean_tensor = torch.from_numpy(self.mean).view(1, 1, 3)
        std_tensor = torch.from_numpy(self.std).view(1, 1, 3)
        image_tensor = (image_tensor - mean_tensor) / std_tensor
        
        # HWC -> CHW
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # (3, H, W)
        
        # Convert to tensors
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if len(bboxes) > 0 \
                        else torch.zeros((0, 4), dtype=torch.float32)
        ids_tensor = torch.tensor(global_ids, dtype=torch.long) if len(global_ids) > 0 \
                     else torch.zeros((0,), dtype=torch.long)
        classes_tensor = torch.tensor(classes, dtype=torch.long) if len(classes) > 0 \
                         else torch.zeros((0,), dtype=torch.long)
        vis_tensor = torch.tensor(visibilities, dtype=torch.float32) if len(visibilities) > 0 \
                     else torch.zeros((0,), dtype=torch.float32)
        
        return {
            'image': image_tensor,  # (3, H, W) BGR normalized
            'bboxes': bboxes_tensor,  # (N, 4) [x1, y1, x2, y2]
            'ids': ids_tensor,  # (N,) global IDs
            'classes': classes_tensor,  # (N,)
            'visibilities': vis_tensor,  # (N,)
            'seq_id': sample['seq_id'],
            'frame_id': sample['frame_id'],
            'image_path': sample['image_path'],
            'num_objects': len(bboxes),
        }
    
    def _class_to_id(self, class_name: str) -> int:
        """KITTI class name -> class ID"""
        class_map = {
            'Car': 0,
            'Van': 0,  
            'Pedestrian': 1,
        }
        return class_map.get(class_name, 0)
    
    def get_id_info(self):
        id_counts = defaultdict(int)
        for sample in self.samples:
            for ann in sample['annotations']:
                id_counts[ann['global_id']] += 1
        
        return {
            'num_ids': self.num_ids,
            'id_counts': dict(id_counts),
            'min_frames': min(id_counts.values()) if id_counts else 0,
            'max_frames': max(id_counts.values()) if id_counts else 0,
            'avg_frames': np.mean(list(id_counts.values())) if id_counts else 0,
        }


def collate_fn_reid(batch):

    # Stack images
    images = torch.stack([item['image'] for item in batch])  # (B, 3, H, W)
    
    # Keep as lists
    bboxes = [item['bboxes'] for item in batch]  # List[Tensor(N_i, 4)]
    ids = [item['ids'] for item in batch]  # List[Tensor(N_i,)]
    classes = [item['classes'] for item in batch]  # List[Tensor(N_i,)]
    visibilities = [item['visibilities'] for item in batch]  # List[Tensor(N_i,)]
    
    # Metadata
    seq_ids = [item['seq_id'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    num_objects = [item['num_objects'] for item in batch]
    
    return {
        'images': images,  # (B, 3, H, W) BGR normalized
        'bboxes': bboxes,  # List[Tensor(N_i, 4)]
        'ids': ids,  # List[Tensor(N_i,)] - Global IDs
        'classes': classes,  # List[Tensor(N_i,)]
        'visibilities': visibilities,  # List[Tensor(N_i,)]
        'seq_ids': seq_ids,
        'frame_ids': frame_ids,
        'image_paths': image_paths,
        'num_objects': num_objects,
        'batch_size': len(batch),
    }


def create_kitti_reid_dataloaders(
    image_root: str,
    label_root : str,
    val_idxs : List,
    batch_size: int = 8,
    num_workers: int = 4,
    input_size: Tuple[int, int] = (621, 188),
    min_visibility: float = 0.25,
    verbose : bool = False,
    **kwargs
):
    
    train_file_idxs = []
    val_file_idxs = []
    for i in range(21):
        if i in val_idxs:
            val_file_idxs.append(i)
        else:
            train_file_idxs.append(i)

    # Train dataset
    train_dataset = KITTITrackingReID(
        image_root= image_root,
        label_root= label_root,
        split='train',
        split_idxs= train_file_idxs,
        input_size=input_size,
        min_visibility=min_visibility,
        verbose = verbose,
        **kwargs
    )
    
    # Val dataset
    val_dataset = KITTITrackingReID(
        image_root= image_root,
        label_root= label_root,
        split='val',
        split_idxs= val_file_idxs,
        input_size=input_size,
        min_visibility=min_visibility,
        verbose = verbose,
        **kwargs
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_reid,
        pin_memory=True,
        drop_last=True,  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_reid,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader