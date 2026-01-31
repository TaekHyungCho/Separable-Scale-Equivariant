import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import yaml
import argparse
import ast
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tracking_reid.libs.dataset.dataset import create_kitti_reid_dataloaders

# Model import
from tracking_reid.libs.models.re_id.vanilla_reid import KITTI_REID_Vanilla
from tracking_reid.libs.models.re_id.sesn_reid import KITTI_REID_SESN
from tracking_reid.libs.models.re_id.disco_reid import KITTI_REID_DISCO
from tracking_reid.libs.models.re_id.sse_reid import KITTI_REID_SSE

from tracking_reid.libs.loss.reid_loss import ReIDLoss
from tracking_reid.libs.utils.roi_align_feature_extraction import extract_reid_features_at_roi
from tracking_reid.libs.tracking.eval import evaluate_model
from tracking_reid.libs.utils.logger import TopKModelManager

__model__ = {
    'ResNet': KITTI_REID_Vanilla,
    'ResNet_SESN': KITTI_REID_SESN,
    'ResNet_DISCO': KITTI_REID_DISCO,
    'ResNet_SSE': KITTI_REID_SSE
}

def parse_config():
    """Parse configuration file"""
    parser = argparse.ArgumentParser(description='REID branches Training and Evaluation')
    parser.add_argument('--cfg_path', type=str, help='Configuration file path')
    parser.add_argument('--val_idxs', type=str, help='Validation sequence indices, for example "[idx1, idx2, ..., idxN]')
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:
        cfgs = yaml.safe_load(f)

    if args.val_idxs:
        args.val_idxs = ast.literal_eval(args.val_idxs)
    
    return cfgs, args.val_idxs

def train_one_epoch(model, train_loader, reid_loss, optimizer, device):

    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):

        images = batch['images'].to(device)  
        bboxes = batch['bboxes']  
        ids = batch['ids']      

        reid_feature_map = model(images)

        features = extract_reid_features_at_roi(
            reid_feature_map, 
            bboxes, 
            scale_factor=8,
            output_size=(7, 7),
            sampling_ratio=2,
        )

        if features is None:  # No objects in batch
            continue

        ids_tensor = torch.cat(ids, dim=0).to(device)  # (total_N,)
        
        loss, loss_dict = reid_loss(features, ids_tensor)
        
        optimizer.zero_grad()
        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(reid_loss.parameters(), 5.0)
            optimizer.step()

        
        total_loss += loss_dict['id_loss']
        total_accuracy += loss_dict['id_accuracy']

        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': total_accuracy / (batch_idx + 1),
            'embs': f"{loss_dict['emb_scale']:.2f}"
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_accuracy / len(train_loader)
    
    print(f"Train Loss: {avg_loss:.4f} | Avg accuracy: {avg_acc:.4f}")

    return avg_loss

def main(cfgs, val_idxs):

    # Set seed
    torch.manual_seed(cfgs['seed'])
    np.random.seed(cfgs['seed'])
    random.seed(cfgs['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load DataLodaer
    train_loader, val_loader = create_kitti_reid_dataloaders(
        image_root = cfgs['image_root'],
        label_root = cfgs['label_root'],
        val_idxs = val_idxs,
        batch_size = cfgs['batch_size'],
        num_workers = cfgs['num_workers'],
        input_size = tuple(cfgs['input_size']),
        verbose = False)
    

    model_init_params = {
        'input_dims' : 512,
        'output_dims' : 128,
        'pretrained_backbone_path': cfgs['pretrained_backbone_path'],
        'basis_dir' : cfgs['basis_dir'] if 'basis_dir' in cfgs.keys() else None,
        'basis' : cfgs['basis'] if 'basis' in cfgs.keys() else None,  
        'permute' : cfgs['permute'] if 'permute' in cfgs.keys() else False 
    }

    model = __model__[cfgs['model']](**model_init_params).to(device)

    num_train_ids = train_loader.dataset.num_ids
    reid_loss = ReIDLoss(
        num_train_ids=num_train_ids,
        feat_dim=128,
        use_focal=True
    ).to(device)


    initial_lr = cfgs.get('initial_lr', 1e-4)
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': initial_lr, 'weight_decay': 1e-4},
        {'params': reid_loss.classifier.parameters(), 'lr': initial_lr, 'weight_decay': 1e-4}
    ])

    num_epochs = cfgs.get('epochs', 20)
    warmup_epochs = cfgs.get('warmup_epochs', 3)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1, 
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max= num_epochs - warmup_epochs,
        eta_min= cfgs.get('last_lr', 1e-5)
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    save_dir = cfgs['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_name = (
        f"{cfgs['model']}" 
        f"_{'combine' if cfgs.get('permute') else 'isolate'}" 
    ) if 'permute' in cfgs else f"{cfgs['model']}"

    top_k_manager = TopKModelManager(save_dir, model_name, k=5)
    log_file = os.path.join(save_dir, f"{model_name}_logging.json")

    training_log = {
        'model': model_name,
        'top_5_models': [],
        'epochs': []
    }


    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e} | {model_name}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, reid_loss, optimizer, device
        )

        tracker_outputs = evaluate_model(model,val_loader,save_dir,device)
        idf1 = float(tracker_outputs['IDF1']['average'])

        scheduler.step()

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'IDF1' : idf1,
            'learning_rate': float(current_lr),
        }

        training_log['epochs'].append(epoch_log)

        # Update Top-K models and save log
        top_k_manager.update(
            epoch=epoch + 1,
            eval_metric=idf1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            train_loss=train_loss
        )
        
        training_log['top_5_models'] = top_k_manager.get_top_models()
        training_log['best_metric'] = float(top_k_manager.get_best_metric())

        if training_log['top_5_models'][0]['epoch'] == epoch_log['epoch']:
            top_k_manager.copy_best_idf1_results(save_dir)
        
        top_k_manager.save_training_log(log_file, training_log)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Training 1 epoch time : {int(epoch_duration//60)}m {epoch_duration%60:.1f}s")
    
    training_log['total_epochs'] = num_epochs
    top_k_manager.save_training_log(log_file, training_log)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best metric: {top_k_manager.get_best_metric():.4f}")
    print(f"Checkpoints saved in: {save_dir}")
    print(f"Training log: {log_file}")

if __name__ == '__main__':

    # Load configuration
    cfgs, val_idxs= parse_config()
    main(cfgs, val_idxs)