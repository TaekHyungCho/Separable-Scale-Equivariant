import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from fvcore.nn import sigmoid_focal_loss_jit

class ReIDLoss(nn.Module):

    def __init__(self, num_train_ids, feat_dim=128,use_focal=True, triplet_margin=0.3):
        super().__init__()

        self.num_train_ids = num_train_ids
        self.feat_dim = feat_dim
        self.use_focal = use_focal
        self.triplet_margin = triplet_margin

        # Classifier
        self.classifier = nn.Linear(feat_dim, num_train_ids)
        
        if use_focal:
            nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.classifier.bias, bias_value)
        else:
            nn.init.normal_(self.classifier.weight, std=0.001)
            nn.init.constant_(self.classifier.bias, 0)
        
        self.emb_scale = math.sqrt(2) * math.log(num_train_ids - 1)

        if not use_focal:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    def _batch_hard_triplet_loss(self, features, ids, margin):

        # features: (N, D), already normalized
        pairwise_dist = 1.0 - torch.mm(features, features.t())  # (N, N)
        
        pairwise_dist = torch.clamp(pairwise_dist, min=0.0)
        
        # ID matching mask
        ids = ids.view(-1, 1)  # (N, 1)
        mask_pos = (ids == ids.t()).float()  # (N, N) - same ID
        mask_neg = (ids != ids.t()).float()  # (N, N) - different ID
        
        hardest_pos_dist = (pairwise_dist * mask_pos).max(dim=1)[0]  # (N,)
        
        masked_neg_dist = pairwise_dist + mask_pos * 1e6
        hardest_neg_dist = masked_neg_dist.min(dim=1)[0]  # (N,)
        
        triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
        
        valid_triplets = (mask_pos.sum(dim=1) > 0) & (mask_neg.sum(dim=1) > 0)
        
        if valid_triplets.sum() > 0:
            triplet_loss = triplet_loss[valid_triplets].mean()
        else:
            triplet_loss = torch.tensor(0.0, device=features.device)
        
        return triplet_loss

    def forward(self, features, ids):
        """
        Args:
            features: (N, 128) 
            ids: (N,) 
        
        Returns:
            loss: scalar
        """
        features_norm = F.normalize(features, dim=1)  # L2 normalize
        features_scaled = self.emb_scale * features_norm  # Scale

        logits = self.classifier(features_scaled)

        if self.use_focal:
            # Focal Loss
            target_one_hot = logits.new_zeros((logits.size(0), self.num_train_ids))
            target_one_hot.scatter_(1, ids.long().view(-1, 1), 1)
            
            classification_loss = sigmoid_focal_loss_jit(
                logits, 
                target_one_hot,
                alpha=0.25, 
                gamma=2.0, 
                reduction="sum"
            ) / logits.size(0)
        else:
            # CrossEntropy Loss
            classification_loss = self.criterion(logits, ids)
        
        total_loss = classification_loss

        # 4. Metrics for monitoring
        with torch.no_grad():
            pred_ids = torch.argmax(logits, dim=1)
            accuracy = (pred_ids == ids).float().mean()
            
            # Top-5 accuracy
            _, top5_preds = torch.topk(logits, k=min(5, self.num_train_ids), dim=1)
            top5_correct = (top5_preds == ids.unsqueeze(1)).any(dim=1).float().mean()
        
        loss_dict = {
            'id_loss': total_loss.item(),
            'id_accuracy': accuracy.item(),
            'id_top5_accuracy': top5_correct.item(),
            'emb_scale': self.emb_scale,
        }
        
        return total_loss, loss_dict