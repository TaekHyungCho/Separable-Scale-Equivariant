import os
import json
import torch
import shutil

class TopKModelManager:

    def __init__(self, save_dir, model_name, k=5):
        self.save_dir = save_dir
        self.model_name = model_name
        self.k = k
        self.top_models = [] 
    
    def update(self, epoch, eval_metric, model_state, optimizer_state, train_loss):

        ckpt_name = f"{self.model_name}_epoch{epoch}.pth"
        ckpt_dir = os.path.join(self.save_dir, "ckpt")

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        
        checkpoint = {
            'epoch': epoch,
            'metric': eval_metric,
            'train_loss': train_loss,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state
        }

        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_name}")
        
        self.top_models.append((eval_metric, epoch, ckpt_path))
        self.top_models.sort(reverse=True, key=lambda x: x[0])
        
        if len(self.top_models) > self.k:
            _, removed_epoch, removed_path = self.top_models.pop()
            if os.path.exists(removed_path):
                os.remove(removed_path)
                print(f"Removed checkpoint: {os.path.basename(removed_path)}")
    
    def get_top_models(self):
        return [
            {
                'rank': i + 1,
                'epoch': epoch,
                'metric': float(metric),
                'checkpoint': os.path.basename(ckpt_path)
            }
            for i, (metric, epoch, ckpt_path) in enumerate(self.top_models)
        ]
    
    def get_best_metric(self):
        return self.top_models[0][0] if self.top_models else 0.0


    def save_training_log(self, log_path, log_data):

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"Updated log: {log_path}")
    
    def copy_best_idf1_results(self,save_path):

        # Copy summary files
        summary_dst_path_IDF1 = os.path.join(save_path, 'tracking', 'Best_IDF1')
        os.makedirs(summary_dst_path_IDF1, exist_ok=True)
        
        car_summary_src = os.path.join(save_path, 'tracking', 'car_summary.txt')
        ped_summary_src = os.path.join(save_path, 'tracking', 'pedestrian_summary.txt')
        
        if os.path.exists(car_summary_src):
            shutil.copy2(car_summary_src, os.path.join(summary_dst_path_IDF1, 'car_summary.txt'))
        if os.path.exists(ped_summary_src):
            shutil.copy2(ped_summary_src, os.path.join(summary_dst_path_IDF1, 'pedestrian_summary.txt'))