import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Iterator, Tuple, List, Dict, Any
from torch.utils.data import DataLoader
from tracking_reid.libs.tracking.tracker import Tracker

def iterate_val_loader_by_sequence(
    val_loader: DataLoader
) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:

    current_seq_id = None
    current_sequence_data = []

    for batch in val_loader:

        seq_id = batch['seq_ids'][0]

        if seq_id != 19:
            continue
        
        if current_seq_id is None:
            current_seq_id = seq_id
        
        if seq_id != current_seq_id:
            yield current_seq_id, current_sequence_data
            
            current_seq_id = seq_id
            current_sequence_data = []

        frame_data = {
            'image': batch['images'][0],
            'bboxes': batch['bboxes'][0],
            'ids': batch['ids'][0],
            'classes': batch['classes'][0],
            'visibilities': batch['visibilities'][0],
            'frame_id': batch['frame_ids'][0],
        }
        current_sequence_data.append(frame_data)

    if current_seq_id is not None and current_sequence_data:
        yield current_seq_id, current_sequence_data

def convert_to_kitti_format(tracker_outputs, output_dir, sequence_name, scale_factor = 2.0):

    final_output_dir = os.path.join(output_dir, 'data')
    os.makedirs(final_output_dir, exist_ok=True)
    output_file = os.path.join(final_output_dir, f"{sequence_name:04d}.txt")
    
    try:
        with open(output_file, 'w') as f:
            # Sort frames for consistent output
            for frame_id in sorted(tracker_outputs.keys()):
                tracks = tracker_outputs[frame_id]
                
                # Handle both single track and multiple tracks
                if len(tracks.shape) == 1:
                    tracks = tracks.reshape(1, -1)
                
                for track in tracks:
                    if len(track) >= 6:
                        # Extract required info from [x1,y1,x2,y2,track_id,class_id] format
                        x1, y1, x2, y2, track_id, class_id = track[:6]
                        if scale_factor is not None:
                            x1 *= scale_factor
                            y1 *= scale_factor
                            x2 *= scale_factor
                            y2 *= scale_factor
                        
                        # Determine class type (0 = Car, 1 = Pedestrian)
                        obj_type = "Pedestrian" if int(class_id) == 1 else "Car"
                        
                        # Set default values for KITTI format
                        truncated = -1
                        occluded = -1
                        alpha = -1
                        h, w, l = -1, -1, -1  # 3D size
                        x, y, z = -1, -1, -1  # 3D position
                        rotation_y = -1
                        
                        # Write in KITTI format (space-separated)
                        f.write(f"{frame_id} {int(track_id)} {obj_type} {truncated} {occluded} {alpha} "
                                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h} {w} {l} {x} {y} {z} {rotation_y} {rotation_y}\n")
        
        print(f"Successfully saved tracking results to {output_file}")
    except Exception as e:
        print(f"Error saving tracking results to {output_file}: {e}")
        # Create empty file to avoid issues
        with open(output_file, 'w') as f:
            pass

# Evaluate tracking using TrackEval
def evaluate_tracking(pred_dir, gt_dir, seqmap_file=None):
    """
    Evaluate tracking performance on KITTI dataset using TrackEval API
    
    Returns:
        metrics: Dictionary of simple metrics
    """
    try:
        import trackeval
        
        # Evaluation settings
        eval_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': True,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_DETAILED': False,
            'PLOT_CURVES': False,
            'BREAK_ON_ERROR': True
        }
        
        # KITTI dataset settings
        dataset_config = {
            'GT_FOLDER': gt_dir,
            'TRACKERS_FOLDER': os.path.dirname(pred_dir),
            'TRACKERS_TO_EVAL': [os.path.basename(pred_dir)],
            'SPLIT_TO_EVAL': 'training',  # KITTI uses 'training' split
            'CLASSES_TO_EVAL': ['car', 'pedestrian'],  # KITTI classes
            'PRINT_CONFIG': True,
        }
        
        if seqmap_file and os.path.exists(seqmap_file):
            dataset_config['SEQMAP_FILE'] = seqmap_file
        
        # Run evaluation
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.Kitti2DBox(dataset_config)  # Use KITTI dataset
        metrics = [
            trackeval.metrics.HOTA(),
            trackeval.metrics.CLEAR(),
            trackeval.metrics.Identity()
        ]
        
        results, _ = evaluator.evaluate([dataset], metrics)
        
        # Parse results - safe handling
        try:
            # Print results structure for debugging
            print("Results structure:", results.keys())
            
            # If results is a tuple, use first element
            if isinstance(results, tuple) and len(results) > 0:
                results = results[0]
            
            # Safely extract results
            tracker_name = os.path.basename(pred_dir)
            
            # Initialize metrics result
            metrics_result = {
                'IDF1': {
                    'car': 0.0,
                    'pedestrian': 0.0,
                    'average': 0.0
                }
            }

            if 'Kitti2DBox' in results and tracker_name in results['Kitti2DBox']:
                tracker_results = results['Kitti2DBox'][tracker_name]['COMBINED_SEQ']
                
                # Extract car results if available
                if 'car' in tracker_results:
                    car_results = tracker_results['car']
                    
                    if 'Identity' in car_results and 'IDF1' in car_results['Identity']:
                        metrics_result['IDF1']['car'] = car_results['Identity']['IDF1']
                
                # Extract pedestrian results if available
                if 'pedestrian' in tracker_results:
                    ped_results = tracker_results['pedestrian']
                    
                    if 'Identity' in ped_results and 'IDF1' in ped_results['Identity']:
                        metrics_result['IDF1']['pedestrian'] = ped_results['Identity']['IDF1']

                # Calculate averages
                metrics_result['IDF1']['average'] = (metrics_result['IDF1']['car'] + metrics_result['IDF1']['pedestrian']) / 2
            
            return metrics_result
        
        except Exception as e:
            print(f"Error during results parsing: {e}")
            print("Results type:", type(results))
            if hasattr(results, 'items'):
                print("Results keys:", results.keys())
            return {
                'IDF1': {'car': 0.0, 'pedestrian': 0.0, 'average': 0.0}
            }
        
    except Exception as e:
        print(f"Tracking evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'IDF1': {'car': 0.0, 'pedestrian': 0.0, 'average': 0.0}
        }

def track_sequence(model,batch,tracker=None,tracker_outputs=None,device='cuda:0',max_age=30,n_init=3):

    if (tracker is None) and (tracker_outputs is None):
        tracker = Tracker(
            model=model,
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=0.5,
            max_cosine_distance=0.2,
            max_mahalanobis_distance=9.4877,
            nn_budget=100,
            use_cuda=True if device.type == 'cuda' else False,
        )
        tracker_outputs = {}

        # Set the trained model
        if model is not None:
            tracker.model = model.to(device)
            tracker.model.eval()
            tracker.device = device
    

    image = batch['images'].to(device)
    bboxes = [batch['bboxes'][0].detach().cpu().numpy()]
    classes = batch['classes'][0].detach().cpu().numpy()
    frame_id = batch['frame_ids'][0]

    # Update tracker with detections
    if len(bboxes) > 0:
        tracks = tracker.update(model, bboxes, classes, image)
    else:
        tracks = tracker.update(model, np.array([]),np.array([]), image)
    # Save results by frame
    if len(tracks) > 0:
        if isinstance(tracks, torch.Tensor): 
            tracker_outputs[frame_id] = tracks.detach().cpu().numpy()
        else: # If numpy
            tracker_outputs[frame_id] = tracks
    
    return tracker, tracker_outputs

def evaluate_model(model,val_loader,output_dir,device,max_age=30,n_init=3):

    model.eval()
    sequence_results = {}
    current_sequence_id = None
    tracker = None
    tracker_outputs = None

    tracker_output_dir = os.path.join(output_dir, "tracking")
    os.makedirs(tracker_output_dir, exist_ok=True)

    for batch in tqdm(val_loader, desc="Feature extraction"):

        seq_id = batch['seq_ids'][0]

        if current_sequence_id is None: ## Frist sequence
            current_sequence_id = seq_id
            print(f"--- Processing Sequence ID: {seq_id} ---")

        elif current_sequence_id != seq_id:

            ## Handling previous tracking results
            previous_sequence_id = current_sequence_id
            if len(tracker_outputs) > 0:
                output_file = convert_to_kitti_format(tracker_outputs, tracker_output_dir, previous_sequence_id)
                sequence_results[f"{previous_sequence_id:04d}"] = len(tracker_outputs)
                print(f"  Saved results to: {output_file}")
            else:
                print(f"  Warning: No tracking results for sequence {previous_sequence_id:04d}")
                sequence_results[f"{previous_sequence_id:04d}"] = 0
            
            ## New sequence incoming
            current_sequence_id = seq_id
            print(f"--- Processing Sequence ID: {seq_id} ---")
            tracker = None # Re-Initialize tracker
            tracker_outputs = None # Re-Initialize tracker outputs
        
        tracker, tracker_outputs = track_sequence(model,batch,tracker,tracker_outputs,device,max_age,n_init)

    ## Last sequences
    if len(tracker_outputs) > 0:
        output_file = convert_to_kitti_format(tracker_outputs, tracker_output_dir, seq_id)
        sequence_results[f"{seq_id:04d}"] = len(tracker_outputs)
        print(f"  Saved results to: {output_file}")
    else:
        print(f"  Warning: No tracking results for sequence {seq_id:04d}")
        sequence_results[f"{seq_id:04d}"] = 0
        
    # Run evaluation if we have tracking results
    if any(count > 0 for count in sequence_results.values()):
        print(f"\nRunning evaluation for tracker...")
        
        try:
            gt_dir = os.path.join("tracking_reid", "track_label")
            seqmap_file = os.path.join(gt_dir, "evaluate_tracking.seqmap.training")
            
            if not os.path.exists(gt_dir):
                print(f"Warning: Ground truth directory not found: {gt_dir}")
                metrics = {
                    'IDF1': {'car': 0.0, 'pedestrian': 0.0, 'average': 0.0}
                }
            else:
                results = evaluate_tracking(
                    pred_dir=tracker_output_dir,
                    gt_dir=gt_dir,
                    seqmap_file=seqmap_file if os.path.exists(seqmap_file) else None
                )
            
            
            print(f"  Evaluation Results:")
            print(f"  Car IDF1: {results['IDF1']['car']:.4f}")
            print(f"  Pedestrian IDF1: {results['IDF1']['pedestrian']:.4f}")
            print(f"  Average IDF1: {results['IDF1']['average']:.4f}")
        except Exception as e:
            print(f"Error during evaluation for tracking: {e}")
            import traceback
            traceback.print_exc()
            
            results= {'IDF1': {'car': 0.0, 'pedestrian': 0.0, 'average': 0.0}}
    else:
        print(f"No valid tracking results, skipping evaluation")
        results= {'IDF1': {'car': 0.0, 'pedestrian': 0.0, 'average': 0.0}}
    
    return results