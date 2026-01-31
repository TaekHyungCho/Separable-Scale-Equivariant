import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import cv2
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict
from tracking_reid.libs.tracking.KF import KalmanFilter
from tracking_reid.libs.utils.roi_align_feature_extraction import extract_reid_features_at_roi

class Track:
    """
    A single target track with state space (u, v, γ, h, u̇, v̇, γ̇, ḣ).
    """
    
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        
        self._n_init = n_init
        self._max_age = max_age
    
    def to_tlwh(self):
        """Get current position in bounding box format (top left, width, height)."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def to_tlbr(self):
        """Get current position in bounding box format (min x, min y, max x, max y)."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def predict(self, kf):
        """Propagate the state distribution to the current time step using a Kalman filter prediction step."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature cache."""
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
    
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative
    
    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        """Returns True if this track is deleted."""
        return self.state == TrackState.Deleted


class TrackState:
    """Enumeration type for the single target track state."""
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Detection:
    """
    This class represents a bounding box detection in a single image.
    """
    
    def __init__(self, tlwh, feature, class_id=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.feature = np.asarray(feature.detach().cpu(), dtype=np.float32)
        self.class_id = class_id
    
    def to_tlbr(self):
        """Convert bounding box to format (min x, min y, max x, max y)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    def to_xyah(self):
        """Convert bounding box to format (center x, center y, aspect ratio, height)."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class Tracker:
    """
    DeepSORT tracker implementation
    """
    
    def __init__(self, model, max_age=30, n_init=3,
                 max_iou_distance=0.7, max_cosine_distance=0.2,
                 max_mahalanobis_distance=9.4877, nn_budget=100,
                 use_cuda=True):
        
        self.model = model
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.max_mahalanobis_distance = max_mahalanobis_distance
        self.nn_budget = nn_budget
        self.use_cuda = use_cuda
        
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)
    
    def update(self, model, bboxes, classes, image=None):
        """Perform measurement update and track management."""
        
        # Run matching cascade
        self.predict()
        
        # Convert detections to internal format
        if len(bboxes) == 0:
            detection_objects = []
        else:
            # Extract features for bboxes
            with torch.no_grad():
                reid_feature_map = model.forward(image)

                features = extract_reid_features_at_roi(
                    reid_feature_map, 
                    bboxes, 
                    scale_factor=8,
                    output_size=(7, 7),
                    sampling_ratio=2
                )

                if features is not None:
                    features = F.normalize(features, p=2, dim=1).cpu()
                else:
                    features = np.zeros((len(bboxes), 128))

            
            detection_objects = []
            for i, det in enumerate(bboxes[0]):
                x1, y1, x2, y2 = det
                
                # Convert to tlwh format
                tlwh = [x1, y1, x2 - x1, y2 - y1]
                
                detection_objects.append(Detection(
                    tlwh=tlwh,
                    feature=features[i],
                    class_id=classes[i]
                ))
        
        # Run matching cascade
        matches, unmatched_dets, unmatched_trks = self._match(detection_objects)
        
        # Update matched tracks
        for m in matches:
            self.tracks[m[0]].update(self.kf, detection_objects[m[1]])
        
        # Mark unmatched tracks as missed
        for i in unmatched_trks:
            self.tracks[i].mark_missed()
        
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            self._initiate_track(detection_objects[i])
        
        # Delete dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return results in format [x1, y1, x2, y2, track_id, class_id]
        results = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_tlbr()
                # Get class_id from the most recent detection if available
                class_id = 0  # Default
                if hasattr(track, 'class_id'):
                    class_id = track.class_id
                results.append([*bbox, track.track_id, class_id])
        
        return np.array(results) if results else np.empty((0, 6))
    
    def _match(self, detections):
        """Implementation of the assignment problem."""
        
        def gated_metric(tracks, dets, track_indices, detection_indices):
            """Compute distance matrix with gating."""
            # Extract features and compute cosine distance
            features = np.array([dets[i].feature for i in detection_indices])
            
            # Compute cost matrix
            cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
            
            for row, track_idx in enumerate(track_indices):
                if tracks[track_idx].time_since_update > 1:
                    cost_matrix[row, :] = self.max_cosine_distance + 1e-5
                    continue
                
                track_features = tracks[track_idx].features
                if len(track_features) == 0:
                    cost_matrix[row, :] = self.max_cosine_distance + 1e-5
                    continue
                
                if len(track_features) > self.nn_budget:
                    track_features = track_features[-self.nn_budget:]
                
                for col, det_idx in enumerate(detection_indices):
                    # Compute cosine distance
                    det_feature = features[col]
                    
                    distances = []
                    for track_feature in track_features:
                        dist = 1.0 - np.dot(det_feature, track_feature)
                        distances.append(dist)
                    
                    cost_matrix[row, col] = min(distances) if distances else self.max_cosine_distance + 1e-5
            
            return cost_matrix
        
        def gate_cost_matrix(cost_matrix, tracks, dets, track_indices, 
                           detection_indices, gating_threshold):
            """Apply gating to cost matrix."""
            gated_cost = np.copy(cost_matrix)
            
            for row, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                
                # Get measurements
                measurements = np.array([dets[i].to_xyah() for i in detection_indices])
                
                if len(measurements) == 0:
                    continue
                
                # Compute Mahalanobis distance
                gating_distance = self.kf.gating_distance(
                   track.mean, track.covariance, measurements, only_position=False)
                
                # Apply gating
                for col, det_idx in enumerate(detection_indices):
                    if gating_distance[col] > gating_threshold:
                       gated_cost[row, col] = self.max_cosine_distance + 1e-5
                    
                    if cost_matrix[row, col] > self.max_cosine_distance:
                        gated_cost[row, col] = self.max_cosine_distance + 1e-5
            
            return gated_cost
        
        # Matching cascade
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # Cascade matching for confirmed tracks
        matches_a, unmatched_trks_a, unmatched_dets = \
            self._matching_cascade(gated_metric, gate_cost_matrix, 
                                 self.max_cosine_distance, self.max_age,
                                 self.tracks, detections, confirmed_tracks)
        
        # IOU matching for unconfirmed tracks and remaining detections
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_trks_a 
                                                   if self.tracks[k].time_since_update == 1]
        unmatched_trks_a = [k for k in unmatched_trks_a 
                          if self.tracks[k].time_since_update != 1]
        
        matches_b, unmatched_trks_b, unmatched_dets = \
            self._min_cost_matching(self._iou_distance, self.max_iou_distance,
                                  self.tracks, detections, iou_track_candidates, unmatched_dets)
        
        matches = matches_a + matches_b
        unmatched_trks = list(set(unmatched_trks_a + unmatched_trks_b))
        
        return matches, unmatched_dets, unmatched_trks
    
    def _matching_cascade(self, distance_metric, gate_metric, max_distance, 
                         cascade_depth, tracks, detections, track_indices=None):
        """Implementation of matching cascade"""
        if track_indices is None:
            track_indices = list(range(len(tracks)))
        
        unmatched_detections = list(range(len(detections)))
        matches = []
        
        # Cascade by track age
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:
                break
            
            # Select tracks by age
            track_indices_l = [k for k in track_indices 
                             if tracks[k].time_since_update == 1 + level]
            
            if len(track_indices_l) == 0:
                continue
            
            # Run matching on this level
            matches_l, _, unmatched_detections = \
                self._min_cost_matching(distance_metric, max_distance, tracks, 
                                      detections, track_indices_l, unmatched_detections,
                                      gate_metric)
            matches += matches_l
        
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections
    
    def _min_cost_matching(self, distance_metric, max_distance, tracks, detections,
                          track_indices, detection_indices, gate_metric=None):
        """Solve linear assignment problem using Hungarian algorithm."""
        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices
        
        # Compute cost matrix
        cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
        
        if gate_metric is not None:
            cost_matrix = gate_metric(cost_matrix, tracks, detections,
                                    track_indices, detection_indices, self.max_mahalanobis_distance)
        
        # Set costs above threshold to high value
        cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
        
        # Solve assignment problem
        indices = linear_sum_assignment(cost_matrix)
        
        matches, unmatched_tracks, unmatched_detections = [], [], []
        
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[1]:
                unmatched_detections.append(detection_idx)
        
        for row, track_idx in enumerate(track_indices):
            if row not in indices[0]:
                unmatched_tracks.append(track_idx)
        
        for row, col in zip(indices[0], indices[1]):
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _iou_distance(self, tracks, detections, track_indices, detection_indices):
        """Compute IOU distance matrix."""
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        
        for row, track_idx in enumerate(track_indices):
            if tracks[track_idx].time_since_update > 1:
                cost_matrix[row, :] = 1.0
                continue
            
            bbox = tracks[track_idx].to_tlbr()
            candidates = np.array([detections[i].to_tlbr() for i in detection_indices])
            
            cost_matrix[row, :] = 1.0 - self._iou(bbox, candidates)
        
        return cost_matrix
    
    def _iou(self, bbox, candidates):
        """Compute intersection over union."""
        bbox_tl, bbox_br = bbox[:2], bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, 2:]
        
        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                   np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                   np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        
        wh = np.maximum(0., br - tl)
        
        area_intersection = wh.prod(axis=1)
        area_bbox = (bbox_br - bbox_tl).prod()
        area_candidates = (candidates_br - candidates_tl).prod(axis=1)
        
        return area_intersection / (area_bbox + area_candidates - area_intersection)
    
    def _initiate_track(self, detection):
        """Create new track."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(mean, covariance, self._next_id, self.n_init, self.max_age,
                     feature=detection.feature)
        if hasattr(detection, 'class_id'):
            track.class_id = detection.class_id
        self.tracks.append(track)
        self._next_id += 1

