#!/usr/bin/env python3
import numpy as np
from threading import Lock

class MapPoint:
    """
    MapPoint class representing a 3D point in the map.
    Similar to ORB-SLAM2 implementation.
    """
    
    def __init__(self, position_3d, first_keyframe_id, first_feature_id):
        """
        Initialize a map point.
        
        Args:
            position_3d: 3D position in world coordinates (numpy array)
            first_keyframe_id: ID of the keyframe where this point was first observed
            first_feature_id: Feature ID in the first keyframe
        """
        self.position_3d = position_3d.copy()  # 3D position in world coordinates
        self.normal = None  # Normal vector (will be computed later)
        
        # Observations: {keyframe_id: feature_id} and {frame_id: feature_id}
        self.observations = {first_keyframe_id: first_feature_id}
        self.frame_observations = {}  # {frame_id: feature_id} for regular frames
        
        # Quality metrics
        self.observations_count = 1
        self.bad_observations_count = 0
        self.mean_observation_direction = None
        self.max_distance = 0
        self.min_distance = 0
        
        # Tracking info
        self.track_proj_x = 0
        self.track_proj_y = 0
        self.track_view_cos = 0
        self.track_scale_level = 0
        self.track_view_cos_angle = 0
        
        # Map point status
        self.is_bad = False
        self.is_visible = False
        self.is_found = False
        
        # Thread safety
        self.lock = Lock()
    
    def add_observation(self, keyframe_id, feature_id):
        """
        Add a new observation of this map point from a keyframe.
        
        Args:
            keyframe_id: ID of the keyframe observing this point
            feature_id: Feature ID in the keyframe
        """
        with self.lock:
            if keyframe_id not in self.observations:
                self.observations[keyframe_id] = feature_id
                self.observations_count += 1
    
    def add_frame_observation(self, frame_id, feature_id):
        """
        Add a new observation of this map point from a regular frame.
        
        Args:
            frame_id: ID of the frame observing this point
            feature_id: Feature ID in the frame
        """
        with self.lock:
            if frame_id not in self.frame_observations:
                self.frame_observations[frame_id] = feature_id
                self.observations_count += 1
    
    def remove_observation(self, keyframe_id):
        """
        Remove an observation of this map point from a keyframe.
        
        Args:
            keyframe_id: ID of the keyframe to remove
        """
        with self.lock:
            if keyframe_id in self.observations:
                del self.observations[keyframe_id]
                self.observations_count -= 1
                self.bad_observations_count += 1
    
    def remove_frame_observation(self, frame_id):
        """
        Remove an observation of this map point from a frame.
        
        Args:
            frame_id: ID of the frame to remove
        """
        with self.lock:
            if frame_id in self.frame_observations:
                del self.frame_observations[frame_id]
                self.observations_count -= 1
                self.bad_observations_count += 1
    
    def get_observations(self):
        """
        Get all observations of this map point from keyframes.
        
        Returns:
            Dictionary of {keyframe_id: feature_id}
        """
        with self.lock:
            return self.observations.copy()
    
    def get_frame_observations(self):
        """
        Get all observations of this map point from regular frames.
        
        Returns:
            Dictionary of {frame_id: feature_id}
        """
        with self.lock:
            return self.frame_observations.copy()
    
    def get_all_observations(self):
        """
        Get all observations of this map point from both keyframes and frames.
        
        Returns:
            Dictionary of {id: feature_id} where id can be keyframe_id or frame_id
        """
        with self.lock:
            all_obs = self.observations.copy()
            all_obs.update(self.frame_observations)
            return all_obs
    
    def get_observations_count(self):
        """
        Get the number of observations.
        
        Returns:
            Number of observations
        """
        with self.lock:
            return self.observations_count
    
    def get_unique_observers_count(self):
        """
        Get the number of unique frames/keyframes observing this map point.
        
        Returns:
            Number of unique observers (keyframes + frames)
        """
        with self.lock:
            return len(self.observations) + len(self.frame_observations)
    
    def is_observed_by_keyframe(self, keyframe_id):
        """
        Check if this map point is observed by a specific keyframe.
        
        Args:
            keyframe_id: ID of the keyframe to check
            
        Returns:
            True if observed, False otherwise
        """
        with self.lock:
            return keyframe_id in self.observations
    
    def is_observed_by_frame(self, frame_id):
        """
        Check if this map point is observed by a specific frame.
        
        Args:
            frame_id: ID of the frame to check
            
        Returns:
            True if observed, False otherwise
        """
        with self.lock:
            return frame_id in self.frame_observations
    
    def get_feature_id_in_keyframe(self, keyframe_id):
        """
        Get the feature ID of this map point in a specific keyframe.
        
        Args:
            keyframe_id: ID of the keyframe
            
        Returns:
            Feature ID if observed, None otherwise
        """
        with self.lock:
            return self.observations.get(keyframe_id, None)
    
    def get_feature_id_in_frame(self, frame_id):
        """
        Get the feature ID of this map point in a specific frame.
        
        Args:
            frame_id: ID of the frame
            
        Returns:
            Feature ID if observed, None otherwise
        """
        with self.lock:
            return self.frame_observations.get(frame_id, None)
    
    def set_bad(self):
        """Mark this map point as bad."""
        with self.lock:
            self.is_bad = True
    
    def is_bad_point(self):
        """
        Check if this map point is marked as bad.
        
        Returns:
            True if bad, False otherwise
        """
        with self.lock:
            return self.is_bad
    
    def get_position(self):
        """
        Get the 3D position of this map point.
        
        Returns:
            3D position as numpy array
        """
        with self.lock:
            return self.position_3d.copy()
    
    def set_position(self, position_3d):
        """
        Set the 3D position of this map point.
        
        Args:
            position_3d: New 3D position
        """
        with self.lock:
            self.position_3d = position_3d.copy()
    
    def compute_distinctive_descriptors(self):
        """
        Compute distinctive descriptors for this map point.
        This would be used for place recognition (not implemented here).
        """
        # Placeholder for future implementation
        pass
    
    def update_normal_and_depth(self, keyframe_poses):
        """
        Update the normal vector and depth information.
        
        Args:
            keyframe_poses: Dictionary of {keyframe_id: pose_matrix}
        """
        with self.lock:
            if len(self.observations) < 2:
                return
            
            # Compute mean observation direction
            directions = []
            for keyframe_id in self.observations:
                if keyframe_id in keyframe_poses:
                    pose = keyframe_poses[keyframe_id]
                    camera_center = pose[:3, 3]
                    direction = self.position_3d - camera_center
                    direction = direction / np.linalg.norm(direction)
                    directions.append(direction)
            
            if directions:
                self.mean_observation_direction = np.mean(directions, axis=0)
                self.mean_observation_direction = self.mean_observation_direction / np.linalg.norm(self.mean_observation_direction)
    
    def predict_scale(self, view_cos, median_depth):
        """
        Predict the scale level for this map point.
        
        Args:
            view_cos: Cosine of viewing angle
            median_depth: Median depth of the scene
            
        Returns:
            Predicted scale level
        """
        # Simple scale prediction based on viewing angle and depth
        # This is a simplified version
        return 0  # Placeholder 