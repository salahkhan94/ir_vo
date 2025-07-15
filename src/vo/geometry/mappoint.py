#!/usr/bin/env python3
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2

class MapPoint:
    """
    Represents a 3D point in the map with observations, descriptors, and tracking information.
    This is a core component of ORB-SLAM2 for representing map structure.
    """
    
    def __init__(self, world_pos: np.ndarray, first_keyframe_id: int, map_id: int):
        """
        Initialize a MapPoint.
        
        Args:
            world_pos (np.ndarray): 3D position in world coordinates (x, y, z)
            first_keyframe_id (int): ID of the keyframe that first observed this point
            map_id (int): Unique ID for this map point
        """
        self.mWorldPos = world_pos.astype(np.float32)  # 3D position in world coordinates
        self.mnFirstKFid = first_keyframe_id           # First keyframe ID
        self.mnId = map_id                             # Unique map point ID
        
        # Observations
        self.mObservations = {}  # Dict: keyframe_id -> keypoint_index
        self.mNormalVector = None  # Normal vector for viewing direction
        self.mDescriptor = None    # Best descriptor for this map point
        
        # Tracking
        self.mnVisible = 1         # Number of times this point has been visible
        self.mnFound = 1           # Number of times this point has been found
        self.mfMaxDistance = 0     # Maximum distance for matching
        self.mfMinDistance = 0     # Minimum distance for matching
        
        # Bad flag
        self.mbBad = False         # Flag indicating if this map point is bad
        
        # Scale invariance
        self.mnScaleLevels = 8     # Number of scale levels
        self.mfScaleFactor = 1.2   # Scale factor between levels
        
        # Covisible keyframes
        self.mnObs = 0             # Number of observations
        
    def __str__(self):
        return f"MapPoint(id={self.mnId}, pos={self.mWorldPos}, observations={len(self.mObservations)})"
    
    def __repr__(self):
        return self.__str__()
    
    def add_observation(self, keyframe_id: int, keypoint_idx: int):
        """
        Add an observation of this map point from a keyframe.
        
        Args:
            keyframe_id (int): ID of the keyframe
            keypoint_idx (int): Index of the keypoint in the keyframe
        """
        self.mObservations[keyframe_id] = keypoint_idx
        self.mnObs = len(self.mObservations)
    
    def erase_observation(self, keyframe_id: int):
        """
        Remove an observation of this map point.
        
        Args:
            keyframe_id (int): ID of the keyframe to remove
        """
        if keyframe_id in self.mObservations:
            del self.mObservations[keyframe_id]
            self.mnObs = len(self.mObservations)
    
    def get_observations(self) -> Dict[int, int]:
        """
        Get all observations of this map point.
        
        Returns:
            Dict[int, int]: Dictionary mapping keyframe_id to keypoint_index
        """
        return self.mObservations.copy()
    
    def get_observation_count(self) -> int:
        """
        Get the number of observations.
        
        Returns:
            int: Number of observations
        """
        return self.mnObs
    
    def set_world_pos(self, world_pos: np.ndarray):
        """
        Set the 3D world position.
        
        Args:
            world_pos (np.ndarray): 3D position in world coordinates
        """
        self.mWorldPos = world_pos.astype(np.float32)
    
    def get_world_pos(self) -> np.ndarray:
        """
        Get the 3D world position.
        
        Returns:
            np.ndarray: 3D position in world coordinates
        """
        return self.mWorldPos.copy()
    
    def set_normal_vector(self, normal: np.ndarray):
        """
        Set the normal vector for viewing direction.
        
        Args:
            normal (np.ndarray): Normal vector
        """
        self.mNormalVector = normal.astype(np.float32)
    
    def get_normal_vector(self) -> Optional[np.ndarray]:
        """
        Get the normal vector.
        
        Returns:
            np.ndarray or None: Normal vector if set
        """
        return self.mNormalVector.copy() if self.mNormalVector is not None else None
    
    def set_descriptor(self, descriptor: np.ndarray):
        """
        Set the best descriptor for this map point.
        
        Args:
            descriptor (np.ndarray): ORB descriptor
        """
        self.mDescriptor = descriptor.copy()
    
    def get_descriptor(self) -> Optional[np.ndarray]:
        """
        Get the best descriptor.
        
        Returns:
            np.ndarray or None: ORB descriptor if set
        """
        return self.mDescriptor.copy() if self.mDescriptor is not None else None
    
    def increase_visible(self, n: int = 1):
        """
        Increase the visible count.
        
        Args:
            n (int): Number to increase by
        """
        self.mnVisible += n
    
    def increase_found(self, n: int = 1):
        """
        Increase the found count.
        
        Args:
            n (int): Number to increase by
        """
        self.mnFound += n
    
    def get_visible_ratio(self) -> float:
        """
        Get the ratio of found to visible observations.
        
        Returns:
            float: Ratio of found to visible observations
        """
        if self.mnVisible == 0:
            return 0.0
        return self.mnFound / self.mnVisible
    
    def set_bad(self):
        """Mark this map point as bad."""
        self.mbBad = True
    
    def is_bad(self) -> bool:
        """
        Check if this map point is bad.
        
        Returns:
            bool: True if bad, False otherwise
        """
        return self.mbBad
    
    def set_max_distance(self, distance: float):
        """
        Set the maximum distance for matching.
        
        Args:
            distance (float): Maximum distance
        """
        self.mfMaxDistance = distance
    
    def set_min_distance(self, distance: float):
        """
        Set the minimum distance for matching.
        
        Args:
            distance (float): Minimum distance
        """
        self.mfMinDistance = distance
    
    def get_max_distance(self) -> float:
        """
        Get the maximum distance for matching.
        
        Returns:
            float: Maximum distance
        """
        return self.mfMaxDistance
    
    def get_min_distance(self) -> float:
        """
        Get the minimum distance for matching.
        
        Returns:
            float: Minimum distance
        """
        return self.mfMinDistance
    
    def predict_scale(self, view_cos: float, level: int) -> int:
        """
        Predict the scale level based on viewing angle and current level.
        
        Args:
            view_cos (float): Cosine of viewing angle
            level (int): Current pyramid level
            
        Returns:
            int: Predicted scale level
        """
        ratio = self.mfMaxDistance / self.mfMinDistance
        scale = 1.0 / (ratio * view_cos)
        scale = np.clip(scale, 0.5, 2.0)
        
        level_scale = self.mfScaleFactor ** level
        scale_level = int(np.log(scale / level_scale) / np.log(self.mfScaleFactor))
        scale_level = np.clip(scale_level, 0, self.mnScaleLevels - 1)
        
        return scale_level 