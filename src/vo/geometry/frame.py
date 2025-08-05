#!/usr/bin/env python3
import numpy as np
from threading import Lock

class Frame:
    """
    Frame class representing a processed frame with pose and inlier observations.
    Similar to KeyFrame but lighter weight for tracking frames.
    """
    
    def __init__(self, frame_id, timestamp, pose, keypoints, descriptors, camera_matrix):
        """
        Initialize a frame.
        
        Args:
            frame_id: Unique frame ID
            timestamp: Frame timestamp
            pose: Estimated pose matrix (4x4)
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            camera_matrix: Camera intrinsic matrix (3x3)
        """
        # Debug check for camera_matrix being None
        if camera_matrix is None:
            import rospy
#             rospy.logerr(f"ERROR: camera_matrix is None when creating Frame {frame_id}!")
        
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.pose = pose.copy() if pose is not None else np.eye(4)
        self.keypoints = list(keypoints) if keypoints is not None else []
        self.descriptors = descriptors.copy() if descriptors is not None else None
        self.camera_matrix = camera_matrix.copy() if camera_matrix is not None else None
        
        # Inlier observations: {feature_id: (map_point, keypoint)}
        # These are the inliers from Essential Matrix estimation
        self.inlier_observations = {}
        
        # Thread safety
        self.lock = Lock()
    
    def add_inlier_observation(self, feature_id, map_point, keypoint):
        """
        Add an inlier observation to this frame.
        
        Args:
            feature_id: Feature ID in this frame
            map_point: Associated MapPoint object
            keypoint: 2D keypoint coordinates
        """
        with self.lock:
            self.inlier_observations[feature_id] = (map_point, keypoint)
    
    def get_inlier_observations(self):
        """
        Get all inlier observations for this frame.
        
        Returns:
            Dictionary of {feature_id: (map_point, keypoint)}
        """
        with self.lock:
            return self.inlier_observations.copy()
    
    def get_pose(self):
        """
        Get the pose of this frame.
        
        Returns:
            4x4 pose matrix
        """
        with self.lock:
            return self.pose.copy()
    
    def set_pose(self, pose):
        """
        Set the pose of this frame.
        
        Args:
            pose: 4x4 pose matrix
        """
        with self.lock:
            self.pose = pose.copy()
    
    def get_keypoints(self):
        """
        Get the keypoints of this frame.
        
        Returns:
            List of keypoints
        """
        with self.lock:
            return list(self.keypoints)
    
    def get_descriptors(self):
        """
        Get the descriptors of this frame.
        
        Returns:
            Descriptor array
        """
        with self.lock:
            return self.descriptors.copy() if self.descriptors is not None else None
    
    def get_camera_matrix(self):
        """
        Get the camera matrix of this frame.
        
        Returns:
            3x3 camera matrix
        """
        with self.lock:
            return self.camera_matrix.copy() if self.camera_matrix is not None else None
    
    def get_inlier_count(self):
        """
        Get the number of inlier observations.
        
        Returns:
            Number of inlier observations
        """
        with self.lock:
            return len(self.inlier_observations) 