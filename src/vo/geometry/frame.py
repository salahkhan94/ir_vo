#!/usr/bin/env python3
import numpy as np
import cv2
from threading import Lock

class Frame:
    """
    Frame class representing a regular frame in the SLAM system.
    Stores pose and inlier observations from pose recovery for bundle adjustment.
    Similar to KeyFrame but lighter weight.
    """
    
    def __init__(self, frame_id, timestamp, pose, keypoints, descriptors, camera_matrix):
        """
        Initialize a frame.
        
        Args:
            frame_id: Unique ID for this frame
            timestamp: Timestamp of the frame
            pose: 4x4 pose matrix (camera to world transformation)
            keypoints: List of OpenCV keypoints
            descriptors: ORB descriptors for the keypoints
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.pose = pose.copy()  # 4x4 pose matrix
        self.keypoints = list(keypoints) if keypoints is not None else []
        self.descriptors = descriptors.copy() if descriptors is not None else None
        self.camera_matrix = camera_matrix.copy()
        
        # Inlier observations from pose recovery: {feature_id: (map_point, observation_2d)}
        # observation_2d is the 2D pixel coordinates of the observation
        self.inlier_observations = {}
        
        # Thread safety
        self.lock = Lock()
    
    def add_inlier_observation(self, feature_id, map_point, observation_2d):
        """
        Add an inlier observation from pose recovery.
        
        Args:
            feature_id: Index of the feature in this frame
            map_point: MapPoint object
            observation_2d: 2D pixel coordinates of the observation (numpy array [u, v])
        """
        with self.lock:
            self.inlier_observations[feature_id] = (map_point, observation_2d.copy())
    
    def remove_inlier_observation(self, feature_id):
        """
        Remove an inlier observation.
        
        Args:
            feature_id: Index of the feature
        """
        with self.lock:
            if feature_id in self.inlier_observations:
                del self.inlier_observations[feature_id]
    
    def get_inlier_observation(self, feature_id):
        """
        Get an inlier observation by feature ID.
        
        Args:
            feature_id: Index of the feature
            
        Returns:
            Tuple of (map_point, observation_2d) if exists, None otherwise
        """
        with self.lock:
            return self.inlier_observations.get(feature_id, None)
    
    def get_inlier_observations(self):
        """
        Get all inlier observations.
        
        Returns:
            Dictionary of {feature_id: (map_point, observation_2d)}
        """
        with self.lock:
            return self.inlier_observations.copy()
    
    def get_inlier_map_points(self):
        """
        Get all map points observed as inliers in this frame.
        
        Returns:
            List of MapPoint objects
        """
        with self.lock:
            return [obs[0] for obs in self.inlier_observations.values()]
    
    def get_inlier_observations_count(self):
        """
        Get the number of inlier observations.
        
        Returns:
            Number of inlier observations
        """
        with self.lock:
            return len(self.inlier_observations)
    
    def has_inlier_observation(self, feature_id):
        """
        Check if a feature has an inlier observation.
        
        Args:
            feature_id: Index of the feature
            
        Returns:
            True if feature has inlier observation, False otherwise
        """
        with self.lock:
            return feature_id in self.inlier_observations
    
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
            pose: New 4x4 pose matrix
        """
        with self.lock:
            self.pose = pose.copy()
    
    def get_camera_center(self):
        """
        Get the camera center in world coordinates.
        
        Returns:
            3D camera center position
        """
        with self.lock:
            return self.pose[:3, 3].copy()
    
    def get_rotation(self):
        """
        Get the rotation matrix of this frame.
        
        Returns:
            3x3 rotation matrix
        """
        with self.lock:
            return self.pose[:3, :3].copy()
    
    def get_translation(self):
        """
        Get the translation vector of this frame.
        
        Returns:
            3D translation vector
        """
        with self.lock:
            return self.pose[:3, 3].copy()
    
    def get_keypoint(self, feature_id):
        """
        Get a keypoint by feature ID.
        
        Args:
            feature_id: Index of the feature
            
        Returns:
            OpenCV keypoint if valid, None otherwise
        """
        with self.lock:
            if 0 <= feature_id < len(self.keypoints):
                return self.keypoints[feature_id]
            return None
    
    def get_keypoints(self):
        """
        Get all keypoints.
        
        Returns:
            List of OpenCV keypoints
        """
        with self.lock:
            return self.keypoints.copy()
    
    def get_descriptor(self, feature_id):
        """
        Get a descriptor by feature ID.
        
        Args:
            feature_id: Index of the feature
            
        Returns:
            ORB descriptor if valid, None otherwise
        """
        with self.lock:
            if self.descriptors is not None and 0 <= feature_id < len(self.descriptors):
                return self.descriptors[feature_id]
            return None
    
    def get_descriptors(self):
        """
        Get all descriptors.
        
        Returns:
            ORB descriptors array
        """
        with self.lock:
            return self.descriptors.copy() if self.descriptors is not None else None
    
    def project_point(self, point_3d):
        """
        Project a 3D point to 2D image coordinates.
        
        Args:
            point_3d: 3D point in world coordinates
            
        Returns:
            2D image coordinates (u, v) if valid, None otherwise
        """
        with self.lock:
            # Transform point to camera coordinates
            point_cam = np.linalg.inv(self.pose) @ np.append(point_3d, 1.0)
            point_cam = point_cam[:3] / point_cam[3]
            
            # Check if point is in front of camera
            if point_cam[2] <= 0:
                return None
            
            # Project to image coordinates
            point_2d = self.camera_matrix @ point_cam
            point_2d = point_2d[:2] / point_2d[2]
            
            return point_2d
    
    def is_in_frustum(self, point_3d, margin=1.0):
        """
        Check if a 3D point is in the viewing frustum of this frame.
        
        Args:
            point_3d: 3D point in world coordinates
            margin: Margin around the image boundaries
            
        Returns:
            True if in frustum, False otherwise
        """
        with self.lock:
            # Project point to image coordinates
            point_2d = self.project_point(point_3d)
            if point_2d is None:
                return False
            
            u, v = point_2d
            height, width = 480, 640  # Assuming standard image size
            
            # Check if point is within image boundaries with margin
            if u < -margin or u > width + margin or v < -margin or v > height + margin:
                return False
            
            # Check if point is in front of camera (already done in project_point)
            return True
    
    def clear_inlier_observations(self):
        """Clear all inlier observations."""
        with self.lock:
            self.inlier_observations.clear()
    
    def get_observation_2d(self, feature_id):
        """
        Get the 2D observation for a feature.
        
        Args:
            feature_id: Index of the feature
            
        Returns:
            2D observation coordinates if exists, None otherwise
        """
        with self.lock:
            if feature_id in self.inlier_observations:
                return self.inlier_observations[feature_id][1]
            return None 