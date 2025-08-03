#!/usr/bin/env python3
import numpy as np
import cv2
from threading import Lock

class KeyFrame:
    """
    KeyFrame class representing a keyframe in the SLAM system.
    Similar to ORB-SLAM2 implementation.
    """
    
    def __init__(self, frame_id, timestamp, pose, keypoints, descriptors, camera_matrix):
        """
        Initialize a keyframe.
        
        Args:
            frame_id: Unique ID for this keyframe
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
        
        # Map point associations: {feature_id: map_point}
        self.map_points = {}
        
        # Covisible keyframes: {keyframe_id: weight}
        self.covisible_keyframes = {}
        
        # BoW (Bag of Words) representation (placeholder for future)
        self.bow_vec = None
        self.feature_vec = None
        
        # Scale pyramid info
        self.scale_levels = 8
        self.scale_factor = 1.2
        self.log_scale_factor = np.log(self.scale_factor)
        
        # Thread safety
        self.lock = Lock()
    
    def add_map_point(self, feature_id, map_point):
        """
        Associate a map point with a feature in this keyframe.
        
        Args:
            feature_id: Index of the feature in this keyframe
            map_point: MapPoint object
        """
        with self.lock:
            self.map_points[feature_id] = map_point
    
    def remove_map_point(self, feature_id):
        """
        Remove a map point association.
        
        Args:
            feature_id: Index of the feature
        """
        with self.lock:
            if feature_id in self.map_points:
                del self.map_points[feature_id]
    
    def get_map_point(self, feature_id):
        """
        Get the map point associated with a feature.
        
        Args:
            feature_id: Index of the feature
            
        Returns:
            MapPoint object if associated, None otherwise
        """
        with self.lock:
            return self.map_points.get(feature_id, None)
    
    def get_map_points(self):
        """
        Get all map point associations.
        
        Returns:
            Dictionary of {feature_id: map_point}
        """
        with self.lock:
            return self.map_points.copy()
    
    def has_map_point(self, feature_id):
        """
        Check if a feature has an associated map point.
        
        Args:
            feature_id: Index of the feature
            
        Returns:
            True if feature has associated map point, False otherwise
        """
        with self.lock:
            return feature_id in self.map_points
    
    def get_pose(self):
        """
        Get the pose of this keyframe.
        
        Returns:
            4x4 pose matrix
        """
        with self.lock:
            return self.pose.copy()
    
    def set_pose(self, pose):
        """
        Set the pose of this keyframe.
        
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
        Get the rotation matrix of this keyframe.
        
        Returns:
            3x3 rotation matrix
        """
        with self.lock:
            return self.pose[:3, :3].copy()
    
    def get_translation(self):
        """
        Get the translation vector of this keyframe.
        
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
        Check if a 3D point is in the viewing frustum of this keyframe.
        
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
    
    def add_covisible_keyframe(self, keyframe_id, weight=1):
        """
        Add a covisible keyframe.
        
        Args:
            keyframe_id: ID of the covisible keyframe
            weight: Weight of the covisibility
        """
        with self.lock:
            self.covisible_keyframes[keyframe_id] = weight
    
    def get_covisible_keyframes(self):
        """
        Get all covisible keyframes.
        
        Returns:
            Dictionary of {keyframe_id: weight}
        """
        with self.lock:
            return self.covisible_keyframes.copy()
    
    def get_scale_factor(self, level):
        """
        Get the scale factor for a given pyramid level.
        
        Args:
            level: Pyramid level
            
        Returns:
            Scale factor
        """
        return self.scale_factor ** level
    
    def get_log_scale_factor(self):
        """
        Get the log of the scale factor.
        
        Returns:
            Log of scale factor
        """
        return self.log_scale_factor 