#!/usr/bin/env python3
import numpy as np
from threading import Lock
from .mappoint import MapPoint
from .keyframe import KeyFrame

class Map:
    """
    Map class representing the entire SLAM map.
    Similar to ORB-SLAM2 implementation.
    """
    
    def __init__(self):
        """Initialize the map."""
        # Map points and keyframes
        self.map_points = {}  # {map_point_id: MapPoint}
        self.keyframes = {}   # {keyframe_id: KeyFrame}
        
        # Reference keyframes for tracking
        self.reference_keyframe = None
        
        # Map statistics
        self.max_map_point_id = 0
        self.max_keyframe_id = 0
        
        # Thread safety
        self.lock = Lock()
    
    def add_map_point(self, map_point):
        """
        Add a map point to the map.
        
        Args:
            map_point: MapPoint object to add
        """
        with self.lock:
            map_point_id = self.max_map_point_id
            self.map_points[map_point_id] = map_point
            self.max_map_point_id += 1
            return map_point_id
    
    def add_keyframe(self, keyframe):
        """
        Add a keyframe to the map.
        
        Args:
            keyframe: KeyFrame object to add
        """
        with self.lock:
            keyframe_id = self.max_keyframe_id
            self.keyframes[keyframe_id] = keyframe
            self.max_keyframe_id += 1
            return keyframe_id
    
    def remove_map_point(self, map_point_id):
        """
        Remove a map point from the map.
        
        Args:
            map_point_id: ID of the map point to remove
        """
        # Note: This method should be called while the map lock is already held
        if map_point_id in self.map_points:
            map_point = self.map_points[map_point_id]
            
            # Remove observations from keyframes
            observations = map_point.get_observations()
            for keyframe_id, feature_id in observations.items():
                if keyframe_id in self.keyframes:
                    self.keyframes[keyframe_id].remove_map_point(feature_id)
            
            # Mark map point as bad
            map_point.set_bad()
            del self.map_points[map_point_id]
    
    def remove_keyframe(self, keyframe_id):
        """
        Remove a keyframe from the map.
        
        Args:
            keyframe_id: ID of the keyframe to remove
        """
        with self.lock:
            if keyframe_id in self.keyframes:
                keyframe = self.keyframes[keyframe_id]
                
                # Remove map point associations
                map_points = keyframe.get_map_points()
                for feature_id, map_point in map_points.items():
                    if map_point is not None:
                        map_point.remove_observation(keyframe_id)
                        
                        # Remove map point if it has no observations
                        if map_point.get_observations_count() == 0:
                            map_point.set_bad()
                
                del self.keyframes[keyframe_id]
    
    def get_map_points(self):
        """
        Get all map points.
        
        Returns:
            Dictionary of {map_point_id: MapPoint}
        """
        with self.lock:
            return self.map_points.copy()
    
    def get_keyframes(self):
        """
        Get all keyframes.
        
        Returns:
            Dictionary of {keyframe_id: KeyFrame}
        """
        with self.lock:
            return self.keyframes.copy()
    
    def get_map_point(self, map_point_id):
        """
        Get a specific map point.
        
        Args:
            map_point_id: ID of the map point
            
        Returns:
            MapPoint object if exists, None otherwise
        """
        with self.lock:
            return self.map_points.get(map_point_id, None)
    
    def get_keyframe(self, keyframe_id):
        """
        Get a specific keyframe.
        
        Args:
            keyframe_id: ID of the keyframe
            
        Returns:
            KeyFrame object if exists, None otherwise
        """
        with self.lock:
            return self.keyframes.get(keyframe_id, None)
    
    def set_reference_keyframe(self, keyframe):
        """
        Set the reference keyframe for tracking.
        
        Args:
            keyframe: KeyFrame object to set as reference
        """
        with self.lock:
            self.reference_keyframe = keyframe
    
    def get_reference_keyframe(self):
        """
        Get the reference keyframe.
        
        Returns:
            Reference KeyFrame object
        """
        with self.lock:
            return self.reference_keyframe
    
    def cull_map_points(self, max_observations=2, max_age=10):
        """
        Cull bad map points from the map.
        
        Args:
            max_observations: Maximum number of observations for a point to be considered bad
            max_age: Maximum age (in frames) for a point to be considered bad
        """
        import rospy
        rospy.loginfo("  🔍 Starting map point culling...")
        
        with self.lock:
            rospy.loginfo(f"  🔍 Acquired map lock, processing {len(self.map_points)} map points")
            map_points_to_remove = []
            
            for map_point_id, map_point in self.map_points.items():
                # Check if map point is bad
                if map_point.is_bad_point():
                    map_points_to_remove.append(map_point_id)
                    continue
                
                # Check observation count
                observations_count = map_point.get_observations_count()
                if observations_count <= max_observations:
                    map_points_to_remove.append(map_point_id)
                    continue
                
                # Check age (simplified - could be more sophisticated)
                # For now, we'll use a simple heuristic based on observations
                if observations_count < 3 and len(map_point.get_observations()) < 2:
                    map_points_to_remove.append(map_point_id)
            
            rospy.loginfo(f"  🔍 Found {len(map_points_to_remove)} map points to remove")
            
            # Remove bad map points
            for i, map_point_id in enumerate(map_points_to_remove):
                # rospy.loginfo(f"  🔍 Removing map point {i+1}/{len(map_points_to_remove)}: ID {map_point_id}")
                self.remove_map_point(map_point_id)
            
            rospy.loginfo("  🔍 Map point culling completed")
    
    def get_recent_keyframes(self, n=10):
        """
        Get the most recent N keyframes.
        
        Args:
            n: Number of recent keyframes to return
            
        Returns:
            List of recent KeyFrame objects
        """
        with self.lock:
            keyframe_list = list(self.keyframes.values())
            keyframe_list.sort(key=lambda kf: kf.frame_id, reverse=True)
            return keyframe_list[:n]
    
    def get_map_points_for_optimization(self, keyframe_ids):
        """
        Get map points observed by the specified keyframes.
        
        Args:
            keyframe_ids: List of keyframe IDs
            
        Returns:
            List of MapPoint objects
        """
        with self.lock:
            map_points = set()
            
            for keyframe_id in keyframe_ids:
                if keyframe_id in self.keyframes:
                    keyframe = self.keyframes[keyframe_id]
                    keyframe_map_points = keyframe.get_map_points()
                    for map_point in keyframe_map_points.values():
                        if map_point is not None and not map_point.is_bad_point():
                            map_points.add(map_point)
            
            return list(map_points)
    
    def get_observations_for_optimization(self, keyframe_ids):
        """
        Get all 3D-2D observations for optimization.
        
        Args:
            keyframe_ids: List of keyframe IDs
            
        Returns:
            Dictionary of {keyframe_id: {feature_id: (map_point, keypoint)}}
        """
        with self.lock:
            observations = {}
            
            for keyframe_id in keyframe_ids:
                if keyframe_id in self.keyframes:
                    keyframe = self.keyframes[keyframe_id]
                    keyframe_observations = {}
                    
                    map_points = keyframe.get_map_points()
                    for feature_id, map_point in map_points.items():
                        if map_point is not None and not map_point.is_bad_point():
                            keypoint = keyframe.get_keypoint(feature_id)
                            if keypoint is not None:
                                keyframe_observations[feature_id] = (map_point, keypoint)
                    
                    if keyframe_observations:
                        observations[keyframe_id] = keyframe_observations
            
            return observations
    
    def update_map_point_positions(self, map_point_positions):
        """
        Update map point positions after optimization.
        
        Args:
            map_point_positions: Dictionary of {map_point_id: new_position}
        """
        with self.lock:
            for map_point_id, new_position in map_point_positions.items():
                if map_point_id in self.map_points:
                    self.map_points[map_point_id].set_position(new_position)
    
    def update_keyframe_poses(self, keyframe_poses):
        """
        Update keyframe poses after optimization.
        
        Args:
            keyframe_poses: Dictionary of {keyframe_id: new_pose}
        """
        with self.lock:
            for keyframe_id, new_pose in keyframe_poses.items():
                if keyframe_id in self.keyframes:
                    self.keyframes[keyframe_id].set_pose(new_pose)
    
    def get_map_size(self):
        """
        Get the size of the map.
        
        Returns:
            Tuple of (num_map_points, num_keyframes)
        """
        with self.lock:
            return len(self.map_points), len(self.keyframes)
    
    def clear(self):
        """Clear the entire map."""
        with self.lock:
            self.map_points.clear()
            self.keyframes.clear()
            self.reference_keyframe = None
            self.max_map_point_id = 0
            self.max_keyframe_id = 0 