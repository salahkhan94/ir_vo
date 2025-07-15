#!/usr/bin/env python3
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import threading

class Map:
    """
    Represents the global map with keyframes, map points, and covisibility graph.
    This is a core component of ORB-SLAM2 for managing the map structure.
    """
    
    def __init__(self):
        """Initialize an empty map."""
        # Map points
        self.mspMapPoints = set()  # Set of MapPoint objects
        self.mnMaxMPid = 0         # Maximum map point ID
        
        # Keyframes
        self.mspKeyFrames = set()  # Set of KeyFrame objects
        self.mnMaxKFid = 0         # Maximum keyframe ID
        
        # Covisibility graph
        self.mConnectedKeyFrameWeights = defaultdict(dict)  # keyframe_id -> {neighbor_id: weight}
        
        # Spanning tree
        self.mvpKeyFrameOrigins = []  # Root keyframes
        
        # Map mutex for thread safety
        self.mMutexMap = threading.RLock()
        
        # Map update flags
        self.mbMapUpdated = False
        self.mnMapChangeNotified = 0
        
    def __str__(self):
        return f"Map(keyframes={len(self.mspKeyFrames)}, mappoints={len(self.mspMapPoints)})"
    
    def __repr__(self):
        return self.__str__()
    
    def add_map_point(self, map_point):
        """
        Add a map point to the map.
        
        Args:
            map_point: MapPoint object to add
        """
        with self.mMutexMap:
            self.mspMapPoints.add(map_point)
            self.mnMaxMPid = max(self.mnMaxMPid, map_point.mnId)
            self.mbMapUpdated = True
    
    def add_keyframe(self, keyframe):
        """
        Add a keyframe to the map.
        
        Args:
            keyframe: KeyFrame object to add
        """
        with self.mMutexMap:
            self.mspKeyFrames.add(keyframe)
            self.mnMaxKFid = max(self.mnMaxKFid, keyframe.mnId)
            self.mbMapUpdated = True
    
    def erase_map_point(self, map_point):
        """
        Remove a map point from the map.
        
        Args:
            map_point: MapPoint object to remove
        """
        with self.mMutexMap:
            if map_point in self.mspMapPoints:
                self.mspMapPoints.remove(map_point)
                self.mbMapUpdated = True
    
    def erase_keyframe(self, keyframe):
        """
        Remove a keyframe from the map.
        
        Args:
            keyframe: KeyFrame object to remove
        """
        with self.mMutexMap:
            if keyframe in self.mspKeyFrames:
                self.mspKeyFrames.remove(keyframe)
                # Remove from covisibility graph
                if keyframe.mnId in self.mConnectedKeyFrameWeights:
                    del self.mConnectedKeyFrameWeights[keyframe.mnId]
                # Remove connections to this keyframe
                for kf_id in list(self.mConnectedKeyFrameWeights.keys()):
                    if keyframe.mnId in self.mConnectedKeyFrameWeights[kf_id]:
                        del self.mConnectedKeyFrameWeights[kf_id][keyframe.mnId]
                self.mbMapUpdated = True
    
    def get_map_points(self) -> Set:
        """
        Get all map points in the map.
        
        Returns:
            Set: Set of MapPoint objects
        """
        with self.mMutexMap:
            return self.mspMapPoints.copy()
    
    def get_keyframes(self) -> Set:
        """
        Get all keyframes in the map.
        
        Returns:
            Set: Set of KeyFrame objects
        """
        with self.mMutexMap:
            return self.mspKeyFrames.copy()
    
    def get_map_point_by_id(self, map_point_id: int):
        """
        Get a map point by its ID.
        
        Args:
            map_point_id (int): ID of the map point
            
        Returns:
            MapPoint or None: MapPoint object if found, None otherwise
        """
        with self.mMutexMap:
            for mp in self.mspMapPoints:
                if mp.mnId == map_point_id:
                    return mp
        return None
    
    def get_keyframe_by_id(self, keyframe_id: int):
        """
        Get a keyframe by its ID.
        
        Args:
            keyframe_id (int): ID of the keyframe
            
        Returns:
            KeyFrame or None: KeyFrame object if found, None otherwise
        """
        with self.mMutexMap:
            for kf in self.mspKeyFrames:
                if kf.mnId == keyframe_id:
                    return kf
        return None
    
    def get_map_points_count(self) -> int:
        """
        Get the number of map points.
        
        Returns:
            int: Number of map points
        """
        with self.mMutexMap:
            return len(self.mspMapPoints)
    
    def get_keyframes_count(self) -> int:
        """
        Get the number of keyframes.
        
        Returns:
            int: Number of keyframes
        """
        with self.mMutexMap:
            return len(self.mspKeyFrames)
    
    def add_connection(self, keyframe_id1: int, keyframe_id2: int, weight: int):
        """
        Add a connection between two keyframes in the covisibility graph.
        
        Args:
            keyframe_id1 (int): ID of first keyframe
            keyframe_id2 (int): ID of second keyframe
            weight (int): Weight of the connection (number of shared map points)
        """
        with self.mMutexMap:
            self.mConnectedKeyFrameWeights[keyframe_id1][keyframe_id2] = weight
            self.mConnectedKeyFrameWeights[keyframe_id2][keyframe_id1] = weight
    
    def get_connected_keyframes(self, keyframe_id: int) -> Dict[int, int]:
        """
        Get connected keyframes for a given keyframe.
        
        Args:
            keyframe_id (int): ID of the keyframe
            
        Returns:
            Dict[int, int]: Dictionary mapping neighbor keyframe IDs to weights
        """
        with self.mMutexMap:
            return self.mConnectedKeyFrameWeights.get(keyframe_id, {}).copy()
    
    def get_best_covisibles(self, keyframe_id: int, N: int = 10) -> List[Tuple[int, int]]:
        """
        Get the best N covisible keyframes for a given keyframe.
        
        Args:
            keyframe_id (int): ID of the keyframe
            N (int): Number of best covisible keyframes to return
            
        Returns:
            List[Tuple[int, int]]: List of (keyframe_id, weight) tuples sorted by weight
        """
        with self.mMutexMap:
            connections = self.mConnectedKeyFrameWeights.get(keyframe_id, {})
            sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)
            return sorted_connections[:N]
    
    def get_essential_graph(self, keyframe_id: int) -> Set[int]:
        """
        Get the essential graph for a keyframe (spanning tree + strong connections).
        
        Args:
            keyframe_id (int): ID of the keyframe
            
        Returns:
            Set[int]: Set of keyframe IDs in the essential graph
        """
        with self.mMutexMap:
            # This is a simplified version - in full ORB-SLAM2, this would include
            # spanning tree connections and strong covisible connections
            connections = self.mConnectedKeyFrameWeights.get(keyframe_id, {})
            essential_graph = {keyframe_id}
            
            # Add strongly connected keyframes (weight > 100)
            for neighbor_id, weight in connections.items():
                if weight > 100:
                    essential_graph.add(neighbor_id)
            
            return essential_graph
    
    def clear(self):
        """Clear the entire map."""
        with self.mMutexMap:
            self.mspMapPoints.clear()
            self.mspKeyFrames.clear()
            self.mConnectedKeyFrameWeights.clear()
            self.mvpKeyFrameOrigins.clear()
            self.mnMaxMPid = 0
            self.mnMaxKFid = 0
            self.mbMapUpdated = True
    
    def is_empty(self) -> bool:
        """
        Check if the map is empty.
        
        Returns:
            bool: True if map is empty, False otherwise
        """
        with self.mMutexMap:
            return len(self.mspMapPoints) == 0 and len(self.mspKeyFrames) == 0
    
    def get_reference_map_points(self) -> List:
        """
        Get reference map points (well-observed map points).
        
        Returns:
            List: List of reference MapPoint objects
        """
        with self.mMutexMap:
            reference_mps = []
            for mp in self.mspMapPoints:
                if mp.get_observation_count() >= 3 and not mp.is_bad():
                    reference_mps.append(mp)
            return reference_mps
    
    def get_all_map_points(self) -> List:
        """
        Get all map points as a list.
        
        Returns:
            List: List of all MapPoint objects
        """
        with self.mMutexMap:
            return list(self.mspMapPoints)
    
    def get_all_keyframes(self) -> List:
        """
        Get all keyframes as a list.
        
        Returns:
            List: List of all KeyFrame objects
        """
        with self.mMutexMap:
            return list(self.mspKeyFrames)
    
    def set_map_updated(self):
        """Mark the map as updated."""
        with self.mMutexMap:
            self.mbMapUpdated = True
    
    def is_map_updated(self) -> bool:
        """
        Check if the map has been updated.
        
        Returns:
            bool: True if map has been updated, False otherwise
        """
        with self.mMutexMap:
            return self.mbMapUpdated
    
    def get_map_change_index(self) -> int:
        """
        Get the map change notification index.
        
        Returns:
            int: Map change notification index
        """
        with self.mMutexMap:
            return self.mnMapChangeNotified
    
    def inform_new_big_change(self):
        """Inform that a big change has occurred in the map."""
        with self.mMutexMap:
            self.mnMapChangeNotified += 1
    
    def get_last_big_change_idx(self) -> int:
        """
        Get the index of the last big change.
        
        Returns:
            int: Index of the last big change
        """
        with self.mMutexMap:
            return self.mnMapChangeNotified 