#!/usr/bin/env python3
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Set
import threading
from collections import defaultdict

class KeyFrame:
    """
    Represents a keyframe with keypoints, descriptors, pose, and map point associations.
    This is a core component of ORB-SLAM2 for storing and managing keyframe data.
    """
    
    def __init__(self, frame_id: int, timestamp: float, camera, keypoints: List, descriptors: np.ndarray,
                 pose: np.ndarray, keyframe_id: int):
        """
        Initialize a KeyFrame.
        
        Args:
            frame_id (int): ID of the original frame
            timestamp (float): Timestamp of the keyframe
            camera: Camera object with intrinsics
            keypoints (List): List of KeyPoint objects
            descriptors (np.ndarray): ORB descriptors for keypoints
            pose (np.ndarray): 4x4 transformation matrix (camera to world)
            keyframe_id (int): Unique ID for this keyframe
        """
        
        # Basic information
        self.mnFrameId = frame_id
        self.mTimeStamp = timestamp
        self.mnId = keyframe_id
        
        # Camera and image data
        self.mCamera = camera
        self.mvKeys = keypoints  # List of KeyPoint objects
        self.mDescriptors = descriptors  # ORB descriptors
        
        # Pose information
        self.mTcw = pose.astype(np.float32)  # Camera to world transformation
        self.mTwc = np.linalg.inv(self.mTcw).astype(np.float32)  # World to camera transformation
        
        # Map point associations
        self.mvpMapPoints = [None] * len(keypoints)  # MapPoint objects for each keypoint
        
        # Covisibility graph
        self.mConnectedKeyFrameWeights = {}  # keyframe_id -> weight
        self.mvpOrderedConnectedKeyFrames = []  # Ordered list of connected keyframes
        self.mvOrderedWeights = []  # Weights corresponding to ordered keyframes
        
        # Spanning tree
        self.mbFirstConnection = True
        self.mpParent = None  # Parent keyframe in spanning tree
        self.mspChildrens = set()  # Children keyframes in spanning tree
        self.mspLoopEdges = set()  # Loop closure edges
        
        # BoW (Bag of Words) representation
        self.mBowVec = None
        self.mFeatVec = None
        
        # Scale pyramid information
        self.mnScaleLevels = 8
        self.mfScaleFactor = 1.2
        self.mfLogScaleFactor = np.log(self.mfScaleFactor)
        self.mvScaleFactors = [self.mfScaleFactor ** i for i in range(self.mnScaleLevels)]
        self.mvInvScaleFactors = [1.0 / sf for sf in self.mvScaleFactors]
        self.mvLevelSigma2 = [self.mvScaleFactors[i] ** 2 for i in range(self.mnScaleLevels)]
        self.mvInvLevelSigma2 = [1.0 / s2 for s2 in self.mvLevelSigma2]
        
        # Undistorted keypoints
        self.mvKeysUn = []
        self._compute_undistorted_keypoints()
        
        # Mutex for thread safety
        self.mMutexPose = threading.RLock()
        self.mMutexConnections = threading.RLock()
        self.mMutexFeatures = threading.RLock()
        
    def __str__(self):
        return f"KeyFrame(id={self.mnId}, frame_id={self.mnFrameId}, keypoints={len(self.mvKeys)})"
    
    def __repr__(self):
        return self.__str__()
    
    def _compute_undistorted_keypoints(self):
        """Compute undistorted keypoints."""
        if len(self.mvKeys) == 0:
            return
        
        # Extract keypoint coordinates
        points = np.array([[kp.x, kp.y] for kp in self.mvKeys])
        
        # Undistort points
        undistorted_points = self.mCamera.undistort_points(points)
        
        # Create undistorted keypoints
        self.mvKeysUn = []
        for i, (kp, undist_pt) in enumerate(zip(self.mvKeys, undistorted_points)):
            undist_kp = cv2.KeyPoint(
                x=undist_pt[0], y=undist_pt[1],
                size=kp.size, angle=kp.angle,
                response=kp.response, octave=kp.octave,
                class_id=kp.class_id
            )
            self.mvKeysUn.append(undist_kp)
    
    def get_keypoints(self) -> List:
        """
        Get all keypoints.
        
        Returns:
            List: List of KeyPoint objects
        """
        return self.mvKeys.copy()
    
    def get_undistorted_keypoints(self) -> List:
        """
        Get undistorted keypoints.
        
        Returns:
            List: List of undistorted KeyPoint objects
        """
        return self.mvKeysUn.copy()
    
    def get_keypoint(self, idx: int):
        """
        Get a specific keypoint.
        
        Args:
            idx (int): Index of the keypoint
            
        Returns:
            KeyPoint: KeyPoint object at the given index
        """
        if 0 <= idx < len(self.mvKeys):
            return self.mvKeys[idx]
        return None
    
    def get_undistorted_keypoint(self, idx: int):
        """
        Get a specific undistorted keypoint.
        
        Args:
            idx (int): Index of the keypoint
            
        Returns:
            cv2.KeyPoint: Undistorted KeyPoint object at the given index
        """
        if 0 <= idx < len(self.mvKeysUn):
            return self.mvKeysUn[idx]
        return None
    
    def get_descriptors(self) -> np.ndarray:
        """
        Get all descriptors.
        
        Returns:
            np.ndarray: ORB descriptors
        """
        return self.mDescriptors.copy()
    
    def get_descriptor(self, idx: int) -> Optional[np.ndarray]:
        """
        Get a specific descriptor.
        
        Args:
            idx (int): Index of the descriptor
            
        Returns:
            np.ndarray or None: Descriptor at the given index
        """
        if 0 <= idx < len(self.mDescriptors):
            return self.mDescriptors[idx].copy()
        return None
    
    def get_pose(self) -> np.ndarray:
        """
        Get the camera to world transformation.
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        with self.mMutexPose:
            return self.mTcw.copy()
    
    def get_inverse_pose(self) -> np.ndarray:
        """
        Get the world to camera transformation.
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        with self.mMutexPose:
            return self.mTwc.copy()
    
    def set_pose(self, pose: np.ndarray):
        """
        Set the camera to world transformation.
        
        Args:
            pose (np.ndarray): 4x4 transformation matrix
        """
        with self.mMutexPose:
            self.mTcw = pose.astype(np.float32)
            self.mTwc = np.linalg.inv(self.mTcw).astype(np.float32)
    
    def get_rotation(self) -> np.ndarray:
        """
        Get the rotation matrix.
        
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        with self.mMutexPose:
            return self.mTcw[:3, :3].copy()
    
    def get_translation(self) -> np.ndarray:
        """
        Get the translation vector.
        
        Returns:
            np.ndarray: 3x1 translation vector
        """
        with self.mMutexPose:
            return self.mTcw[:3, 3].copy()
    
    def get_center(self) -> np.ndarray:
        """
        Get the camera center in world coordinates.
        
        Returns:
            np.ndarray: 3x1 camera center vector
        """
        with self.mMutexPose:
            return self.mTwc[:3, 3].copy()
    
    def get_map_point(self, idx: int):
        """
        Get the map point associated with a keypoint.
        
        Args:
            idx (int): Index of the keypoint
            
        Returns:
            MapPoint or None: Associated map point
        """
        if 0 <= idx < len(self.mvpMapPoints):
            return self.mvpMapPoints[idx]
        return None
    
    def set_map_point(self, idx: int, map_point):
        """
        Set the map point associated with a keypoint.
        
        Args:
            idx (int): Index of the keypoint
            map_point: MapPoint object to associate
        """
        if 0 <= idx < len(self.mvpMapPoints):
            self.mvpMapPoints[idx] = map_point
    
    def get_map_points(self) -> List:
        """
        Get all associated map points.
        
        Returns:
            List: List of MapPoint objects
        """
        return self.mvpMapPoints.copy()
    
    def get_observed_map_points(self) -> List:
        """
        Get all observed map points (non-None).
        
        Returns:
            List: List of observed MapPoint objects
        """
        return [mp for mp in self.mvpMapPoints if mp is not None]
    
    def add_connection(self, keyframe, weight: int):
        """
        Add a connection to another keyframe.
        
        Args:
            keyframe: KeyFrame object to connect to
            weight (int): Weight of the connection
        """
        with self.mMutexConnections:
            self.mConnectedKeyFrameWeights[keyframe.mnId] = weight
    
    def erase_connection(self, keyframe):
        """
        Remove a connection to another keyframe.
        
        Args:
            keyframe: KeyFrame object to disconnect from
        """
        with self.mMutexConnections:
            if keyframe.mnId in self.mConnectedKeyFrameWeights:
                del self.mConnectedKeyFrameWeights[keyframe.mnId]
    
    def get_connected_keyframes(self) -> Dict[int, int]:
        """
        Get all connected keyframes with weights.
        
        Returns:
            Dict[int, int]: Dictionary mapping keyframe_id to weight
        """
        with self.mMutexConnections:
            return self.mConnectedKeyFrameWeights.copy()
    
    def get_ordered_connected_keyframes(self) -> List:
        """
        Get ordered list of connected keyframes.
        
        Returns:
            List: Ordered list of connected KeyFrame objects
        """
        with self.mMutexConnections:
            return self.mvpOrderedConnectedKeyFrames.copy()
    
    def get_ordered_weights(self) -> List[int]:
        """
        Get ordered list of connection weights.
        
        Returns:
            List[int]: Ordered list of weights
        """
        with self.mMutexConnections:
            return self.mvOrderedWeights.copy()
    
    def update_connections(self):
        """Update the ordered list of connected keyframes based on weights."""
        with self.mMutexConnections:
            # Sort connections by weight
            sorted_connections = sorted(
                self.mConnectedKeyFrameWeights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Update ordered lists
            self.mvpOrderedConnectedKeyFrames = []
            self.mvOrderedWeights = []
            
            for keyframe_id, weight in sorted_connections:
                # Find the keyframe object (this would need access to the map)
                # For now, we'll store the IDs
                self.mvpOrderedConnectedKeyFrames.append(keyframe_id)
                self.mvOrderedWeights.append(weight)
    
    def get_best_covisibles(self, N: int = 10) -> List:
        """
        Get the best N covisible keyframes.
        
        Args:
            N (int): Number of best covisible keyframes to return
            
        Returns:
            List: List of best covisible KeyFrame objects
        """
        with self.mMutexConnections:
            return self.mvpOrderedConnectedKeyFrames[:N]
    
    def get_covisibles_by_weight(self, min_weight: int) -> List:
        """
        Get covisible keyframes with weight >= min_weight.
        
        Args:
            min_weight (int): Minimum weight threshold
            
        Returns:
            List: List of KeyFrame objects with sufficient weight
        """
        with self.mMutexConnections:
            covisible = []
            for keyframe_id, weight in self.mConnectedKeyFrameWeights.items():
                if weight >= min_weight:
                    # Find the keyframe object (this would need access to the map)
                    covisible.append(keyframe_id)
            return covisible
    
    def set_parent(self, parent_keyframe):
        """
        Set the parent keyframe in the spanning tree.
        
        Args:
            parent_keyframe: Parent KeyFrame object
        """
        with self.mMutexConnections:
            self.mpParent = parent_keyframe
    
    def get_parent(self):
        """
        Get the parent keyframe.
        
        Returns:
            KeyFrame or None: Parent keyframe
        """
        with self.mMutexConnections:
            return self.mpParent
    
    def add_child(self, child_keyframe):
        """
        Add a child keyframe.
        
        Args:
            child_keyframe: Child KeyFrame object
        """
        with self.mMutexConnections:
            self.mspChildrens.add(child_keyframe)
    
    def erase_child(self, child_keyframe):
        """
        Remove a child keyframe.
        
        Args:
            child_keyframe: Child KeyFrame object
        """
        with self.mMutexConnections:
            if child_keyframe in self.mspChildrens:
                self.mspChildrens.remove(child_keyframe)
    
    def get_children(self) -> Set:
        """
        Get all children keyframes.
        
        Returns:
            Set: Set of child KeyFrame objects
        """
        with self.mMutexConnections:
            return self.mspChildrens.copy()
    
    def add_loop_edge(self, loop_keyframe):
        """
        Add a loop closure edge.
        
        Args:
            loop_keyframe: KeyFrame object for loop closure
        """
        with self.mMutexConnections:
            self.mspLoopEdges.add(loop_keyframe)
    
    def get_loop_edges(self) -> Set:
        """
        Get all loop closure edges.
        
        Returns:
            Set: Set of loop closure KeyFrame objects
        """
        with self.mMutexConnections:
            return self.mspLoopEdges.copy()
    
    def is_bad(self) -> bool:
        """
        Check if this keyframe is bad.
        
        Returns:
            bool: True if bad, False otherwise
        """
        # A keyframe is considered bad if it has no children and is not the root
        with self.mMutexConnections:
            return len(self.mspChildrens) == 0 and self.mpParent is not None
    
    def get_scale_factor(self, level: int) -> float:
        """
        Get the scale factor for a given pyramid level.
        
        Args:
            level (int): Pyramid level
            
        Returns:
            float: Scale factor
        """
        if 0 <= level < self.mnScaleLevels:
            return self.mvScaleFactors[level]
        return 1.0
    
    def get_inv_scale_factor(self, level: int) -> float:
        """
        Get the inverse scale factor for a given pyramid level.
        
        Args:
            level (int): Pyramid level
            
        Returns:
            float: Inverse scale factor
        """
        if 0 <= level < self.mnScaleLevels:
            return self.mvInvScaleFactors[level]
        return 1.0
    
    def get_level_sigma2(self, level: int) -> float:
        """
        Get the sigma squared for a given pyramid level.
        
        Args:
            level (int): Pyramid level
            
        Returns:
            float: Sigma squared
        """
        if 0 <= level < self.mnScaleLevels:
            return self.mvLevelSigma2[level]
        return 1.0
    
    def get_inv_level_sigma2(self, level: int) -> float:
        """
        Get the inverse sigma squared for a given pyramid level.
        
        Args:
            level (int): Pyramid level
            
        Returns:
            float: Inverse sigma squared
        """
        if 0 <= level < self.mnScaleLevels:
            return self.mvInvLevelSigma2[level]
        return 1.0 