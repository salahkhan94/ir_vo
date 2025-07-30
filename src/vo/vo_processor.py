#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from vo.features.detectors import build_detector
from vo.features.descriptors import compute_descriptors
from vo.features.matcher import match_features, get_matched_points
from vo.estimators.pose_estimator import (
    estimate_fundamental_matrix, 
    estimate_essential_matrix, 
    recover_pose, 
    camera_info_to_K
)

class VOProcessor:
    """
    Simple Visual Odometry Processor for monocular camera.
    Handles feature detection, matching, and pose estimation.
    """
    
    def __init__(self, detector_type="orb"):
        """
        Initialize VO processor.
        
        Args:
            detector_type: Type of feature detector to use
        """
        self.detector = build_detector(detector_type)
        self.detector_type = detector_type
        
        # Frame storage
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_camera_info = None
        
        # Pose tracking
        self.current_pose = np.eye(4)  # Identity matrix as initial pose
        self.pose_history = []  # List to store pose history for trajectory
        
        # Publishers
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher("~path", Path, queue_size=10)
        self.tf_broadcaster = TransformBroadcaster()
        
        # Parameters
        self.min_matches = 50  # Minimum number of matches required
        self.min_inliers = 15  # Minimum number of inliers required (lowered for KITTI)
        
    def process_frame(self, frame, camera_info):
        """
        Process a new frame and estimate pose.
        
        Args:
            frame: Current frame (OpenCV image)
            camera_info: Camera info message
            
        Returns:
            success: Whether pose estimation was successful
            pose: Estimated pose matrix (4x4) if successful, None otherwise
        """
        # Extract camera intrinsics
        K = camera_info_to_K(camera_info)
        
        # Detect features and compute descriptors
        keypoints = self.detector.detect(frame, None)
        if len(keypoints) < 100:  # Need sufficient features
            return False, None
            
        keypoints, descriptors = compute_descriptors(self.detector, frame, keypoints)
        
        # If this is the first frame, store it and return
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_camera_info = camera_info
            return False, None
        
        # Match features between current and previous frame
        matches, good_matches = match_features(
            self.prev_descriptors, descriptors, self.detector_type
        )
        
        if len(good_matches) < self.min_matches:
            # Update previous frame and return
            self.prev_frame = frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_camera_info = camera_info
            return False, None
        
        # Extract matched points
        pts1, pts2 = get_matched_points(self.prev_keypoints, keypoints, good_matches)
        
        # Estimate Fundamental Matrix
        F, F_mask = estimate_fundamental_matrix(pts1, pts2)
        if F is None:
            return False, None
        
        # Count inliers from Fundamental Matrix
        F_inliers = np.sum(F_mask)
        if F_inliers < self.min_inliers:
            return False, None
        
        # Estimate Essential Matrix
        E, E_mask = estimate_essential_matrix(pts1, pts2, K)
        if E is None:
            return False, None
        
        # Count inliers from Essential Matrix
        E_inliers = np.sum(E_mask)
        if E_inliers < self.min_inliers:
            return False, None
        
        # Recover pose from Essential Matrix
        R, t, pose_mask = recover_pose(E, pts1, pts2, K)
        if R is None or t is None:
            return False, None
        
        # Count inliers from pose recovery
        pose_inliers = np.sum(pose_mask)
        if pose_inliers < self.min_inliers:
            return False, None
        
        # Construct pose matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        pose_matrix[:3, 3] = t.flatten()
        
        # Update current pose (accumulate transformations)
        self.current_pose = self.current_pose @ pose_matrix
        
        # Publish pose and path
        self._publish_pose(self.current_pose, camera_info.header)
        self._publish_path(self.current_pose, camera_info.header)
        
        # Store current frame as previous frame
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_camera_info = camera_info
        
        return True, self.current_pose
    
    def _publish_pose(self, pose_matrix, header):
        """
        Publish pose as ROS message.
        
        Args:
            pose_matrix: 4x4 pose matrix
            header: Header from camera info message
        """
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "world"
        
        # Extract position
        pose_msg.pose.position.x = pose_matrix[0, 3]
        pose_msg.pose.position.y = pose_matrix[1, 3]
        pose_msg.pose.position.z = pose_matrix[2, 3]
        
        # Extract orientation (convert rotation matrix to quaternion)
        R = pose_matrix[:3, :3]
        # Use Rodrigues formula to get rotation vector, then convert to quaternion
        rvec, _ = cv2.Rodrigues(R)
        # Simple conversion to quaternion (assuming small rotations)
        angle = np.linalg.norm(rvec)
        if angle > 1e-6:
            axis = rvec.flatten() / angle
            pose_msg.pose.orientation.x = axis[0] * np.sin(angle/2)
            pose_msg.pose.orientation.y = axis[1] * np.sin(angle/2)
            pose_msg.pose.orientation.z = axis[2] * np.sin(angle/2)
            pose_msg.pose.orientation.w = np.cos(angle/2)
        else:
            pose_msg.pose.orientation.w = 1.0
        
        # Publish pose
        self.pose_pub.publish(pose_msg)
        
        # Publish transform from world to camera frame
        transform = TransformStamped()
        transform.header = header
        transform.header.frame_id = "world"
        transform.child_frame_id = "camera_frame"
        transform.transform.translation.x = pose_msg.pose.position.x
        transform.transform.translation.y = pose_msg.pose.position.y
        transform.transform.translation.z = pose_msg.pose.position.z
        transform.transform.rotation = pose_msg.pose.orientation
        
        self.tf_broadcaster.sendTransform(transform)
    
    def _publish_path(self, pose_matrix, header):
        """
        Publish trajectory path as ROS message.
        
        Args:
            pose_matrix: 4x4 pose matrix
            header: Header from camera info message
        """
        # Add current pose to history
        self.pose_history.append(pose_matrix.copy())
        
        # Create Path message
        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = "world"
        path_msg.poses = []  # Explicitly initialize poses list
        
        # Add all poses to path
        for pose in self.pose_history:
            pose_stamped = PoseStamped()
            pose_stamped.header = header
            pose_stamped.header.frame_id = "world"
            
            # Extract position
            pose_stamped.pose.position.x = pose[0, 3]
            pose_stamped.pose.position.y = pose[1, 3]
            pose_stamped.pose.position.z = pose[2, 3]
            
            # Extract orientation
            R = pose[:3, :3]
            rvec, _ = cv2.Rodrigues(R)
            angle = np.linalg.norm(rvec)
            if angle > 1e-6:
                axis = rvec.flatten() / angle
                pose_stamped.pose.orientation.x = axis[0] * np.sin(angle/2)
                pose_stamped.pose.orientation.y = axis[1] * np.sin(angle/2)
                pose_stamped.pose.orientation.z = axis[2] * np.sin(angle/2)
                pose_stamped.pose.orientation.w = np.cos(angle/2)
            else:
                pose_stamped.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose_stamped)
        
        # Publish path
        self.path_pub.publish(path_msg) 