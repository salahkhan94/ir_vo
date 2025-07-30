#!/usr/bin/env python3
import numpy as np
import cv2
import rospy
import threading
import time
from scipy.optimize import least_squares
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
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
from vo.geometry.map import Map
from vo.geometry.keyframe import KeyFrame
from vo.geometry.mappoint import MapPoint

class SLAMSystem:
    """
    Full-fledged Visual Odometry system with tracking and mapping threads.
    Similar to ORB-SLAM2 architecture.
    """
    
    def __init__(self, detector_type="orb"):
        """
        Initialize the SLAM system.
        
        Args:
            detector_type: Type of feature detector to use
        """
        # Feature detection
        self.detector = build_detector(detector_type)
        self.detector_type = detector_type
        
        # Map and keyframes
        self.map = Map()
        self.current_keyframe = None
        self.reference_keyframe = None
        
        # Frame tracking
        self.current_frame_id = 0
        self.last_keyframe_id = 0
        self.keyframe_interval = 10  # Create keyframe every N frames
        
        # Tracking state
        self.tracking_state = "NO_IMAGES_YET"  # NO_IMAGES_YET, NOT_INITIALIZED, OK, LOST
        self.current_pose = np.eye(4)
        
        # Publishers
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher("~path", Path, queue_size=10)
        self.tf_broadcaster = TransformBroadcaster()
        
        # Threading
        self.mapping_thread = None
        self.mapping_running = False
        self.map_mutex = threading.Lock()
        
        # Bundle adjustment parameters
        self.ba_window_size = 10  # Number of recent frames for BA
        self.ba_running = False
        
        # Initialize mapping thread
        self.start_mapping_thread()
    
    def start_mapping_thread(self):
        """Start the mapping thread for bundle adjustment."""
        self.mapping_running = True
        self.mapping_thread = threading.Thread(target=self.mapping_loop)
        self.mapping_thread.daemon = True
        self.mapping_thread.start()
        rospy.loginfo("Mapping thread started")
    
    def stop_mapping_thread(self):
        """Stop the mapping thread."""
        self.mapping_running = False
        if self.mapping_thread:
            self.mapping_thread.join()
        rospy.loginfo("Mapping thread stopped")
    
    def process_frame(self, frame, camera_info):
        """
        Process a new frame through the SLAM system.
        
        Args:
            frame: Current frame (OpenCV image)
            camera_info: Camera info message
            
        Returns:
            success: Whether tracking was successful
            pose: Estimated pose matrix (4x4) if successful, None otherwise
        """
        # Extract camera intrinsics
        K = camera_info_to_K(camera_info)
        
        # Detect features
        keypoints = self.detector.detect(frame, None)
        if len(keypoints) < 100:
            return False, None
        
        keypoints, descriptors = compute_descriptors(self.detector, frame, keypoints)
        
        # Update tracking state
        if self.tracking_state == "NO_IMAGES_YET":
            self.tracking_state = "NOT_INITIALIZED"
            # Set initial pose to identity for first keyframe
            self.current_pose = np.eye(4)
            self.reference_keyframe = self.create_keyframe(frame, keypoints, descriptors, K, camera_info.header)
            self.current_frame_id += 1
            return False, None
        
        # Track current frame
        success, pose = self.track_frame(frame, keypoints, descriptors, K, camera_info.header)
        
        if success:
            self.tracking_state = "OK"
            self.current_pose = pose
            
            # Check if we should create a new keyframe
            if self.should_create_keyframe():
                self.create_keyframe(frame, keypoints, descriptors, K, camera_info.header)
        else:
            self.tracking_state = "LOST"
            # Try to reinitialize
            if self.tracking_state == "LOST":
                self.reinitialize(frame, keypoints, descriptors, K, camera_info.header)
        
        self.current_frame_id += 1
        return success, self.current_pose
    
    def track_frame(self, frame, keypoints, descriptors, K, header):
        """
        Track the current frame using PnP.
        
        Args:
            frame: Current frame
            keypoints: Detected keypoints
            descriptors: ORB descriptors
            K: Camera intrinsic matrix
            header: ROS header
            
        Returns:
            success: Whether tracking was successful
            pose: Estimated pose
        """
        if self.reference_keyframe is None:
            return False, None
        
        # Get map points from reference keyframe
        map_points = self.reference_keyframe.get_map_points()
        if len(map_points) < 10:
            return False, None
        
        # Match features with reference keyframe
        ref_descriptors = self.reference_keyframe.get_descriptors()
        if ref_descriptors is None:
            return False, None
        
        matches, good_matches = match_features(ref_descriptors, descriptors, self.detector_type)
        
        if len(good_matches) < 10:
            return False, None
        
        # Prepare 3D-2D correspondences for PnP
        points_3d = []
        points_2d = []
        
        for match in good_matches:
            ref_feature_id = match.queryIdx
            curr_feature_id = match.trainIdx
            
            map_point = self.reference_keyframe.get_map_point(ref_feature_id)
            if map_point is not None and not map_point.is_bad_point():
                point_3d = map_point.get_position()
                point_2d = keypoints[curr_feature_id].pt
                
                points_3d.append(point_3d)
                points_2d.append(point_2d)
        
        if len(points_3d) < 6:
            return False, None
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        # Use RANSAC PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=2.0,
            confidence=0.99,
            iterationsCount=100
        )
        
        if not success or len(inliers) < 10:
            return False, None
        
        # Convert to pose matrix
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()
        
        # Publish pose and transform
        self.publish_pose(pose, header)
        self.publish_path(pose, header)
        
        return True, pose
    
    def create_keyframe(self, frame, keypoints, descriptors, K, header):
        """
        Create a new keyframe and add it to the map.
        
        Args:
            frame: Current frame
            keypoints: Detected keypoints
            descriptors: ORB descriptors
            K: Camera intrinsic matrix
            header: ROS header
            
        Returns:
            Created KeyFrame object
        """
        with self.map_mutex:
            # Create keyframe
            keyframe = KeyFrame(
                frame_id=self.current_frame_id,
                timestamp=header.stamp.to_sec(),
                pose=self.current_pose,
                keypoints=keypoints,
                descriptors=descriptors,
                camera_matrix=K
            )
            
            # Add to map
            keyframe_id = self.map.add_keyframe(keyframe)
            # Update the keyframe's frame_id to the map-assigned ID
            keyframe.frame_id = keyframe_id
            
            # If this is the first keyframe, initialize map points
            if self.reference_keyframe is None:
                self.initialize_map_points(keyframe)
            else:
                # Triangulate new map points
                self.triangulate_new_points(keyframe)
            
            # Update reference keyframe
            self.reference_keyframe = keyframe
            self.map.set_reference_keyframe(keyframe)
            
            self.last_keyframe_id = self.current_frame_id
            rospy.loginfo(f"Created keyframe {keyframe_id}")
            
            return keyframe
    
    def initialize_map_points(self, keyframe):
        """
        Initialize map points from the first keyframe.
        This is a simplified initialization - in practice, you'd need two keyframes.
        
        Args:
            keyframe: First keyframe
        """
        # For now, we'll create dummy map points with reasonable depths
        # In a real implementation, you'd need proper initialization with two frames
        keypoints = keyframe.get_keypoints()
        
        for i, kp in enumerate(keypoints):
            # Create a dummy 3D point in world coordinates
            # Convert from image coordinates to world coordinates
            # Assuming camera is at origin looking along Z-axis
            fx = keyframe.camera_matrix[0, 0]
            fy = keyframe.camera_matrix[1, 1]
            cx = keyframe.camera_matrix[0, 2]
            cy = keyframe.camera_matrix[1, 2]
            
            # Back-project to 3D (assuming depth of 5.0 meters)
            depth = 5.0
            x = (kp.pt[0] - cx) * depth / fx
            y = (kp.pt[1] - cy) * depth / fy
            z = depth
            
            point_3d = np.array([x, y, z])
            
            map_point = MapPoint(point_3d, keyframe.frame_id, i)
            map_point_id = self.map.add_map_point(map_point)
            keyframe.add_map_point(i, map_point)
    
    def triangulate_new_points(self, keyframe):
        """
        Triangulate new map points from the current keyframe.
        
        Args:
            keyframe: Current keyframe
        """
        # This is a simplified triangulation
        # In practice, you'd match features between keyframes and triangulate
        pass
    
    def should_create_keyframe(self):
        """
        Determine if we should create a new keyframe.
        
        Returns:
            True if should create keyframe, False otherwise
        """
        # Simple heuristic: create keyframe every N frames
        return (self.current_frame_id - self.last_keyframe_id) >= self.keyframe_interval
    
    def reinitialize(self, frame, keypoints, descriptors, K, header):
        """
        Reinitialize the system when tracking is lost.
        
        Args:
            frame: Current frame
            keypoints: Detected keypoints
            descriptors: ORB descriptors
            K: Camera intrinsic matrix
            header: ROS header
        """
        rospy.logwarn("Reinitializing SLAM system")
        
        with self.map_mutex:
            # Clear the map
            self.map.clear()
            
            # Reset tracking state
            self.tracking_state = "NOT_INITIALIZED"
            self.current_pose = np.eye(4)
            self.reference_keyframe = None
            
            # Create new reference keyframe
            self.reference_keyframe = self.create_keyframe(frame, keypoints, descriptors, K, header)
    
    def mapping_loop(self):
        """
        Main loop for the mapping thread.
        Performs bundle adjustment on recent keyframes.
        """
        while self.mapping_running:
            try:
                # Get recent keyframes for bundle adjustment
                recent_keyframes = self.map.get_recent_keyframes(self.ba_window_size)
                
                if len(recent_keyframes) >= 2:
                    # Perform bundle adjustment
                    self.perform_bundle_adjustment(recent_keyframes)
                
                # Cull bad map points
                self.map.cull_map_points()
                
                # Sleep for a short time
                time.sleep(0.1)
                
            except Exception as e:
                rospy.logerr(f"Error in mapping thread: {e}")
                time.sleep(0.1)
    
    def perform_bundle_adjustment(self, keyframes):
        """
        Perform bundle adjustment on the given keyframes.
        
        Args:
            keyframes: List of KeyFrame objects to optimize
        """
        if self.ba_running:
            return
        
        self.ba_running = True
        
        try:
            # Get observations for optimization
            keyframe_ids = [kf.frame_id for kf in keyframes]
            observations = self.map.get_observations_for_optimization(keyframe_ids)
            
            if len(observations) < 2:
                return
            
            # Prepare optimization variables
            poses = []
            points_3d = []
            observations_list = []
            
            # Collect poses and 3D points
            for keyframe in keyframes:
                poses.append(keyframe.get_pose())
            
            # Collect 3D points and observations
            point_id_map = {}
            point_counter = 0
            
            for keyframe_id, keyframe_observations in observations.items():
                for feature_id, (map_point, keypoint) in keyframe_observations.items():
                    point_3d = map_point.get_position()
                    
                    # Check if we've seen this point before
                    if map_point not in point_id_map:
                        point_id_map[map_point] = point_counter
                        points_3d.append(point_3d)
                        point_counter += 1
                    
                    observations_list.append({
                        'keyframe_idx': keyframe_ids.index(keyframe_id),
                        'point_idx': point_id_map[map_point],
                        'observation': keypoint.pt
                    })
            
            if len(observations_list) < 10:
                return
            
            # Flatten optimization variables
            poses_flat = np.array(poses).flatten()
            points_3d_flat = np.array(points_3d).flatten()
            
            # Optimization variables: [poses, points_3d]
            x0 = np.concatenate([poses_flat, points_3d_flat])
            
            # Perform optimization
            result = least_squares(
                self.reprojection_error,
                x0,
                args=(observations_list, keyframes, len(poses), len(points_3d)),
                method='lm',
                max_nfev=100
            )
            
            if result.success:
                # Update poses and 3D points
                poses_flat_opt = result.x[:len(poses_flat)]
                points_3d_flat_opt = result.x[len(poses_flat):]
                
                # Reshape and update
                poses_opt = poses_flat_opt.reshape(-1, 4, 4)
                points_3d_opt = points_3d_flat_opt.reshape(-1, 3)
                
                # Update map
                keyframe_poses = {kf.frame_id: pose for kf, pose in zip(keyframes, poses_opt)}
                self.map.update_keyframe_poses(keyframe_poses)
                
                # Update map point positions
                point_positions = {i: pos for i, pos in enumerate(points_3d_opt)}
                self.map.update_map_point_positions(point_positions)
                
                rospy.loginfo(f"Bundle adjustment completed with {len(observations_list)} observations")
        
        except Exception as e:
            rospy.logerr(f"Error in bundle adjustment: {e}")
        
        finally:
            self.ba_running = False
    
    def reprojection_error(self, x, observations, keyframes, num_poses, num_points):
        """
        Compute reprojection error for bundle adjustment.
        
        Args:
            x: Optimization variables [poses, points_3d]
            observations: List of observations
            keyframes: List of KeyFrame objects
            num_poses: Number of poses
            num_points: Number of 3D points
            
        Returns:
            Reprojection errors
        """
        # Extract poses and 3D points
        poses_flat = x[:num_poses * 16]
        points_3d_flat = x[num_poses * 16:]
        
        poses = poses_flat.reshape(num_poses, 4, 4)
        points_3d = points_3d_flat.reshape(num_points, 3)
        
        errors = []
        
        for obs in observations:
            keyframe_idx = obs['keyframe_idx']
            point_idx = obs['point_idx']
            observation = obs['observation']
            
            # Get pose and 3D point
            pose = poses[keyframe_idx]
            point_3d = points_3d[point_idx]
            
            # Project 3D point to 2D
            point_cam = np.linalg.inv(pose) @ np.append(point_3d, 1.0)
            point_cam = point_cam[:3] / point_cam[3]
            
            if point_cam[2] <= 0:  # Point behind camera
                errors.extend([1000, 1000])  # Large error
                continue
            
            # Get camera matrix from keyframe
            keyframe = keyframes[keyframe_idx]
            K = keyframe.camera_matrix
            
            # Project to image coordinates
            point_2d = K @ point_cam
            point_2d = point_2d[:2] / point_2d[2]
            
            # Compute reprojection error
            error = point_2d - np.array(observation)
            errors.extend(error)
        
        return np.array(errors)
    
    def publish_pose(self, pose_matrix, header):
        """
        Publish pose as ROS message.
        
        Args:
            pose_matrix: 4x4 pose matrix
            header: ROS header
        """
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "world"
        
        # Extract position
        pose_msg.pose.position.x = pose_matrix[0, 3]
        pose_msg.pose.position.y = pose_matrix[1, 3]
        pose_msg.pose.position.z = pose_matrix[2, 3]
        
        # Extract orientation
        R = pose_matrix[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        angle = np.linalg.norm(rvec)
        if angle > 1e-6:
            axis = rvec.flatten() / angle
            pose_msg.pose.orientation.x = axis[0] * np.sin(angle/2)
            pose_msg.pose.orientation.y = axis[1] * np.sin(angle/2)
            pose_msg.pose.orientation.z = axis[2] * np.sin(angle/2)
            pose_msg.pose.orientation.w = np.cos(angle/2)
        else:
            pose_msg.pose.orientation.w = 1.0
        
        self.pose_pub.publish(pose_msg)
        
        # Publish transform
        transform = TransformStamped()
        transform.header = header
        transform.header.frame_id = "world"
        transform.child_frame_id = "camera_frame"
        transform.transform.translation.x = pose_msg.pose.position.x
        transform.transform.translation.y = pose_msg.pose.position.y
        transform.transform.translation.z = pose_msg.pose.position.z
        transform.transform.rotation = pose_msg.pose.orientation
        
        self.tf_broadcaster.sendTransform(transform)
    
    def publish_path(self, pose_matrix, header):
        """
        Publish trajectory path.
        
        Args:
            pose_matrix: 4x4 pose matrix
            header: ROS header
        """
        # This would maintain a path history and publish it
        # For now, we'll just publish the current pose
        pass
    
    def shutdown(self):
        """Shutdown the SLAM system."""
        self.stop_mapping_thread()
        rospy.loginfo("SLAM system shutdown complete") 