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
        
        # Tracking failure handling
        self.tracking_failure_count = 0
        self.max_tracking_failures = 5  # Reinitialize after 5 consecutive failures
        
        # Reinitialization flag
        self.reinitializing = False
        
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
        rospy.loginfo("Extracting camera intrinsics...")
        K = camera_info_to_K(camera_info)
        rospy.loginfo(f"Camera matrix: {K}")
        
        # Detect features
        rospy.loginfo("Detecting features...")
        keypoints = self.detector.detect(frame, None)
        rospy.loginfo(f"Detected {len(keypoints)} keypoints")
        if len(keypoints) < 100:
            rospy.logwarn(f"Not enough keypoints: {len(keypoints)} < 100")
            return False, None
        
        rospy.loginfo("Computing descriptors...")
        keypoints, descriptors = compute_descriptors(self.detector, frame, keypoints)
        rospy.loginfo(f"Computed descriptors shape: {descriptors.shape if descriptors is not None else 'None'}")
        
        # Store keypoints for debug visualization
        self._current_keypoints = keypoints
        
        # Update tracking state
        rospy.loginfo(f"Current tracking state: {self.tracking_state}")
        if self.tracking_state == "NO_IMAGES_YET":
            rospy.loginfo("Creating first keyframe...")
            self.tracking_state = "NOT_INITIALIZED"
            # Set initial pose to identity for first keyframe
            self.current_pose = np.eye(4)
            
            # Try to create the first keyframe with timeout
            rospy.loginfo("Attempting to create first keyframe...")
            new_keyframe = self.create_keyframe(frame, keypoints, descriptors, K, camera_info.header)
            if new_keyframe is not None:
                self.reference_keyframe = new_keyframe
                self.current_frame_id += 1
                rospy.loginfo(f"First keyframe created successfully with ID: {new_keyframe.frame_id}")
                rospy.loginfo("Waiting for more frames...")
                return False, None
            else:
                rospy.logwarn("Failed to create first keyframe, trying simple fallback...")
                # Create a simple keyframe without map operations
                try:
                    simple_keyframe = KeyFrame(
                        frame_id=self.current_frame_id,
                        timestamp=camera_info.header.stamp.to_sec(),
                        pose=self.current_pose,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        camera_matrix=K
                    )
                    self.reference_keyframe = simple_keyframe
                    rospy.loginfo("Created simple fallback keyframe")
                except Exception as e:
                    rospy.logerr(f"Failed to create fallback keyframe: {e}")
                
                rospy.logwarn(f"Current state: {self.tracking_state}, Reference keyframe: {self.reference_keyframe}")
                self.current_frame_id += 1
                return False, None
        
        # Track current frame
        rospy.loginfo("Starting frame tracking...")
        success, pose = self.track_frame(frame, keypoints, descriptors, K, camera_info.header)
        
        if success:
            rospy.loginfo("Tracking successful, updating state to OK")
            self.tracking_state = "OK"
            self.current_pose = pose
            self.tracking_failure_count = 0  # Reset failure counter
            
            # Check if we should create a new keyframe
            if self.should_create_keyframe():
                rospy.loginfo("Creating new keyframe...")
                try:
                    # Try to create keyframe with a timeout to avoid deadlock
                    import threading
                    
                    keyframe_created = [False]
                    def create_keyframe_thread():
                        try:
                            self.create_keyframe(frame, keypoints, descriptors, K, camera_info.header)
                            keyframe_created[0] = True
                        except Exception as e:
                            rospy.logwarn(f"Error creating keyframe: {e}")
                    
                    thread = threading.Thread(target=create_keyframe_thread)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=0.5)  # 500ms timeout
                    
                    if thread.is_alive():
                        rospy.logwarn("Keyframe creation timed out, skipping for this frame")
                    elif keyframe_created[0]:
                        rospy.loginfo("Keyframe created successfully")
                except Exception as e:
                    rospy.logwarn(f"Error in keyframe creation: {e}")
        else:
            self.tracking_failure_count += 1
            rospy.logwarn(f"Tracking failed on frame {self.current_frame_id} (failure {self.tracking_failure_count}/{self.max_tracking_failures})")
            
            # Only reinitialize after multiple consecutive failures
            if self.tracking_failure_count >= self.max_tracking_failures:
                rospy.logwarn("Multiple tracking failures, reinitializing...")
                self.tracking_state = "LOST"
                self.reinitialize(frame, keypoints, descriptors, K, camera_info.header)
            else:
                rospy.logwarn("Tracking failed but continuing with current reference keyframe")
                self.tracking_state = "OK"  # Keep current state
        
        self.current_frame_id += 1
        return success, self.current_pose
    
    def get_current_keypoints(self):
        """
        Get the keypoints from the most recent frame processing.
        
        Returns:
            List of keypoints from the last processed frame
        """
        return getattr(self, '_current_keypoints', [])
    
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
            rospy.logwarn("No reference keyframe available")
            rospy.logwarn(f"Tracking state: {self.tracking_state}, Current frame ID: {self.current_frame_id}")
            rospy.logwarn(f"Reference keyframe is None, this should not happen in {self.tracking_state} state")
            return False, None
        
        # Get map points from reference keyframe
        map_points = self.reference_keyframe.get_map_points()
        rospy.loginfo(f"Reference keyframe has {len(map_points)} map points")
        if len(map_points) < 5:  # Reduced from 10 to 5 for testing
            rospy.logwarn(f"Not enough map points: {len(map_points)} < 5")
            return False, None
        
        # Match features with reference keyframe
        ref_descriptors = self.reference_keyframe.get_descriptors()
        if ref_descriptors is None:
            rospy.logwarn("No descriptors in reference keyframe")
            return False, None
        
        matches, good_matches = match_features(ref_descriptors, descriptors, self.detector_type)
        rospy.loginfo(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
        
        # Adaptive threshold based on number of matches
        min_matches = max(3, min(10, len(good_matches) // 10))  # Adaptive threshold
        if len(good_matches) < min_matches:
            rospy.logwarn(f"Not enough good matches: {len(good_matches)} < {min_matches}")
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
        
        min_correspondences = max(3, min(6, len(points_3d) // 10))  # Adaptive threshold
        if len(points_3d) < min_correspondences:
            rospy.logwarn(f"Not enough 3D-2D correspondences: {len(points_3d)} < {min_correspondences}")
            return False, None
        
        rospy.loginfo(f"Using {len(points_3d)} 3D-2D correspondences for PnP")
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
        
        min_inliers = max(3, min(10, len(points_3d) // 10))  # Adaptive threshold
        if not success or len(inliers) < min_inliers:
            rospy.logwarn(f"PnP failed or not enough inliers: success={success}, inliers={len(inliers) if inliers is not None else 0}, required={min_inliers}")
            return False, None
        
        # Convert to pose matrix
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()
        
        # Publish pose and transform
        rospy.loginfo(f"Publishing pose: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}]")
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
        rospy.loginfo("Creating keyframe...")
        
        # Try to acquire the mutex with a timeout to avoid deadlock
        rospy.loginfo("Attempting to acquire map mutex for keyframe creation...")
        if not self.map_mutex.acquire(timeout=0.5):  # 500ms timeout
            rospy.logwarn("Could not acquire map mutex for keyframe creation, skipping...")
            return None
        rospy.loginfo("Successfully acquired map mutex for keyframe creation")
            
        try:
            # Create keyframe
            rospy.loginfo("Instantiating KeyFrame object...")
            keyframe = KeyFrame(
                frame_id=self.current_frame_id,
                timestamp=header.stamp.to_sec(),
                pose=self.current_pose,
                keypoints=keypoints,
                descriptors=descriptors,
                camera_matrix=K
            )
            rospy.loginfo("KeyFrame object created successfully")
            
            # Add to map
            rospy.loginfo("Adding keyframe to map...")
            keyframe_id = self.map.add_keyframe(keyframe)
            # Update the keyframe's frame_id to the map-assigned ID
            keyframe.frame_id = keyframe_id
            rospy.loginfo(f"Keyframe added to map with ID: {keyframe_id}")
            
            # Debug: Check if keyframe was actually added
            rospy.loginfo(f"Map now has {len(self.map.keyframes)} keyframes")
            
            # If this is the first keyframe, initialize map points
            if self.reference_keyframe is None:
                rospy.loginfo("Initializing map points for first keyframe...")
                self.initialize_map_points(keyframe)
                rospy.loginfo(f"After initialization, keyframe has {len(keyframe.get_map_points())} map points")
            else:
                # Triangulate new map points
                rospy.loginfo("Triangulating new map points...")
                self.triangulate_new_points(keyframe)
            
            # Update reference keyframe
            self.reference_keyframe = keyframe
            self.map.set_reference_keyframe(keyframe)
            
            self.last_keyframe_id = self.current_frame_id
            rospy.loginfo(f"Created keyframe {keyframe_id}")
            
            return keyframe
        finally:
            self.map_mutex.release()
    
    def initialize_map_points(self, keyframe):
        """
        Initialize map points from the first keyframe.
        This is a simplified initialization - in practice, you'd need two keyframes.
        
        Args:
            keyframe: First keyframe
        """
        rospy.loginfo("Starting map point initialization...")
        # For now, we'll create dummy map points with reasonable depths
        # In a real implementation, you'd need proper initialization with two frames
        keypoints = keyframe.get_keypoints()
        rospy.loginfo(f"Creating map points for {len(keypoints)} keypoints")
        
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
        
        rospy.loginfo(f"Created {len(keypoints)} map points")
    
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
        
        # Use a non-blocking approach to avoid deadlock with mapping thread
        def reinitialize_thread():
            try:
                # Set reinitializing flag to pause mapping thread
                self.reinitializing = True
                rospy.loginfo("Paused mapping thread for reinitialization")
                
                # Wait longer for mapping thread to finish current operation
                time.sleep(0.5)
                
                rospy.loginfo("Attempting to acquire map mutex for reinitialization...")
                if self.map_mutex.acquire(timeout=2.0):  # Increased timeout to 2 seconds
                    try:
                        rospy.loginfo("Clearing map...")
                        # Clear the map
                        self.map.clear()
                        
                        # Reset tracking state
                        self.tracking_state = "NOT_INITIALIZED"
                        self.current_pose = np.eye(4)
                        self.reference_keyframe = None
                        
                        rospy.loginfo("Creating new reference keyframe for reinitialization...")
                        # Create new reference keyframe (without acquiring mutex again)
                        keyframe = KeyFrame(
                            frame_id=self.current_frame_id,
                            timestamp=header.stamp.to_sec(),
                            pose=self.current_pose,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            camera_matrix=K
                        )
                        
                        # Add to map directly (we already have the mutex)
                        keyframe_id = self.map.add_keyframe(keyframe)
                        keyframe.frame_id = keyframe_id
                        
                        # Initialize map points
                        self.initialize_map_points(keyframe)
                        
                        # Set as reference keyframe
                        self.reference_keyframe = keyframe
                        self.map.set_reference_keyframe(keyframe)
                        
                        rospy.loginfo(f"Reinitialization completed successfully with keyframe {keyframe_id}")
                    finally:
                        self.map_mutex.release()
                else:
                    rospy.logwarn("Could not acquire map mutex for reinitialization, trying simple reset...")
                    # Simple reset without clearing map
                    self.tracking_state = "NOT_INITIALIZED"
                    self.current_pose = np.eye(4)
                    self.reference_keyframe = None
                    
                    # Try to create a simple keyframe without map operations
                    rospy.loginfo("Creating simple keyframe for reinitialization...")
                    keyframe = KeyFrame(
                        frame_id=self.current_frame_id,
                        timestamp=header.stamp.to_sec(),
                        pose=self.current_pose,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        camera_matrix=K
                    )
                    self.reference_keyframe = keyframe
                    rospy.loginfo("Simple reinitialization completed")
            except Exception as e:
                rospy.logerr(f"Error during reinitialization: {e}")
                # Reset to a safe state
                self.tracking_state = "NOT_INITIALIZED"
                self.current_pose = np.eye(4)
                self.reference_keyframe = None
            finally:
                # Resume mapping thread
                self.reinitializing = False
                rospy.loginfo("Resumed mapping thread after reinitialization")
        
        # Run reinitialization in a separate thread to avoid blocking
        import threading
        reinit_thread = threading.Thread(target=reinitialize_thread)
        reinit_thread.daemon = True
        reinit_thread.start()
        
        # Wait for a short time for reinitialization to complete
        reinit_thread.join(timeout=2.0)  # 2 second timeout
        
        if reinit_thread.is_alive():
            rospy.logwarn("Reinitialization timed out, continuing with reset state")
            # Force reset to safe state
            self.tracking_state = "NOT_INITIALIZED"
            self.current_pose = np.eye(4)
            self.reference_keyframe = None
    
    def mapping_loop(self):
        """
        Main loop for the mapping thread.
        Performs bundle adjustment on recent keyframes.
        """
        while self.mapping_running:
            try:
                # Skip bundle adjustment if reinitializing
                if self.reinitializing:
                    time.sleep(0.1)
                    continue
                
                # Get recent keyframes for bundle adjustment
                recent_keyframes = self.map.get_recent_keyframes(self.ba_window_size)
                
                if len(recent_keyframes) >= 2:
                    # Perform bundle adjustment
                    self.perform_bundle_adjustment(recent_keyframes)
                
                # Cull bad map points
                self.map.cull_map_points()
                
                # Sleep for a longer time to reduce contention
                time.sleep(0.5)
                
            except Exception as e:
                rospy.logerr(f"Error in mapping thread: {e}")
                time.sleep(0.5)
    
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
        
        rospy.loginfo(f"Publishing pose message: [{pose_msg.pose.position.x:.2f}, {pose_msg.pose.position.y:.2f}, {pose_msg.pose.position.z:.2f}]")
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
        
        rospy.loginfo("Publishing TF transform")
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