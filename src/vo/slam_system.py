#!/usr/bin/env python3
import numpy as np
import cv2
import rospy
import threading
import time
from scipy.optimize import least_squares
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
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
        # Keyframe selection parameters
        self.keyframe_interval = 10  # Create keyframe every N frames (fallback)
        self.min_translation = 0.1  # Minimum translation distance (meters)
        self.min_rotation = 0.1  # Minimum rotation angle (radians, ~5.7 degrees)
        self.min_tracking_quality = 0.7  # Minimum tracking quality threshold
        self.min_tracked_features = 50  # Minimum number of tracked features
        self.max_reprojection_error = 2.0  # Maximum acceptable reprojection error (pixels)
        
        # Store last keyframe pose for motion comparison
        self.last_keyframe_pose = np.eye(4)
        
        # Tracking state
        self.tracking_state = "NO_IMAGES_YET"  # NO_IMAGES_YET, NOT_INITIALIZED, OK, LOST
        self.current_pose = np.eye(4)
        
        # Publishers
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher("~path", Path, queue_size=10)
        self.debug_pub = rospy.Publisher("~debug_info", String, queue_size=10)
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
        self.max_tracking_failures = 10  # Increased from 5 to 10 for more robustness
        
        # Reinitialization flag
        self.reinitializing = False
        
        # Initialize mapping thread
        # self.start_mapping_thread()  # Commented out to focus on tracking only
    
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
        
        # Store current frame data for potential initialization
        self._current_frame = frame.copy()
        self._current_keypoints = keypoints
        self._current_descriptors = descriptors
        self._current_camera_info = camera_info
        
        # If not initialized, try to initialize with two frames
        if self.tracking_state == "NOT_INITIALIZED":
            rospy.loginfo("System not initialized, checking for sufficient parallax...")
            
            # If we have a reference frame, check parallax with current frame
            if hasattr(self, '_reference_frame') and self._reference_frame is not None:
                rospy.loginfo("Checking parallax with reference frame...")
                
                # Match features between reference and current frame
                matches, good_matches = match_features(
                    self._reference_descriptors, 
                    descriptors, 
                    self.detector_type
                )
                
                if len(good_matches) >= 100:
                    # Extract matched points
                    pts1, pts2 = get_matched_points(self._reference_keypoints, keypoints, good_matches)
                    
                    # Check for sufficient parallax
                    parallax = self.compute_parallax(pts1, pts2, K)
                    rospy.loginfo(f"Parallax with reference frame: {parallax:.3f} degrees")
                    
                    if parallax >= 1.0:  # Sufficient parallax threshold
                        rospy.loginfo("Sufficient parallax detected, attempting initialization...")
                        
                        # Try to initialize with reference and current frame
                        success = self.initialize_system(
                            self._reference_frame, 
                            self._reference_keypoints, 
                            self._reference_descriptors,
                            frame, 
                            keypoints, 
                            descriptors, 
                            K
                        )
                        
                        if success:
                            rospy.loginfo("Two-frame initialization successful!")
                            self.current_frame_id += 1
                            return True, self.current_pose
                        else:
                            rospy.logwarn("Initialization failed despite sufficient parallax, continuing...")
                    else:
                        rospy.logwarn(f"Insufficient parallax ({parallax:.3f} < 1.0), waiting for more motion...")
                        # Keep the same reference frame, don't update it
                        self.current_frame_id += 1
                        return False, None
                else:
                    rospy.logwarn(f"Insufficient matches ({len(good_matches)} < 100), updating reference frame...")
                    # Only update reference frame when we have insufficient matches
                    self._reference_frame = frame.copy()
                    self._reference_keypoints = keypoints
                    self._reference_descriptors = descriptors
                    self._reference_camera_info = camera_info
            
            # If no reference frame exists yet, store current frame as reference
            else:
                rospy.loginfo("No reference frame exists, storing current frame as reference...")
                self._reference_frame = frame.copy()
                self._reference_keypoints = keypoints
                self._reference_descriptors = descriptors
                self._reference_camera_info = camera_info
            
            self.current_frame_id += 1
            return False, None
        elif self.tracking_state == "NO_IMAGES_YET":
            # First frame - store it as reference frame
            rospy.loginfo("First frame received, storing as reference frame...")
            self.tracking_state = "NOT_INITIALIZED"
            self._reference_frame = frame.copy()
            self._reference_keypoints = keypoints
            self._reference_descriptors = descriptors
            self._reference_camera_info = camera_info
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
            
            # Publish debug information
            debug_msg = f"Frame {self.current_frame_id}: No reference keyframe, State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Get all map points from the system
        all_map_points = self.map.get_map_points()
        rospy.loginfo(f"System has {len(all_map_points)} total map points")
        
        # Get descriptors from reference keyframe for matching
        ref_descriptors = self.reference_keyframe.get_descriptors()
        if ref_descriptors is None:
            rospy.logwarn("No descriptors in reference keyframe")
            return False, None
        
        # Match features between reference keyframe and current frame
        matches, good_matches = match_features(ref_descriptors, descriptors, self.detector_type)
        rospy.loginfo(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
        
        # Publish debug information about matches
        debug_msg = f"Frame {self.current_frame_id}: {len(good_matches)}/{len(matches)} matches, State: {self.tracking_state}"
        self.debug_pub.publish(debug_msg)
        
        # Adaptive threshold based on number of matches
        min_matches = max(5, min(15, len(good_matches) // 5))  # Ensure at least 5, but be more lenient
        if len(good_matches) < min_matches:
            rospy.logwarn(f"Not enough good matches: {len(good_matches)} < {min_matches}")
            debug_msg = f"Frame {self.current_frame_id}: Insufficient matches ({len(good_matches)}/{min_matches}), State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Prepare 3D-2D correspondences for PnP
        points_3d = []
        points_2d = []
        bad_points_count = 0
        no_map_point_count = 0
        
        for match in good_matches:
            ref_feature_id = match.queryIdx
            curr_feature_id = match.trainIdx
            
            # Try to get map point from reference keyframe first
            map_point = self.reference_keyframe.get_map_point(ref_feature_id)
            if map_point is not None and not map_point.is_bad_point():
                point_3d = map_point.get_position()
                point_2d = keypoints[curr_feature_id].pt
                
                points_3d.append(point_3d)
                points_2d.append(point_2d)
            elif map_point is not None:
                bad_points_count += 1
            else:
                no_map_point_count += 1
        
        rospy.loginfo(f"3D-2D correspondences: {len(points_3d)} valid, {bad_points_count} bad map points, {no_map_point_count} no map point")
        
        # More detailed debugging
        if len(points_3d) < 10:  # If we have very few correspondences, debug in detail
            rospy.logwarn(f"DEBUG: Only {len(points_3d)} valid 3D-2D correspondences from {len(good_matches)} good matches")
            rospy.logwarn(f"DEBUG: Bad map points: {bad_points_count}, No map point: {no_map_point_count}")
            
            # Check a few specific matches to understand the issue
            for i, match in enumerate(good_matches[:5]):  # Check first 5 matches
                ref_feature_id = match.queryIdx
                map_point = self.reference_keyframe.get_map_point(ref_feature_id)
                if map_point is not None:
                    rospy.logwarn(f"DEBUG: Match {i}: Feature {ref_feature_id} has map point, bad={map_point.is_bad_point()}")
                else:
                    rospy.logwarn(f"DEBUG: Match {i}: Feature {ref_feature_id} has NO map point")
        else:
            # Even when we have enough correspondences, check why some matches don't have map points
            if no_map_point_count > 0:
                rospy.logwarn(f"DEBUG: {no_map_point_count} matches have no map points out of {len(good_matches)} total matches")
                # Check a few random matches that don't have map points
                no_map_point_features = []
                for match in good_matches:
                    ref_feature_id = match.queryIdx
                    if not self.reference_keyframe.has_map_point(ref_feature_id):
                        no_map_point_features.append(ref_feature_id)
                        if len(no_map_point_features) >= 5:  # Show first 5
                            break
                rospy.logwarn(f"DEBUG: Features without map points: {no_map_point_features}")
        
        min_correspondences = max(4, min(10, len(points_3d) // 5))  # Ensure at least 4, but be more lenient
        if len(points_3d) < min_correspondences:
            rospy.logwarn(f"Not enough 3D-2D correspondences: {len(points_3d)} < {min_correspondences}")
            debug_msg = f"Frame {self.current_frame_id}: Insufficient 3D-2D correspondences ({len(points_3d)}/{min_correspondences}), State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        rospy.loginfo(f"Using {len(points_3d)} 3D-2D correspondences for PnP")
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        # Ensure we have at least 4 points for PnP
        if len(points_3d) < 4:
            rospy.logwarn(f"Not enough 3D-2D correspondences for PnP: {len(points_3d)} < 4")
            debug_msg = f"Frame {self.current_frame_id}: Insufficient 3D-2D correspondences for PnP ({len(points_3d)} < 4), State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Use RANSAC PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=2.0,
            confidence=0.99,
            iterationsCount=100
        )

        min_inliers = max(3, min(8, len(points_3d) // 10))  # More lenient: max 8 instead of 10, min 3
        if not success or len(inliers) < min_inliers:
            rospy.logwarn(f"PnP failed or not enough inliers: success={success}, inliers={len(inliers) if inliers is not None else 0}, required={min_inliers}")
            debug_msg = f"Frame {self.current_frame_id}: PnP failed (success={success}, inliers={len(inliers) if inliers is not None else 0}/{min_inliers}), State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        else:
            rospy.loginfo(f"PnP success: {success}, inliers: {len(inliers)}")
        # Convert to pose matrix
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()
        
        # Publish pose and transform
        rospy.loginfo(f"Publishing pose: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}]")
        self.publish_pose(pose, header)
        self.publish_path(pose, header)
        
        # Publish success debug information
        debug_msg = f"Frame {self.current_frame_id}: Tracking SUCCESS! Pose: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}], State: {self.tracking_state}"
        self.debug_pub.publish(debug_msg)
        
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
            
            # Triangulate new map points (system is already initialized)
            rospy.loginfo("Triangulating new map points...")
            self.triangulate_new_points(keyframe)
            
            # Update reference keyframe
            self.reference_keyframe = keyframe
            self.map.set_reference_keyframe(keyframe)
            
            # Store current keyframe as old reference for next time
            self._old_reference_keyframe = keyframe
            
            # Store current pose as last keyframe pose for motion comparison
            self.last_keyframe_pose = self.current_pose.copy()
            
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
            
            # Debug: Check if map point was properly added
            # if i < 5:  # Debug first 5 map points
            #     rospy.loginfo(f"DEBUG: Created map point {map_point_id} for feature {i}, bad={map_point.is_bad_point()}")
            #     rospy.loginfo(f"DEBUG: Keyframe has map point for feature {i}: {keyframe.has_map_point(i)}")
        
        rospy.loginfo(f"Created {len(keypoints)} map points")
    
    def initialize_system(self, frame1, keypoints1, descriptors1, frame2, keypoints2, descriptors2, K):
        """
        Initialize the SLAM system using two frames with sufficient parallax.
        Uses ORB-SLAM2 style two-model approach (Homography + Fundamental Matrix).
        
        Args:
            frame1: First frame
            keypoints1: Keypoints from first frame
            descriptors1: Descriptors from first frame
            frame2: Second frame
            keypoints2: Keypoints from second frame
            descriptors2: Descriptors from second frame
            K: Camera intrinsic matrix
            
        Returns:
            success: Whether initialization was successful
        """
        rospy.loginfo("Starting two-frame initialization...")
        
        # 1. Feature Extraction and Matching
        matches, good_matches = match_features(descriptors1, descriptors2, self.detector_type)
        rospy.loginfo(f"Found {len(good_matches)} good matches for initialization")
        
        if len(good_matches) < 100:
            rospy.logwarn(f"Not enough matches for initialization: {len(good_matches)} < 100")
            return False
        
        # 2. Extract matched points
        pts1, pts2 = get_matched_points(keypoints1, keypoints2, good_matches)
        
        # 3. Check for sufficient parallax
        parallax = self.compute_parallax(pts1, pts2, K)
        rospy.loginfo(f"Parallax between frames: {parallax:.3f}")
        
        if parallax < 1.0:  # Minimum parallax threshold (1 degree)
            rospy.logwarn(f"Insufficient parallax: {parallax:.3f} < 1.0 degrees")
            return False
        
        # 4. Parallel Model Computation
        rospy.loginfo("Computing Homography and Fundamental matrices...")
        
        # Compute Homography Matrix
        H, H_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        H_inliers = np.sum(H_mask) if H_mask is not None else 0
        
        # Compute Fundamental Matrix
        F, F_mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 3.0, 0.99)
        F_inliers = np.sum(F_mask) if F_mask is not None else 0
        
        rospy.loginfo(f"Homography inliers: {H_inliers}/{len(good_matches)}")
        rospy.loginfo(f"Fundamental Matrix inliers: {F_inliers}/{len(good_matches)}")
        
        if H is None or F is None:
            rospy.logwarn("Failed to compute geometric models")
            return False
        
        # 5. Model Selection
        R_H = self.compute_model_selection_score(H, F, pts1, pts2, H_mask, F_mask)
        rospy.loginfo(f"Model selection score R_H: {R_H:.3f}")
        
        if R_H > 0.40:
            rospy.loginfo("Scene is non-planar, using Fundamental Matrix")
            use_homography = False
        else:
            rospy.loginfo("Scene is planar or low parallax, using Homography")
            use_homography = True
        
        # 6. Create Initial Map
        if use_homography:
            success = self.initialize_with_homography(H, pts1, pts2, K, keypoints1, keypoints2, descriptors1, descriptors2, good_matches, H_mask)
        else:
            success = self.initialize_with_fundamental(F, pts1, pts2, K, keypoints1, keypoints2, descriptors1, descriptors2, good_matches, F_mask)
        
        if success:
            rospy.loginfo("Initialization successful!")
            return True
        else:
            rospy.logwarn("Initialization failed")
            return False
    
    def compute_parallax(self, pts1, pts2, K):
        """
        Compute the parallax between two sets of matched points.
        
        Args:
            pts1: Points in first frame (Nx1x2)
            pts2: Points in second frame (Nx1x2)
            K: Camera intrinsics matrix (3x3)
            
        Returns:
            parallax: Average parallax in degrees
        """
        # Convert to normalized coordinates
        pts1_norm = pts1.reshape(-1, 2)
        pts2_norm = pts2.reshape(-1, 2)
        
        # Compute displacement vectors
        displacements = pts2_norm - pts1_norm
        distances = np.linalg.norm(displacements, axis=1)
        
        # Get focal length from camera intrinsics
        focal_length = (K[0, 0] + K[1, 1]) / 2.0  # Average of fx and fy
        rospy.loginfo(f"Using focal length: {focal_length:.2f} pixels (fx={K[0,0]:.2f}, fy={K[1,1]:.2f})")
        
        # Convert to angular parallax
        parallax_angles = np.arctan2(distances, focal_length) * 180 / np.pi
        
        # Return average parallax
        return np.mean(parallax_angles)
    
    def compute_model_selection_score(self, H, F, pts1, pts2, H_mask, F_mask):
        """
        Compute the model selection score R_H for choosing between Homography and Fundamental Matrix.
        
        Args:
            H: Homography matrix
            F: Fundamental matrix
            pts1, pts2: Matched points
            H_mask, F_mask: Inlier masks
            
        Returns:
            R_H: Model selection score
        """
        # Compute symmetric transfer error for Homography
        H_error = self.compute_homography_error(H, pts1, pts2, H_mask)
        
        # Compute symmetric transfer error for Fundamental Matrix
        F_error = self.compute_fundamental_error(F, pts1, pts2, F_mask)
        
        # Compute score
        if F_error > 0:
            R_H = H_error / F_error
        else:
            R_H = float('inf')
        
        return R_H
    
    def compute_homography_error(self, H, pts1, pts2, mask):
        """
        Compute symmetric transfer error for Homography.
        """
        if H is None or mask is None:
            return float('inf')
        
        pts1_norm = pts1.reshape(-1, 2)
        pts2_norm = pts2.reshape(-1, 2)
        
        # Forward transfer error
        pts1_homo = np.hstack([pts1_norm, np.ones((pts1_norm.shape[0], 1))])
        pts2_pred = (H @ pts1_homo.T).T
        pts2_pred = pts2_pred[:, :2] / pts2_pred[:, 2:]
        forward_error = np.linalg.norm(pts2_norm - pts2_pred, axis=1)
        
        # Backward transfer error
        H_inv = np.linalg.inv(H)
        pts2_homo = np.hstack([pts2_norm, np.ones((pts2_norm.shape[0], 1))])
        pts1_pred = (H_inv @ pts2_homo.T).T
        pts1_pred = pts1_pred[:, :2] / pts1_pred[:, 2:]
        backward_error = np.linalg.norm(pts1_norm - pts1_pred, axis=1)
        
        # Symmetric error
        symmetric_error = forward_error + backward_error
        
        # Return mean error for inliers
        inlier_errors = symmetric_error[mask.flatten() == 1]
        return np.mean(inlier_errors) if len(inlier_errors) > 0 else float('inf')
    
    def compute_fundamental_error(self, F, pts1, pts2, mask):
        """
        Compute symmetric epipolar error for Fundamental Matrix.
        """
        if F is None or mask is None:
            return float('inf')
        
        pts1_norm = pts1.reshape(-1, 2)
        pts2_norm = pts2.reshape(-1, 2)
        
        # Convert to homogeneous coordinates
        pts1_homo = np.hstack([pts1_norm, np.ones((pts1_norm.shape[0], 1))])
        pts2_homo = np.hstack([pts2_norm, np.ones((pts2_norm.shape[0], 1))])
        
        # Compute epipolar errors
        lines1 = (F.T @ pts2_homo.T).T  # Epipolar lines in image 1
        lines2 = (F @ pts1_homo.T).T    # Epipolar lines in image 2
        
        # Distance from points to epipolar lines
        error1 = np.abs(np.sum(pts1_homo * lines1, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
        error2 = np.abs(np.sum(pts2_homo * lines2, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
        
        # Symmetric error
        symmetric_error = error1 + error2
        
        # Return mean error for inliers
        inlier_errors = symmetric_error[mask.flatten() == 1]
        return np.mean(inlier_errors) if len(inlier_errors) > 0 else float('inf')
    
    def initialize_with_homography(self, H, pts1, pts2, K, keypoints1, keypoints2, descriptors1, descriptors2, matches, mask):
        """
        Initialize the system using the Homography matrix.
        """
        rospy.loginfo("Initializing with Homography...")
        
        # 1. Estimate pose from Homography
        retval, R, t = cv2.recoverPose(H, pts1, pts2, K)
        
        if retval:
            rospy.loginfo(f"Pose recovered from Homography: R={R}, t={t}")
            
            # 2. Triangulate 3D points
            points_3d = self.triangulate_points(pts1, pts2, np.eye(4), np.hstack([R, t]), K)
            
            if points_3d is None or len(points_3d) == 0:
                rospy.logwarn("Triangulation failed with Homography")
                return False
            
            # 3. Validate and create map points
            # (Validation now happens inline with mask filtering)
            
            # 4. Create keyframes
            keyframe1 = KeyFrame(
                frame_id=self.current_frame_id - 1,
                timestamp=0.0,
                pose=np.eye(4), # First camera at origin
                keypoints=keypoints1,
                descriptors=descriptors1,
                camera_matrix=K
            )
            
            keyframe2 = KeyFrame(
                frame_id=self.current_frame_id,
                timestamp=0.0,
                pose=np.hstack([R, t]),
                keypoints=keypoints2,
                descriptors=descriptors2,
                camera_matrix=K
            )
            
            # 5. Add keyframes to map
            keyframe1_id = self.map.add_keyframe(keyframe1)
            keyframe2_id = self.map.add_keyframe(keyframe2)
            keyframe1.frame_id = keyframe1_id
            keyframe2.frame_id = keyframe2_id
            
            # 6. Create map points from triangulated 3D points
            rospy.loginfo("Creating map points from triangulated 3D points (Homography)...")
            
            # Filter matches and points using the same mask
            inlier_matches = []
            inlier_points_3d = []
            
            for i, (match, point_3d) in enumerate(zip(matches, points_3d)):
                if mask[i, 0]:  # This point is an inlier
                    # Check if the triangulated point is valid
                    if point_3d[2] > 0.1 and point_3d[2] < 100.0:  # Depth between 0.1m and 100m
                        inlier_matches.append(match)
                        inlier_points_3d.append(point_3d)
            
            rospy.loginfo(f"Valid inlier points for map point creation (Homography): {len(inlier_points_3d)}")
            
            if len(inlier_points_3d) < 50:
                rospy.logwarn(f"Not enough valid inlier points from Homography: {len(inlier_points_3d)} < 50")
                return False
            
            # Create map points for each valid inlier point
            for i, (match, point_3d) in enumerate(zip(inlier_matches, inlier_points_3d)):
                # Create map point
                map_point = MapPoint(point_3d, keyframe2.frame_id, match.trainIdx)
                map_point_id = self.map.add_map_point(map_point)
                
                # Add observation to second keyframe
                keyframe2.add_map_point(match.trainIdx, map_point)
                
                # Add observation to first keyframe
                keyframe1.add_map_point(match.queryIdx, map_point)
                
                if i < 5:  # Debug first 5 map points
                    rospy.loginfo(f"DEBUG: Created map point {map_point_id} at [{point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}] (Homography)")
            
            # 7. Set reference keyframe and update system state
            self.reference_keyframe = keyframe2
            self.map.set_reference_keyframe(keyframe2)
            self.current_pose = keyframe2.get_pose()
            self.last_keyframe_pose = keyframe2.get_pose().copy()  # Set initial last keyframe pose
            self.tracking_state = "OK"
            
            rospy.loginfo(f"Initialization successful with Homography! Created {len(inlier_points_3d)} map points")
            rospy.loginfo(f"Reference keyframe has {len(keyframe2.get_map_points())} map points")
            
            return True
        else:
            rospy.logwarn("Pose recovery from Homography failed")
            return False
    
    def initialize_with_fundamental(self, F, pts1, pts2, K, keypoints1, keypoints2, descriptors1, descriptors2, matches, mask):
        """
        Initialize the system using the Fundamental matrix.
        """
        rospy.loginfo("Initializing with Fundamental Matrix...")
        
        # 1. Estimate pose from Fundamental Matrix
        retval, R, t, mask_ = cv2.recoverPose(F, pts1, pts2, K)
        
        if retval:
            rospy.loginfo(f"Pose recovered from Fundamental Matrix: R={R}, t={t}")
            
            # 2. Triangulate 3D points
            points_3d = self.triangulate_points(pts1, pts2, np.eye(4), np.hstack([R, t]), K)
            
            if points_3d is None or len(points_3d) == 0:
                rospy.logwarn("Triangulation failed with Fundamental Matrix")
                return False
            
            # 3. Validate and create map points
            # (Validation now happens inline with mask filtering)
            
            # 4. Create keyframes
            keyframe1 = KeyFrame(
                frame_id=self.current_frame_id - 1,
                timestamp=0.0,
                pose=np.eye(4), # First camera at origin
                keypoints=keypoints1,
                descriptors=descriptors1,
                camera_matrix=K
            )
            
            keyframe2 = KeyFrame(
                frame_id=self.current_frame_id,
                timestamp=0.0,
                pose=np.hstack([R, t]),
                keypoints=keypoints2,
                descriptors=descriptors2,
                camera_matrix=K
            )
            
            # 5. Add keyframes to map
            keyframe1_id = self.map.add_keyframe(keyframe1)
            keyframe2_id = self.map.add_keyframe(keyframe2)
            keyframe1.frame_id = keyframe1_id
            keyframe2.frame_id = keyframe2_id
            
            # 6. Create map points from triangulated 3D points
            rospy.loginfo("Creating map points from triangulated 3D points (Fundamental)...")
            
            # Filter matches and points using the same mask
            inlier_matches = []
            inlier_points_3d = []
            
            for i, (match, point_3d) in enumerate(zip(matches, points_3d)):
                if mask[i, 0]:  # This point is an inlier
                    # Check if the triangulated point is valid
                    if point_3d[2] > 0.1 and point_3d[2] < 100.0:  # Depth between 0.1m and 100m
                        inlier_matches.append(match)
                        inlier_points_3d.append(point_3d)
            
            rospy.loginfo(f"Valid inlier points for map point creation (Fundamental): {len(inlier_points_3d)}")
            
            if len(inlier_points_3d) < 50:
                rospy.logwarn(f"Not enough valid inlier points from Fundamental Matrix: {len(inlier_points_3d)} < 50")
                return False
            
            # Create map points for each valid inlier point
            for i, (match, point_3d) in enumerate(zip(inlier_matches, inlier_points_3d)):
                # Create map point
                map_point = MapPoint(point_3d, keyframe2.frame_id, match.trainIdx)
                map_point_id = self.map.add_map_point(map_point)
                
                # Add observation to second keyframe
                keyframe2.add_map_point(match.trainIdx, map_point)
                
                # Add observation to first keyframe
                keyframe1.add_map_point(match.queryIdx, map_point)
                
                # if i < 5:  # Debug first 5 map points
                #     rospy.loginfo(f"DEBUG: Created map point {map_point_id} at [{point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}] (Fundamental)")
            
            # 7. Set reference keyframe and update system state
            self.reference_keyframe = keyframe2
            self.map.set_reference_keyframe(keyframe2)
            self.current_pose = keyframe2.get_pose()
            self.last_keyframe_pose = keyframe2.get_pose().copy()  # Set initial last keyframe pose
            self.tracking_state = "OK"
            
            rospy.loginfo(f"Initialization successful with Fundamental Matrix! Created {len(inlier_points_3d)} map points")
            rospy.loginfo(f"Reference keyframe has {len(keyframe2.get_map_points())} map points")
            
            return True
        else:
            rospy.logwarn("Pose recovery from Fundamental Matrix failed")
            return False
    
    def triangulate_new_points(self, keyframe):
        """
        Triangulate new map points from the current keyframe.
        
        Args:
            keyframe: Current keyframe
        """
        if self.reference_keyframe is None:
            rospy.logwarn("No reference keyframe for triangulation")
            return
        
        rospy.loginfo("Starting triangulation of new map points...")
        
        # Get keypoints and descriptors from both keyframes
        ref_keypoints = self.reference_keyframe.get_keypoints()
        ref_descriptors = self.reference_keyframe.get_descriptors()
        curr_keypoints = keyframe.get_keypoints()
        curr_descriptors = keyframe.get_descriptors()
        
        if ref_descriptors is None or curr_descriptors is None:
            rospy.logwarn("Missing descriptors for triangulation")
            return
        
        # Match features between reference and current keyframe
        matches, good_matches = match_features(ref_descriptors, curr_descriptors, self.detector_type)
        rospy.loginfo(f"Found {len(good_matches)} good matches for triangulation")
        
        if len(good_matches) < 10:
            rospy.logwarn(f"Not enough matches for triangulation: {len(good_matches)} < 10")
            return
        
        # Get camera poses
        ref_pose = self.reference_keyframe.get_pose()
        curr_pose = keyframe.get_pose()
        
        # Extract matched points
        ref_points = []
        curr_points = []
        
        for match in good_matches:
            ref_idx = match.queryIdx
            curr_idx = match.trainIdx
            
            ref_points.append(ref_keypoints[ref_idx].pt)
            curr_points.append(curr_keypoints[curr_idx].pt)
        
        ref_points = np.array(ref_points, dtype=np.float32)
        curr_points = np.array(curr_points, dtype=np.float32)
        
        # Triangulate 3D points using the two camera poses
        points_3d = self.triangulate_points(ref_points, curr_points, ref_pose, curr_pose, keyframe.camera_matrix)
        
        if points_3d is None or len(points_3d) == 0:
            rospy.logwarn("Triangulation failed")
            return
        
        # Create map points for the triangulated 3D points
        new_map_points = 0
        for i, (match, point_3d) in enumerate(zip(good_matches, points_3d)):
            # Check if point is valid (positive depth, reasonable distance)
            if point_3d[2] > 0.1 and point_3d[2] < 100.0:  # Depth between 0.1m and 100m
                # Create map point
                map_point = MapPoint(point_3d, keyframe.frame_id, match.trainIdx)
                map_point_id = self.map.add_map_point(map_point)
                
                # Add observation to keyframe
                keyframe.add_map_point(match.trainIdx, map_point)
                
                # Add observation to reference keyframe if it doesn't have this point
                ref_feature_id = match.queryIdx
                if not self.reference_keyframe.has_map_point(ref_feature_id):
                    self.reference_keyframe.add_map_point(ref_feature_id, map_point)
                    # rospy.loginfo(f"DEBUG: Added map point {map_point_id} to reference keyframe feature {ref_feature_id}")
                # else:
                #     rospy.loginfo(f"DEBUG: Reference keyframe already has map point for feature {ref_feature_id}")
                
                new_map_points += 1
            else:
                rospy.logwarn(f"DEBUG: Invalid 3D point {i}: depth={point_3d[2]:.2f}")
        
        rospy.loginfo(f"Created {new_map_points} new map points from triangulation")
        
        # Publish debug information about new map points
        debug_msg = f"Frame {self.current_frame_id}: Created {new_map_points} new map points, Total map points: {len(self.map.map_points)}"
        self.debug_pub.publish(debug_msg)
    
    def triangulate_points(self, points1, points2, pose1, pose2, K):
        """
        Triangulate 3D points from two camera poses and matched 2D points.
        
        Args:
            points1: 2D points in first camera frame (Nx2)
            points2: 2D points in second camera frame (Nx2)
            pose1: Camera pose 1 (4x4)
            pose2: Camera pose 2 (4x4)
            K: Camera intrinsic matrix (3x3)
            
        Returns:
            points_3d: Triangulated 3D points (Nx3)
        """
        if len(points1) != len(points2):
            rospy.logwarn("Number of points don't match for triangulation")
            return None
        
        if len(points1) == 0:
            return None
        
        # Convert poses to camera matrices
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]
        
        # Create projection matrices
        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
        
        # Normalize points
        points1_norm = cv2.undistortPoints(points1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        points2_norm = cv2.undistortPoints(points2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, points1_norm.T, points2_norm.T)
        
        # Convert to 3D points (homogeneous to 3D)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T
        
        return points_3d
    
    def should_create_keyframe(self):
        """
        Determine if we should create a new keyframe based on motion and tracking quality.
        
        Returns:
            True if should create keyframe, False otherwise
        """
        # If no reference keyframe exists, we can't make a decision
        if self.reference_keyframe is None:
            return False
        
        # 1. Check motion-based criteria
        motion_criteria = self.check_motion_criteria()
        
        # 2. Check tracking quality criteria
        tracking_criteria = self.check_tracking_quality_criteria()
        
        # 3. Fallback to fixed interval if no other criteria met
        interval_criteria = (self.current_frame_id - self.last_keyframe_id) >= self.keyframe_interval
        
        # Create keyframe if any criteria are met
        should_create = motion_criteria or tracking_criteria or interval_criteria
        
        if should_create:
            rospy.loginfo(f"Keyframe selection: motion={motion_criteria}, tracking={tracking_criteria}, interval={interval_criteria}")
        
        return should_create
    
    def check_motion_criteria(self):
        """
        Check if motion-based criteria are met for keyframe creation.
        
        Returns:
            True if motion criteria are met, False otherwise
        """
        if self.reference_keyframe is None:
            return False
        
        # Get current pose and last keyframe pose
        current_pose = self.current_pose
        last_keyframe_pose = self.last_keyframe_pose
        
        # Validate pose matrices
        if (current_pose is None or last_keyframe_pose is None or
            current_pose.shape != (4, 4) or last_keyframe_pose.shape != (4, 4)):
            rospy.logwarn("Invalid pose matrices for motion criteria check")
            return False
        
        # Check if matrices are valid (not all zeros, proper structure)
        if (np.allclose(current_pose, 0) or np.allclose(last_keyframe_pose, 0) or
            np.allclose(current_pose, np.eye(4)) and np.allclose(last_keyframe_pose, np.eye(4))):
            rospy.logwarn("Pose matrices are invalid (all zeros or identity) for motion check")
            return False
        
        try:
            # Compute relative transformation
            relative_pose = np.linalg.inv(last_keyframe_pose) @ current_pose
            
            # Extract translation and rotation
            translation = relative_pose[:3, 3]
            translation_distance = np.linalg.norm(translation)
            
            # Extract rotation matrix and convert to angle
            rotation_matrix = relative_pose[:3, :3]
            rotation_angle = self.rotation_matrix_to_angle(rotation_matrix)
            
            # Check thresholds
            translation_ok = translation_distance > self.min_translation
            rotation_ok = rotation_angle > self.min_rotation
            
            if translation_ok or rotation_ok:
                rospy.loginfo(f"Motion criteria met: translation={translation_distance:.3f}m, rotation={rotation_angle:.3f}rad")
            
            return translation_ok or rotation_ok
            
        except np.linalg.LinAlgError as e:
            rospy.logwarn(f"Linear algebra error in motion criteria check: {e}")
            return False
        except Exception as e:
            rospy.logwarn(f"Error in motion criteria check: {e}")
            return False
    
    def check_tracking_quality_criteria(self):
        """
        Check if tracking quality criteria are met for keyframe creation.
        
        Returns:
            True if tracking quality criteria are met, False otherwise
        """
        if self.reference_keyframe is None:
            return False
        
        try:
            # Get tracking quality metrics
            tracked_features = self.get_tracked_feature_count()
            tracking_quality = self.compute_tracking_quality()
            
            # Check thresholds
            features_ok = tracked_features < self.min_tracked_features
            quality_ok = tracking_quality < self.min_tracking_quality
            
            if features_ok or quality_ok:
                rospy.loginfo(f"Tracking quality criteria met: features={tracked_features}, quality={tracking_quality:.3f}")
            
            return features_ok or quality_ok
            
        except Exception as e:
            rospy.logwarn(f"Error in tracking quality criteria check: {e}")
            return False
    
    def rotation_matrix_to_angle(self, R):
        """
        Convert rotation matrix to rotation angle.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            angle: Rotation angle in radians
        """
        # Use the trace formula: trace(R) = 1 + 2*cos(theta)
        trace = np.trace(R)
        cos_angle = (trace - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)  # Clamp to valid range
        angle = np.arccos(cos_angle)
        return angle
    
    def get_tracked_feature_count(self):
        """
        Get the number of currently tracked features.
        
        Returns:
            count: Number of tracked features
        """
        if self.reference_keyframe is None:
            return 0
        
        try:
            # Count map points that are currently visible
            map_points = self.map.get_map_points()
            tracked_count = 0
            
            for map_point in map_points.values():
                if not map_point.is_bad_point():
                    # Check if this map point is observed by the reference keyframe
                    if map_point.get_observations_count() > 0:
                        tracked_count += 1
            
            return tracked_count
            
        except Exception as e:
            rospy.logwarn(f"Error getting tracked feature count: {e}")
            return 0
    
    def compute_tracking_quality(self):
        """
        Compute tracking quality based on various metrics.
        
        Returns:
            quality: Tracking quality score (0-1, higher is better)
        """
        if self.reference_keyframe is None:
            return 0.0
        
        try:
            # Get current tracking metrics
            tracked_features = self.get_tracked_feature_count()
            total_map_points = len(self.map.get_map_points())
            
            # Compute feature visibility ratio
            if total_map_points > 0:
                visibility_ratio = tracked_features / total_map_points
            else:
                visibility_ratio = 0.0
            
            # Compute feature density (normalized by image area)
            # For simplicity, assume 640x480 image
            image_area = 640 * 480
            feature_density = tracked_features / image_area * 10000  # Scale for readability
            
            # Combine metrics into a quality score
            quality = (visibility_ratio * 0.6 + 
                      min(feature_density / 100.0, 1.0) * 0.4)
            
            return quality
            
        except Exception as e:
            rospy.logwarn(f"Error computing tracking quality: {e}")
            return 0.0
    
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
                        self.last_keyframe_pose = np.eye(4)  # Reset last keyframe pose
                        self.reference_keyframe = None
                        
                        # Clear previous frame data for fresh initialization
                        if hasattr(self, '_reference_frame'):
                            delattr(self, '_reference_frame')
                        if hasattr(self, '_reference_keypoints'):
                            delattr(self, '_reference_keypoints')
                        if hasattr(self, '_reference_descriptors'):
                            delattr(self, '_reference_descriptors')
                        if hasattr(self, '_reference_camera_info'):
                            delattr(self, '_reference_camera_info')
                        
                        rospy.loginfo("System reset for two-frame reinitialization")
                        
                        # Release mutex and let the system reinitialize naturally
                        rospy.loginfo("Reinitialization complete, system will reinitialize with next frame with sufficient parallax")
                        
                        return
                    finally:
                        self.map_mutex.release()
                else:
                    rospy.logwarn("Could not acquire map mutex for reinitialization, trying simple reset...")
                    # Simple reset without clearing map
                    self.tracking_state = "NOT_INITIALIZED"
                    self.current_pose = np.eye(4)
                    self.last_keyframe_pose = np.eye(4)  # Reset last keyframe pose
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