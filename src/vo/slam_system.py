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
        self.keyframe_interval = 5  # Create keyframe every N frames (fallback) - reduced for more frequent keyframes
        self.min_keyframe_interval = 3  # Minimum frames between keyframes
        self.min_translation = 0.5  # Minimum translation distance (meters) - increased for more selective keyframe selection
        self.min_rotation = 0.2  # Minimum rotation angle (radians, ~11.5 degrees) - increased for more selective keyframe selection
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
        self._last_ba_time = 0.0  # Initialize to 0 so BA can run immediately
        
        # Tracking failure handling
        self.tracking_failure_count = 0
        self.max_tracking_failures = 10  # Increased from 5 to 10 for more robustness
        
        # Reinitialization flag
        self.reinitializing = False
        
        # Path history for trajectory visualization
        self.path_history = []
        self.path_keyframe_mapping = {}  # Maps path index to keyframe_id
        
        # Sliding window for bundle adjustment (last 10 frames)
        self.frame_window = []  # List of (frame_id, pose, keypoints, descriptors) tuples
        self.max_frame_window = 10  # Maximum number of frames in sliding window
        
        # Initialize mapping thread
        self.start_mapping_thread()  # Enable bundle adjustment thread
    
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
                            # Clear path history for fresh trajectory after successful initialization
                            self.path_history = []
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
            
            # Validate pose before setting current_pose
            if pose is not None and pose.shape == (4, 4):
                self.current_pose = pose  # track_frame now returns global pose directly
                
                # Add frame to sliding window for bundle adjustment
                self.add_frame_to_window(self.current_frame_id, pose, keypoints, descriptors)
            else:
                rospy.logwarn(f"Invalid pose shape from track_frame: {pose.shape if pose is not None else 'None'}")
                return False, None
                
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
                    thread.join(timeout=2.0)  # 2 second timeout for keyframe creation
                    
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
    
    def add_frame_to_window(self, frame_id, pose, keypoints, descriptors):
        """
        Add a frame to the sliding window for bundle adjustment.
        
        Args:
            frame_id: Frame ID
            pose: Camera pose (4x4 matrix)
            keypoints: Detected keypoints
            descriptors: Feature descriptors
        """
        # Add frame to window
        frame_data = {
            'frame_id': frame_id,
            'pose': pose.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors.copy() if descriptors is not None else None
        }
        
        self.frame_window.append(frame_data)
        
        # Keep only the last max_frame_window frames
        if len(self.frame_window) > self.max_frame_window:
            self.frame_window.pop(0)  # Remove oldest frame
    
    def get_current_keypoints(self):
        """
        Get the keypoints from the most recent frame processing.
        
        Returns:
            List of keypoints from the last processed frame
        """
        return getattr(self, '_current_keypoints', [])
    
    def track_frame(self, frame, keypoints, descriptors, K, header):
        """
        Track the current frame using 2D-2D correspondences and Essential matrix.
        
        Args:
            frame: Current frame
            keypoints: Detected keypoints
            descriptors: ORB descriptors
            K: Camera intrinsic matrix
            header: ROS header
            
        Returns:
            success: Whether tracking was successful
            pose: Estimated pose (global pose in world coordinates)
        """
        if self.reference_keyframe is None:
            rospy.logwarn("No reference keyframe available")
            rospy.logwarn(f"Tracking state: {self.tracking_state}, Current frame ID: {self.current_frame_id}")
            rospy.logwarn(f"Reference keyframe is None, this should not happen in {self.tracking_state} state")
            
            # Publish debug information
            debug_msg = f"Frame {self.current_frame_id}: No reference keyframe, State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Get keypoints and descriptors from reference keyframe
        ref_keypoints = self.reference_keyframe.get_keypoints()
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
        
        # Check minimum matches
        min_matches = max(8, min(20, len(good_matches) // 3))  # Ensure at least 8 matches
        if len(good_matches) < min_matches:
            rospy.logwarn(f"Not enough good matches: {len(good_matches)} < {min_matches}")
            debug_msg = f"Frame {self.current_frame_id}: Insufficient matches ({len(good_matches)}/{min_matches}), State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Extract 2D-2D correspondences
        pts1, pts2 = get_matched_points(ref_keypoints, keypoints, good_matches)
        
        # Estimate Essential Matrix
        E, E_mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        
        if E is None:
            rospy.logwarn("Failed to estimate Essential Matrix")
            debug_msg = f"Frame {self.current_frame_id}: Essential Matrix estimation failed, State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Count inliers from Essential Matrix
        E_inliers = np.sum(E_mask)
        rospy.loginfo(f"Essential Matrix inliers: {E_inliers}/{len(good_matches)}")
        
        min_inliers = max(8, min(15, len(good_matches) // 4))  # Ensure at least 8 inliers
        if E_inliers < min_inliers:
            rospy.logwarn(f"Not enough Essential Matrix inliers: {E_inliers} < {min_inliers}")
            debug_msg = f"Frame {self.current_frame_id}: Insufficient Essential Matrix inliers ({E_inliers}/{min_inliers}), State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Filter points using Essential Matrix mask
        inlier_pts1 = pts1[E_mask.ravel() == 1]
        inlier_pts2 = pts2[E_mask.ravel() == 1]
        
        # Recover pose from Essential Matrix
        retval, R, t, pose_mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)
        
        if R is None or t is None:
            rospy.logwarn("Failed to recover pose from Essential Matrix")
            debug_msg = f"Frame {self.current_frame_id}: Pose recovery failed, State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Count inliers from pose recovery
        pose_inliers = np.sum(pose_mask)
        rospy.loginfo(f"Pose recovery inliers: {pose_inliers}/{len(inlier_pts1)}")
        
        if pose_inliers < min_inliers:
            rospy.logwarn(f"Not enough pose recovery inliers: {pose_inliers} < {min_inliers}")
            debug_msg = f"Frame {self.current_frame_id}: Insufficient pose recovery inliers ({pose_inliers}/{min_inliers}), State: {self.tracking_state}"
            self.debug_pub.publish(debug_msg)
            return False, None
        
        # Construct relative pose matrix
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = R
        relative_pose[:3, 3] = t.flatten()
        
        # Get reference keyframe pose (global pose)
        reference_pose = self.reference_keyframe.get_pose()
        
        # Validate reference pose
        if reference_pose is None or reference_pose.shape != (4, 4):
            rospy.logwarn(f"Invalid reference pose shape: {reference_pose.shape if reference_pose is not None else 'None'}")
            return False, None
        
        # Compute global pose by composing reference pose with relative pose
        global_pose = reference_pose @ relative_pose
        
        # Validate global pose
        if global_pose.shape != (4, 4):
            rospy.logwarn(f"Invalid global pose shape: {global_pose.shape}")
            return False, None
        
        # Publish global pose and transform
        rospy.loginfo(f"Publishing global pose: [{global_pose[0,3]:.2f}, {global_pose[1,3]:.2f}, {global_pose[2,3]:.2f}]")
        self.publish_pose(global_pose, header)
        self.publish_path(global_pose, header)
        
        # Publish success debug information
        debug_msg = f"Frame {self.current_frame_id}: Tracking SUCCESS! Global Pose: [{global_pose[0,3]:.2f}, {global_pose[1,3]:.2f}, {global_pose[2,3]:.2f}], State: {self.tracking_state}"
        self.debug_pub.publish(debug_msg)
        
        return True, global_pose
    
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
            # Validate current pose before creating keyframe
            if self.current_pose is None or self.current_pose.shape != (4, 4):
                rospy.logwarn(f"Invalid current pose for keyframe creation: {self.current_pose.shape if self.current_pose is not None else 'None'}")
                return None
            
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
            rospy.loginfo(f"✅ KEYFRAME CREATED! ID: {keyframe_id}, Frame: {self.current_frame_id}")
            rospy.loginfo(f"   └─ Position: [{self.current_pose[0,3]:.3f}, {self.current_pose[1,3]:.3f}, {self.current_pose[2,3]:.3f}]")
            rospy.loginfo(f"   └─ Map points: {len(self.map.get_map_points())}")
            
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
            # Create proper 4x4 pose matrix for triangulation
            pose2 = np.eye(4)
            pose2[:3, :3] = R
            pose2[:3, 3] = t.flatten()
            points_3d = self.triangulate_points(pts1, pts2, np.eye(4), pose2, K)
            
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
            
            # Create proper 4x4 pose matrix for second keyframe
            pose2 = np.eye(4)
            pose2[:3, :3] = R
            pose2[:3, 3] = t.flatten()
            
            keyframe2 = KeyFrame(
                frame_id=self.current_frame_id,
                timestamp=0.0,
                pose=pose2,
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
                    # For KITTI dataset, depths can be much larger (500-600m is normal)
                    if point_3d[2] > 0.1 and point_3d[2] < 1000.0:  # Depth between 0.1m and 1000m
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
            # Create proper 4x4 pose matrix for triangulation
            pose2 = np.eye(4)
            pose2[:3, :3] = R
            pose2[:3, 3] = t.flatten()
            points_3d = self.triangulate_points(pts1, pts2, np.eye(4), pose2, K)
            
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
            
            # Create proper 4x4 pose matrix for second keyframe
            pose2 = np.eye(4)
            pose2[:3, :3] = R
            pose2[:3, 3] = t.flatten()
            
            keyframe2 = KeyFrame(
                frame_id=self.current_frame_id,
                timestamp=0.0,
                pose=pose2,
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
                    # For KITTI dataset, depths can be much larger (500-600m is normal)
                    if point_3d[2] > 0.1 and point_3d[2] < 1000.0:  # Depth between 0.1m and 1000m
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
        # Limit the number of new map points to avoid overwhelming the system
        max_new_map_points = 500  # Limit to 500 new map points per keyframe
        new_map_points = 0
        
        for i, (match, point_3d) in enumerate(zip(good_matches, points_3d)):
            # Stop if we've reached the limit
            if new_map_points >= max_new_map_points:
                break
                
            # Check if point is valid (positive depth, reasonable distance)
            # For KITTI dataset, depths can be much larger (500-600m is normal)
            if point_3d[2] > 0.1 and point_3d[2] < 1000.0:  # Depth between 0.1m and 1000m
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
            # else:
            #     rospy.logwarn(f"DEBUG: Invalid 3D point {i}: depth={point_3d[2]:.2f}")
        
        rospy.loginfo(f"Created {new_map_points} new map points from triangulation (limited to {max_new_map_points})")
        
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
        
        # 3. Check minimum interval between keyframes
        min_interval_ok = (self.current_frame_id - self.last_keyframe_id) >= self.min_keyframe_interval
        
        # 4. Fallback to fixed interval if no other criteria met
        interval_criteria = (self.current_frame_id - self.last_keyframe_id) >= self.keyframe_interval
        
        # Create keyframe if motion/tracking criteria are met AND minimum interval is satisfied
        should_create = (motion_criteria or tracking_criteria or interval_criteria) and min_interval_ok
        
        if should_create:
            rospy.loginfo(f"🎯 KEYFRAME SELECTED! Frame {self.current_frame_id}")
            rospy.loginfo(f"   └─ Motion criteria: {motion_criteria}")
            rospy.loginfo(f"   └─ Tracking criteria: {tracking_criteria}")
            rospy.loginfo(f"   └─ Interval criteria: {interval_criteria}")
            rospy.loginfo(f"   └─ Min interval check: {min_interval_ok} (frames since last: {self.current_frame_id - self.last_keyframe_id})")
        else:
            if not min_interval_ok:
                rospy.loginfo(f"⏭️  No keyframe selected for frame {self.current_frame_id} - minimum interval not satisfied ({self.current_frame_id - self.last_keyframe_id} < {self.min_keyframe_interval})")
            else:
                rospy.loginfo(f"⏭️  No keyframe selected for frame {self.current_frame_id}")
        
        return should_create
    
    def check_motion_criteria(self):
        """
        Check if motion-based criteria are met for keyframe creation.
        
        Returns:
            True if motion criteria are met, False otherwise
        """
        if self.reference_keyframe is None:
            return False
        
        # Get current pose and reference keyframe pose (not last keyframe pose)
        current_pose = self.current_pose
        reference_pose = self.reference_keyframe.get_pose()
        
        # Validate pose matrices
        if (current_pose is None or reference_pose is None or
            current_pose.shape != (4, 4) or reference_pose.shape != (4, 4)):
            rospy.logwarn("Invalid pose matrices for motion criteria check")
            return False
        
        # Check if matrices are valid (not all zeros, proper structure)
        if (np.allclose(current_pose, 0) or np.allclose(reference_pose, 0) or
            np.allclose(current_pose, np.eye(4)) and np.allclose(reference_pose, np.eye(4))):
            rospy.logwarn("Pose matrices are invalid (all zeros or identity) for motion check")
            return False
        
        try:
            # Compute relative transformation from reference keyframe to current pose
            relative_pose = np.linalg.inv(reference_pose) @ current_pose
            
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
                rospy.loginfo(f"🚀 Motion criteria met: translation={translation_distance:.3f}m (threshold: {self.min_translation}m), rotation={rotation_angle:.3f}rad (threshold: {self.min_rotation}rad)")
            else:
                rospy.loginfo(f"📏 Motion criteria not met: translation={translation_distance:.3f}m (threshold: {self.min_translation}m), rotation={rotation_angle:.3f}rad (threshold: {self.min_rotation}rad)")
            
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
                rospy.loginfo(f"📊 Tracking quality criteria met: features={tracked_features} (threshold: <{self.min_tracked_features}), quality={tracking_quality:.3f} (threshold: <{self.min_tracking_quality})")
            
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
                        
                        # Clear path history for fresh trajectory
                        self.path_history = []
                        self.path_keyframe_mapping = {}
                        
                        # Clear frame window
                        self.frame_window = []
                        
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
                    
                    # Clear path history for fresh trajectory
                    self.path_history = []
                    self.path_keyframe_mapping = {}
                    
                    # Clear frame window
                    self.frame_window = []
                    
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
                self.path_history = []
                self.path_keyframe_mapping = {}
                self.frame_window = []
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
        Performs bundle adjustment on last 10 frames and updates path.
        """
        rospy.loginfo("Mapping thread started")
        
        while self.mapping_running:
            try:
                rospy.loginfo("=== MAPPING LOOP ITERATION START ===")
                
                # Skip bundle adjustment if reinitializing
                if self.reinitializing:
                    rospy.loginfo("  Skipping BA - reinitializing")
                    time.sleep(0.1)
                    continue
                
                rospy.loginfo(f"  Frame window size: {len(self.frame_window)}")
                
                # Get last 10 frames for bundle adjustment
                if len(self.frame_window) >= 2:
                    rospy.loginfo(f"  ✅ Frame window has {len(self.frame_window)} frames, proceeding with BA")
                    rospy.loginfo(f"Performing bundle adjustment on {len(self.frame_window)} frames")
                    
                    try:
                        rospy.loginfo("  🔄 About to call perform_bundle_adjustment_on_frames...")
                        # Perform bundle adjustment on frame window
                        optimized_poses = self.perform_bundle_adjustment_on_frames(self.frame_window)
                        rospy.loginfo("  ✅ perform_bundle_adjustment_on_frames completed")
                        
                        # Update path with optimized poses
                        if optimized_poses:
                            rospy.loginfo("  🔄 Updating path with optimized poses...")
                            self.update_path_with_optimized_poses(optimized_poses)
                            rospy.loginfo("  ✅ Path updated successfully")
                        else:
                            rospy.loginfo("  ⚠️  No optimized poses returned from BA")
                    except Exception as e:
                        rospy.logerr(f"Bundle adjustment failed: {e}")
                        rospy.logerr(f"Exception type: {type(e).__name__}")
                        import traceback
                        rospy.logerr(f"Traceback: {traceback.format_exc()}")
                        # Continue without bundle adjustment
                else:
                    rospy.loginfo("  ⏭️  Frame window too small, skipping BA")
                
                rospy.loginfo("  🔄 About to cull map points...")
                # Cull bad map points
                self.map.cull_map_points()
                rospy.loginfo("  ✅ Map points culled")
                
                rospy.loginfo("  😴 Sleeping for 2 seconds...")
                # Sleep for bundle adjustment (0.5 Hz - every 2 seconds)
                time.sleep(2.0)
                rospy.loginfo("  ✅ Woke up from sleep")
                
                # Add a timeout check to prevent getting stuck
                if hasattr(self, '_last_ba_time'):
                    if time.time() - self._last_ba_time > 30:  # 30 second timeout
                        rospy.logwarn("Bundle adjustment timeout, resetting")
                        self.ba_running = False
                        # Don't reset _last_ba_time here - only reset when BA actually runs
                # Don't set _last_ba_time here - only set it when BA actually runs
                
                rospy.loginfo("=== MAPPING LOOP ITERATION END ===")
                
            except Exception as e:
                rospy.logerr(f"Error in mapping thread: {e}")
                rospy.logerr(f"Exception type: {type(e).__name__}")
                import traceback
                rospy.logerr(f"Traceback: {traceback.format_exc()}")
                time.sleep(0.5)
    
    def perform_bundle_adjustment_on_frames(self, frame_window):
        """
        Perform bundle adjustment on the given frame window.
        
        Args:
            frame_window: List of frame data dictionaries
            
        Returns:
            optimized_poses: Dictionary mapping frame_id to optimized pose, or None if failed
        """
        rospy.loginfo("🚀 ENTERING perform_bundle_adjustment_on_frames")
        
        if self.ba_running:
            rospy.loginfo("  ⏭️  BA already running, returning None")
            return None
        
        # Check if enough time has passed since last bundle adjustment
        current_time = time.time()
        if hasattr(self, '_last_ba_time') and current_time - self._last_ba_time < 10.0:  # Increased to 10 seconds
            rospy.loginfo("  ⏭️  Not enough time since last BA, returning None")
            return None  # Skip if less than 10 seconds since last BA
        
        rospy.loginfo("  ✅ Setting ba_running = True")
        self.ba_running = True
        
        try:
            # Create observations from frame window for full bundle adjustment
            # We need to triangulate 3D points between frames and create observations
            
            if len(frame_window) < 2:
                return None
            
            # Skip bundle adjustment if too many frames (can be slow)
            if len(frame_window) > 8:  # Limit to 8 frames
                rospy.logwarn(f"Too many frames ({len(frame_window)}), skipping bundle adjustment")
                return None
            
            # Prepare optimization variables
            poses = []
            valid_frames = []
            
            # Collect poses from frame window
            for frame_data in frame_window:
                pose = frame_data['pose']
                # Ensure pose is a 4x4 numpy array
                if pose is not None:
                    pose_array = np.array(pose, dtype=np.float64)
                    if pose_array.shape == (4, 4):
                        poses.append(pose_array)
                        valid_frames.append(frame_data)
                    else:
                        rospy.logwarn(f"Invalid pose shape: {pose_array.shape}, skipping frame {frame_data['frame_id']}")
                        continue
                else:
                    rospy.logwarn(f"None pose for frame {frame_data['frame_id']}, skipping")
                    continue
            
            if len(poses) == 0:
                rospy.logwarn("No valid poses for bundle adjustment")
                return None
            
            rospy.loginfo("=== STARTING FULL BUNDLE ADJUSTMENT ===")
            
            # Create observations from feature correspondences between frames
            rospy.loginfo("Step 1: Creating observations from feature correspondences...")
            observations_list = []
            points_3d = []
            point_id_map = {}
            point_counter = 0
            
            # Extract camera matrix from first frame (assuming same for all frames)
            K = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]], dtype=np.float64)
            rospy.loginfo(f"Camera matrix: {K}")
            
            # Create feature correspondences between consecutive frames
            rospy.loginfo(f"Step 2: Processing {len(valid_frames)} frames for feature matching...")
            for i in range(len(valid_frames) - 1):
                rospy.loginfo(f"  Processing frame pair {i} -> {i+1}")
                frame1_data = valid_frames[i]
                frame2_data = valid_frames[i + 1]
                
                rospy.loginfo(f"    Frame {i}: ID={frame1_data['frame_id']}, descriptors shape={frame1_data['descriptors'].shape if frame1_data['descriptors'] is not None else 'None'}")
                rospy.loginfo(f"    Frame {i+1}: ID={frame2_data['frame_id']}, descriptors shape={frame2_data['descriptors'].shape if frame2_data['descriptors'] is not None else 'None'}")
                
                # Match features between consecutive frames
                rospy.loginfo(f"    Matching features between frames...")
                matches, good_matches = match_features(
                    frame1_data['descriptors'], 
                    frame2_data['descriptors'], 
                    self.detector_type
                )
                
                rospy.loginfo(f"    Found {len(good_matches)} good matches out of {len(matches)} total matches")
                
                if len(good_matches) < 50:  # Need sufficient matches
                    rospy.loginfo(f"    Insufficient matches ({len(good_matches)} < 50), skipping this pair")
                    continue
                
                # Extract matched points
                rospy.loginfo(f"    Extracting matched points...")
                pts1, pts2 = get_matched_points(frame1_data['keypoints'], frame2_data['keypoints'], good_matches)
                rospy.loginfo(f"    Extracted {len(pts1)} matched point pairs")
                
                # Triangulate 3D points
                pose1 = frame1_data['pose']
                pose2 = frame2_data['pose']
                
                rospy.loginfo(f"    Triangulating 3D points...")
                # Triangulate points
                points_3d_batch = self.triangulate_points(pts1, pts2, pose1, pose2, K)
                rospy.loginfo(f"    Triangulated {len(points_3d_batch)} 3D points")
                
                # Add valid 3D points and observations
                rospy.loginfo(f"    Adding valid 3D points and observations...")
                valid_points_added = 0
                for j, (match, point_3d) in enumerate(zip(good_matches, points_3d_batch)):
                    if point_3d is not None and np.linalg.norm(point_3d) > 0.1 and np.linalg.norm(point_3d) < 1000.0:
                        # Check if we've seen this point before
                        point_key = (frame1_data['frame_id'], match.queryIdx)
                        if point_key not in point_id_map:
                            point_id_map[point_key] = point_counter
                            points_3d.append(point_3d)
                            point_counter += 1
                        
                        # Add observations for both frames
                        observations_list.append({
                            'frame_idx': i,
                            'point_idx': point_id_map[point_key],
                            'observation': pts1[j][0]  # 2D observation in frame1
                        })
                        observations_list.append({
                            'frame_idx': i + 1,
                            'point_idx': point_id_map[point_key],
                            'observation': pts2[j][0]  # 2D observation in frame2
                        })
                        valid_points_added += 1
                
                rospy.loginfo(f"    Added {valid_points_added} valid observations for this frame pair")
            
            rospy.loginfo(f"Step 3: Bundle adjustment setup complete")
            rospy.loginfo(f"  Total observations: {len(observations_list)}")
            rospy.loginfo(f"  Total 3D points: {len(points_3d)}")
            rospy.loginfo(f"  Total poses: {len(poses)}")
            
            if len(observations_list) < 20:  # Need sufficient observations
                rospy.logwarn(f"Insufficient observations for bundle adjustment: {len(observations_list)} < 20")
                return None
            
            # Skip if too many observations (can be slow)
            if len(observations_list) > 2000:  # Limit observations for speed
                rospy.logwarn(f"Too many observations ({len(observations_list)}), skipping bundle adjustment")
                return None
            
            # Flatten optimization variables
            rospy.loginfo("Step 4: Preparing optimization variables...")
            poses_flat = np.array(poses, dtype=np.float64).flatten()
            points_3d_flat = np.array(points_3d, dtype=np.float64).flatten()
            
            rospy.loginfo(f"  Poses flat shape: {poses_flat.shape}")
            rospy.loginfo(f"  Points 3D flat shape: {points_3d_flat.shape}")
            
            # Optimization variables: [poses, points_3d]
            x0 = np.concatenate([poses_flat, points_3d_flat])
            rospy.loginfo(f"  Combined optimization variables shape: {x0.shape}")
            
            # Check for invalid values
            if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
                rospy.logwarn("Invalid values in optimization variables (NaN or Inf)")
                return None
            
            rospy.loginfo(f"Bundle adjustment setup: {len(poses)} poses, {len(points_3d)} 3D points, {len(observations_list)} observations")
            rospy.loginfo("Starting bundle adjustment optimization...")
            rospy.loginfo(f"Bundle adjustment parameters: {len(valid_frames)} frames")
            
            # Log frame information for debugging
            for i, frame_data in enumerate(valid_frames):
                rospy.loginfo(f"Frame {i}: ID={frame_data['frame_id']}, pose shape={poses[i].shape}")
            
            # Perform optimization using scipy with threading timeout
            rospy.loginfo("Step 5: Starting optimization with scipy.optimize.least_squares...")
            from scipy.optimize import least_squares
            import threading
            import queue
            
            # Set up timeout for optimization using threading
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def optimization_worker():
                rospy.loginfo("  Optimization worker thread started")
                try:
                    rospy.loginfo("  Calling least_squares...")
                    result = least_squares(
                        self.reprojection_error_frames,
                        x0,
                        args=(observations_list, valid_frames, len(poses), len(points_3d)),
                        method='lm',
                        max_nfev=15,  # Very limited iterations for speed
                        ftol=1e-1,    # Very relaxed function tolerance
                        xtol=1e-1     # Very relaxed variable tolerance
                    )
                    rospy.loginfo("  least_squares completed successfully")
                    result_queue.put(result)
                except Exception as e:
                    rospy.logerr(f"  Exception in optimization worker: {e}")
                    exception_queue.put(e)
            
            # Start optimization in a separate thread
            rospy.loginfo("  Starting optimization worker thread...")
            worker_thread = threading.Thread(target=optimization_worker)
            worker_thread.daemon = True
            worker_thread.start()
            
            # Wait for completion with timeout
            rospy.loginfo("  Waiting for optimization to complete (timeout: 2 seconds)...")
            try:
                worker_thread.join(timeout=2.0)  # 2 second timeout
                if worker_thread.is_alive():
                    rospy.logwarn("Bundle adjustment optimization timed out after 2 seconds")
                    return None
                
                if not exception_queue.empty():
                    raise exception_queue.get()
                
                rospy.loginfo("  Getting result from queue...")
                result = result_queue.get_nowait()
                rospy.loginfo("  Successfully retrieved optimization result")
            except queue.Empty:
                rospy.logwarn("Bundle adjustment optimization failed to return result")
                return None
            
            rospy.loginfo("Step 6: Processing optimization results...")
            if result.success:
                rospy.loginfo("  Optimization was successful")
                # Update poses and 3D points
                poses_flat_opt = result.x[:len(poses_flat)]
                points_3d_flat_opt = result.x[len(poses_flat):]
                
                rospy.loginfo(f"  Optimized poses flat shape: {poses_flat_opt.shape}")
                rospy.loginfo(f"  Optimized points 3D flat shape: {points_3d_flat_opt.shape}")
                
                # Reshape and update
                poses_opt = poses_flat_opt.reshape(-1, 4, 4)
                rospy.loginfo(f"  Reshaped poses to: {poses_opt.shape}")
                
                # Create optimized poses dictionary
                optimized_poses = {frame_data['frame_id']: poses_opt[i] for i, frame_data in enumerate(valid_frames)}
                rospy.loginfo(f"  Created optimized poses dictionary with {len(optimized_poses)} entries")
                
                rospy.loginfo(f"Bundle adjustment completed successfully with {len(observations_list)} observations")
                rospy.loginfo("=== BUNDLE ADJUSTMENT COMPLETED SUCCESSFULLY ===")
                
                # Return optimized poses for path update
                return optimized_poses
            else:
                rospy.logwarn("Bundle adjustment optimization failed")
                rospy.logwarn(f"  Optimization message: {result.message}")
                return None
        
        except Exception as e:
            rospy.logerr(f"Error in bundle adjustment: {e}")
            return None
        
        finally:
            self.ba_running = False
            self._last_ba_time = time.time()
            rospy.loginfo("Bundle adjustment completed (success or failure)")
    
    def reprojection_error_frames(self, x, observations, valid_frames, num_poses, num_points):
        """
        Compute reprojection error for bundle adjustment.
        
        Args:
            x: Optimization variables [poses, points_3d]
            observations: List of observations
            valid_frames: List of frame data dictionaries
            num_poses: Number of poses
            num_points: Number of 3D points
            
        Returns:
            Reprojection errors
        """
        # Extract poses and 3D points from optimization variables
        poses_flat = x[:num_poses * 16]  # 4x4 = 16 elements per pose
        points_3d_flat = x[num_poses * 16:]
        
        poses = poses_flat.reshape(num_poses, 4, 4)
        points_3d = points_3d_flat.reshape(num_points, 3)
        
        # Extract camera matrix (hardcoded for now)
        K = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]], dtype=np.float64)
        
        errors = []
        
        # Add debug counter
        processed_obs = 0
        
        for obs in observations:
            frame_idx = obs['frame_idx']
            point_idx = obs['point_idx']
            observation = obs['observation']
            
            # Bounds checking
            if frame_idx >= len(poses) or point_idx >= len(points_3d):
                continue
            
            pose = poses[frame_idx]
            point_3d = points_3d[point_idx]
            
            # Project 3D point to 2D
            point_homogeneous = np.append(point_3d, 1.0)
            point_camera = pose @ point_homogeneous
            point_camera = point_camera[:3] / point_camera[3]  # Perspective division
            
            if point_camera[2] <= 0:  # Point behind camera
                errors.extend([1000.0, 1000.0])  # Large error
                continue
            
            # Project to image plane
            point_2d = K @ point_camera[:3]
            point_2d = point_2d[:2] / point_2d[2]  # Perspective division
            
            # Compute reprojection error
            error = point_2d - observation
            errors.extend(error)
            
            processed_obs += 1
            
            # Print progress every 100 observations
            if processed_obs % 100 == 0:
                rospy.loginfo(f"    Reprojection error: processed {processed_obs}/{len(observations)} observations")
        
        rospy.loginfo(f"    Reprojection error: completed {processed_obs} observations, returning {len(errors)} errors")
        return np.array(errors)
    
    def update_path_with_optimized_poses(self, optimized_poses):
        """
        Update the path history with optimized poses from bundle adjustment.
        
        Args:
            optimized_poses: Dictionary mapping keyframe_id to optimized pose
        """
        try:
            # Update path history with optimized poses
            updated_count = 0
            
            # Use the path_keyframe_mapping to find which path entries correspond to optimized keyframes
            for path_index, keyframe_id in self.path_keyframe_mapping.items():
                if keyframe_id in optimized_poses:
                    if path_index < len(self.path_history):
                        self.path_history[path_index] = optimized_poses[keyframe_id].copy()
                        updated_count += 1
                        rospy.loginfo(f"Updated path entry {path_index} (keyframe {keyframe_id}) with optimized pose")
            
            if updated_count > 0:
                rospy.loginfo(f"Updated {updated_count} poses in path history with bundle adjustment results")
                
                # Publish updated path
                if len(self.path_history) > 0:
                    # Create a dummy header for path publishing
                    from std_msgs.msg import Header
                    header = Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = "world"
                    
                    # Publish the updated path
                    self.publish_path(self.path_history[-1], header)
        
        except Exception as e:
            rospy.logerr(f"Error updating path with optimized poses: {e}")
    
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
        # Add current pose to path history
        path_index = len(self.path_history)
        self.path_history.append(pose_matrix.copy())
        
        # Track which keyframe this pose corresponds to (if it's a keyframe)
        if hasattr(self, 'current_frame_id'):
            self.path_keyframe_mapping[path_index] = self.current_frame_id
        
        # Create Path message
        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = "world"
        
        # Add all poses to path
        for pose in self.path_history:
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
    
    def shutdown(self):
        """Shutdown the SLAM system."""
        self.stop_mapping_thread()
        rospy.loginfo("SLAM system shutdown complete") 