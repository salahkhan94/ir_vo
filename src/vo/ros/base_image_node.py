#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import cv2

# Import feature detection and drawing modules
from vo.features.detectors import build_detector
from vo.features.draw import draw_keypoints
from vo.slam_system import SLAMSystem

class BaseImageNode(object):
    """
    Handles:  (1) init_node,
              (2) sync Image + CameraInfo (mono or stereo),
              (3) cv_bridge conversion,
              (4) debug image publisher stub.
    
    For monocular: img_topic and info_topic should be strings
    For stereo: img_topic and info_topic should be tuples (left_topic, right_topic)
    
    Child just overrides handle_frame() or handle_stereo_frame()
    """
    def __init__(self, node_name, img_topic, info_topic, queue=5, slop=0.02, stereo=False):
        rospy.init_node(node_name)
        self.bridge = CvBridge()
        self.stereo = stereo

        # Initialize default detector
        self.det = build_detector("orb")
        
        # Initialize SLAM system
        self.slam_system = SLAMSystem("orb")

        if stereo:
            # Stereo mode: expect tuples for topics
            if not isinstance(img_topic, (list, tuple)) or len(img_topic) != 2:
                raise ValueError("For stereo mode, img_topic must be a tuple/list of (left_topic, right_topic)")
            if not isinstance(info_topic, (list, tuple)) or len(info_topic) != 2:
                raise ValueError("For stereo mode, info_topic must be a tuple/list of (left_info_topic, right_info_topic)")
            
            # Create subscribers for stereo
            sub_left_img = Subscriber(img_topic[0], Image)
            sub_right_img = Subscriber(img_topic[1], Image)
            sub_left_info = Subscriber(info_topic[0], CameraInfo)
            sub_right_info = Subscriber(info_topic[1], CameraInfo)
            
            # Synchronize all four topics
            self.sync = ApproximateTimeSynchronizer(
                [sub_left_img, sub_right_img, sub_left_info, sub_right_info],
                queue, slop
            )
            self.sync.registerCallback(self._cb_stereo)
            
        else:
            # Monocular mode: expect single topics
            if isinstance(img_topic, (list, tuple)) or isinstance(info_topic, (list, tuple)):
                raise ValueError("For monocular mode, img_topic and info_topic must be strings")
            
            sub_img = Subscriber(img_topic, Image)
            sub_info = Subscriber(info_topic, CameraInfo)
            self.sync = ApproximateTimeSynchronizer([sub_img, sub_info], queue, slop)
            self.sync.registerCallback(self._cb_mono)

        self.pub_dbg = rospy.Publisher("~debug", Image, queue_size=1)
        
        # Register shutdown handler
        rospy.on_shutdown(self.shutdown)

    def _cb_mono(self, img_msg, info_msg):
        """Callback for monocular input"""
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        dbg = self.handle_frame(cv_img, info_msg, img_msg.header)
        if dbg is not None:
            self.pub_dbg.publish(
                self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8", header=img_msg.header)
            )

    def _cb_stereo(self, left_img_msg, right_img_msg, left_info_msg, right_info_msg):
        """Callback for stereo input"""
        left_cv_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
        right_cv_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
        
        dbg = self.handle_stereo_frame(
            left_cv_img, right_cv_img, 
            left_info_msg, right_info_msg, 
            left_img_msg.header
        )
        
        if dbg is not None:
            self.pub_dbg.publish(
                self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8", header=left_img_msg.header)
            )

    # Default implementation for monocular processing
    def handle_frame(self, cv_img, info_msg, header):
        """
        Default implementation for monocular processing.
        Performs feature detection, matching, and pose estimation.
        
        Args:
            cv_img: OpenCV image (numpy array)
            info_msg: CameraInfo message
            header: Header from the image message
            
        Returns:
            debug image (numpy array) with keypoints drawn and pose info
        """
        # Process frame with SLAM system
        success, pose = self.slam_system.process_frame(cv_img, info_msg)
        
        # Draw keypoints for visualization
        kps = self.det.detect(cv_img, None)
        debug_img = draw_keypoints(cv_img, kps)
        
        # Add tracking state and pose information to debug image
        tracking_state = self.slam_system.tracking_state
        cv2.putText(debug_img, f"State: {tracking_state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if success else (0, 0, 255), 2)
        
        if success and pose is not None:
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            cv2.putText(debug_img, f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add map statistics
            num_map_points, num_keyframes = self.slam_system.map.get_map_size()
            cv2.putText(debug_img, f"Map: {num_map_points} pts, {num_keyframes} KFs", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return debug_img

    # Default implementation for stereo processing
    def handle_stereo_frame(self, left_cv_img, right_cv_img, left_info_msg, right_info_msg, header):
        """
        Default implementation for stereo processing.
        Detects ORB keypoints on both images and creates a side-by-side debug image.
        
        Args:
            left_cv_img: Left OpenCV image (numpy array)
            right_cv_img: Right OpenCV image (numpy array)
            left_info_msg: Left camera CameraInfo message
            right_info_msg: Right camera CameraInfo message
            header: Header from the left image message
            
        Returns:
            debug image (numpy array) with side-by-side keypoint visualization
        """
        # Detect keypoints on both images
        left_kps = self.det.detect(left_cv_img, None)
        right_kps = self.det.detect(right_cv_img, None)
        
        # Draw keypoints on both images
        left_debug = draw_keypoints(left_cv_img, left_kps)
        right_debug = draw_keypoints(right_cv_img, right_kps)
        
        # Create a side-by-side debug image
        # Ensure both images have the same height
        h1, w1 = left_debug.shape[:2]
        h2, w2 = right_debug.shape[:2]
        
        # Resize right image to match left image height if needed
        if h1 != h2:
            right_debug = cv2.resize(right_debug, (int(w2 * h1 / h2), h1))
        
        # Concatenate images horizontally
        debug_img = np.hstack([left_debug, right_debug])
        
        # Add labels to distinguish left and right images
        cv2.putText(debug_img, "Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, "Right", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return debug_img

    def shutdown(self):
        """Shutdown handler for cleanup."""
        if hasattr(self, 'slam_system'):
            self.slam_system.shutdown()
