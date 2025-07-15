#!/usr/bin/env python3
import numpy as np
import cv2
from typing import Tuple, Optional

class Camera:
    """
    Represents a camera with intrinsics, distortion, and projection operations.
    This is a core component of ORB-SLAM2 for camera modeling and coordinate transformations.
    """
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float, 
                 k1: float = 0.0, k2: float = 0.0, p1: float = 0.0, p2: float = 0.0, k3: float = 0.0):
        """
        Initialize camera with intrinsics and distortion parameters.
        
        Args:
            fx (float): Focal length in x direction
            fy (float): Focal length in y direction
            cx (float): Principal point x coordinate
            cy (float): Principal point y coordinate
            k1 (float): Radial distortion coefficient 1
            k2 (float): Radial distortion coefficient 2
            p1 (float): Tangential distortion coefficient 1
            p2 (float): Tangential distortion coefficient 2
            k3 (float): Radial distortion coefficient 3
        """
        # Intrinsic parameters
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        
        # Distortion coefficients
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.k3 = float(k3)
        
        # Camera matrix
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float32)
        
        # Distortion coefficients array
        self.D = np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)
        
        # Precomputed values
        self.inv_fx = 1.0 / self.fx
        self.inv_fy = 1.0 / self.fy
        
    def __str__(self):
        return f"Camera(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})"
    
    def __repr__(self):
        return self.__str__()
    
    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get camera intrinsics.
        
        Returns:
            Tuple[float, float, float, float]: (fx, fy, cx, cy)
        """
        return (self.fx, self.fy, self.cx, self.cy)
    
    def get_camera_matrix(self) -> np.ndarray:
        """
        Get the camera matrix K.
        
        Returns:
            np.ndarray: 3x3 camera matrix
        """
        return self.K.copy()
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """
        Get distortion coefficients.
        
        Returns:
            np.ndarray: Distortion coefficients array
        """
        return self.D.copy()
    
    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d (np.ndarray): 3D points in camera coordinates (N, 3)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (projected_points, valid_mask)
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        # Check if points are in front of camera
        valid_mask = points_3d[:, 2] > 0
        
        # Project to normalized plane
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]
        
        # Apply distortion
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r4 * r2
        
        # Radial distortion
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        
        # Tangential distortion
        tangential_x = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x**2)
        tangential_y = self.p1 * (r2 + 2 * y**2) + 2 * self.p2 * x * y
        
        # Apply distortion
        x_distorted = x * radial + tangential_x
        y_distorted = y * radial + tangential_y
        
        # Project to image plane
        u = self.fx * x_distorted + self.cx
        v = self.fy * y_distorted + self.cy
        
        projected_points = np.column_stack([u, v])
        
        return projected_points, valid_mask
    
    def unproject(self, points_2d: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Unproject 2D image coordinates to 3D camera coordinates.
        
        Args:
            points_2d (np.ndarray): 2D image points (N, 2)
            depths (np.ndarray): Depth values for each point (N,)
            
        Returns:
            np.ndarray: 3D points in camera coordinates (N, 3)
        """
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, -1)
        if depths.ndim == 0:
            depths = np.array([depths])
        
        # Convert to normalized coordinates
        x = (points_2d[:, 0] - self.cx) * self.inv_fx
        y = (points_2d[:, 1] - self.cy) * self.inv_fy
        
        # Undistort (simplified - assumes small distortion)
        # For more accurate undistortion, use cv2.undistortPoints
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r4 * r2
        
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        
        x_undistorted = x / radial
        y_undistorted = y / radial
        
        # Convert to 3D
        points_3d = np.column_stack([x_undistorted * depths, 
                                   y_undistorted * depths, 
                                   depths])
        
        return points_3d
    
    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Undistort 2D image points.
        
        Args:
            points_2d (np.ndarray): Distorted 2D points (N, 2)
            
        Returns:
            np.ndarray: Undistorted 2D points (N, 2)
        """
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, -1)
        
        # Use OpenCV for accurate undistortion
        undistorted = cv2.undistortPoints(points_2d.reshape(-1, 1, 2), 
                                         self.K, self.D, 
                                         P=self.K).reshape(-1, 2)
        return undistorted
    
    def distort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Distort 2D image points.
        
        Args:
            points_2d (np.ndarray): Undistorted 2D points (N, 2)
            
        Returns:
            np.ndarray: Distorted 2D points (N, 2)
        """
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, -1)
        
        # Convert to normalized coordinates
        x = (points_2d[:, 0] - self.cx) * self.inv_fx
        y = (points_2d[:, 1] - self.cy) * self.inv_fy
        
        # Apply distortion
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r4 * r2
        
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        
        tangential_x = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x**2)
        tangential_y = self.p1 * (r2 + 2 * y**2) + 2 * self.p2 * x * y
        
        x_distorted = x * radial + tangential_x
        y_distorted = y * radial + tangential_y
        
        # Convert back to image coordinates
        distorted = np.column_stack([x_distorted * self.fx + self.cx,
                                   y_distorted * self.fy + self.cy])
        
        return distorted
    
    def is_in_image(self, points_2d: np.ndarray, image_width: int, image_height: int, 
                   margin: float = 0.0) -> np.ndarray:
        """
        Check if 2D points are within image bounds.
        
        Args:
            points_2d (np.ndarray): 2D image points (N, 2)
            image_width (int): Image width
            image_height (int): Image height
            margin (float): Margin around image boundaries
            
        Returns:
            np.ndarray: Boolean mask indicating which points are in image
        """
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, -1)
        
        u, v = points_2d[:, 0], points_2d[:, 1]
        
        in_image = ((u >= margin) & (u < image_width - margin) & 
                   (v >= margin) & (v < image_height - margin))
        
        return in_image
    
    @classmethod
    def from_camera_info(cls, camera_info_msg):
        """
        Create Camera instance from ROS CameraInfo message.
        
        Args:
            camera_info_msg: ROS CameraInfo message
            
        Returns:
            Camera: New Camera instance
        """
        K = np.array(camera_info_msg.K).reshape(3, 3)
        D = np.array(camera_info_msg.D)
        
        return cls(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            k1=D[0] if len(D) > 0 else 0.0,
            k2=D[1] if len(D) > 1 else 0.0,
            p1=D[2] if len(D) > 2 else 0.0,
            p2=D[3] if len(D) > 3 else 0.0,
            k3=D[4] if len(D) > 4 else 0.0
        ) 