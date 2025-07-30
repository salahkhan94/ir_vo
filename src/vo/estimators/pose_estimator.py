#!/usr/bin/env python3
import cv2
import numpy as np

def estimate_fundamental_matrix(pts1, pts2, method=cv2.FM_RANSAC, ransac_reproj_thresh=3.0):
    """
    Estimate Fundamental Matrix from matched points.
    
    Args:
        pts1: Points from first frame (Nx1x2 array)
        pts2: Points from second frame (Nx1x2 array)
        method: Method for fundamental matrix estimation
        ransac_reproj_thresh: RANSAC reprojection threshold
    
    Returns:
        F: Fundamental matrix
        mask: Inlier mask
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, method, ransac_reproj_thresh)
    return F, mask

def estimate_essential_matrix(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5):
    """
    Estimate Essential Matrix from matched points and camera intrinsics using RANSAC.
    
    Args:
        pts1: Points from first frame (Nx1x2 array)
        pts2: Points from second frame (Nx1x2 array)
        K: Camera intrinsic matrix (3x3)
        method: Method for essential matrix estimation (default: RANSAC)
        prob: Confidence level (default: 0.999 for 99.9% confidence)
        threshold: RANSAC threshold in pixels (default: 1.5 for KITTI)
    
    Returns:
        E: Essential matrix
        mask: Inlier mask
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method, prob, threshold)
    return E, mask

def recover_pose(E, pts1, pts2, K):
    """
    Recover rotation and translation from Essential Matrix.
    
    Args:
        E: Essential matrix
        pts1: Points from first frame (Nx1x2 array)
        pts2: Points from second frame (Nx1x2 array)
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        mask: Inlier mask
    """
    # Normalize points
    pts1_norm = cv2.undistortPoints(pts1, K, None)
    pts2_norm = cv2.undistortPoints(pts2, K, None)
    
    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm, K)
    
    return R, t, mask

def camera_info_to_K(camera_info):
    """
    Extract camera intrinsic matrix from CameraInfo message.
    
    Args:
        camera_info: ROS CameraInfo message
    
    Returns:
        K: Camera intrinsic matrix (3x3 numpy array)
    """
    K = np.array(camera_info.K).reshape(3, 3)
    return K 