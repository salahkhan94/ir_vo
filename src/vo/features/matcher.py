#!/usr/bin/env python3
import cv2
import numpy as np

def match_features(descriptors1, descriptors2, detector_type="orb", ratio_test=0.75):
    """
    Match features between two sets of descriptors.
    
    Args:
        descriptors1: Descriptors from the first frame
        descriptors2: Descriptors from the second frame
        detector_type: Type of detector used (orb, sift, etc.)
        ratio_test: Ratio for Lowe's ratio test (default 0.75)
    
    Returns:
        matches: List of all matches
        good_matches: Filtered matches based on ratio test
    """
    if detector_type.lower() == "orb":
        # For ORB, use Hamming distance and Lowe's ratio test
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)
    elif detector_type.lower() in ["sift", "surf"]:
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    return matches, good_matches

def get_matched_points(keypoints1, keypoints2, matches):
    """
    Extract matched point coordinates from keypoints and matches.
    
    Args:
        keypoints1: Keypoints from the first frame
        keypoints2: Keypoints from the second frame
        matches: List of matches between keypoints
    
    Returns:
        pts1: Matched points from first frame (Nx1x2 array)
        pts2: Matched points from second frame (Nx1x2 array)
    """
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    return pts1, pts2 