#!/usr/bin/env python3
import cv2

def compute_descriptors(detector, image, keypoints):
    """
    Works for ORB, BRISK, etc. (detector doubles as descriptor extractor)
    Returns: keypoints, descriptor ndarray
    """
    if hasattr(detector, "compute"):
        return detector.compute(image, keypoints)
    raise RuntimeError("Detector has no compute() method")
