#!/usr/bin/env python3
import cv2

def build_detector(name="orb"):
    name = name.lower()
    if name == "orb":
        # 2000 keypoints, FAST threshold 20
        return cv2.ORB_create(nfeatures=2000, fastThreshold=20)
    if name == "fast":
        return cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    # stub for learned models
    raise ValueError(f"Unknown detector: {name}")
