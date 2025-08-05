#!/usr/bin/env python3
import cv2
import numpy as np

def draw_keypoints(img_bgr, keypoints, color=(0,255,0)):
    return cv2.drawKeypoints(
        img_bgr, keypoints, None,
        color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def draw_matched_keypoints(img_bgr, keypoints, color=(0,255,0), size=3):
    """
    Draw small squares around matched keypoints.
    
    Args:
        img_bgr: Input image (BGR format)
        keypoints: List of keypoints to draw
        color: Color of the squares (B, G, R)
        size: Size of the squares (half-width/height)
        
    Returns:
        Image with keypoints drawn as small squares
    """
    img_out = img_bgr.copy()
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # Draw a small square around the keypoint
        cv2.rectangle(img_out, 
                     (x - size, y - size), 
                     (x + size, y + size), 
                     color, 
                     1)  # 1 pixel thickness
    
    return img_out
