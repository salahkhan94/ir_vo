#!/usr/bin/env python3
import cv2

def draw_keypoints(img_bgr, keypoints, color=(0,255,0)):
    return cv2.drawKeypoints(
        img_bgr, keypoints, None,
        color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
