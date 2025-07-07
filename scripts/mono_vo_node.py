#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vo.ros.base_image_node import BaseImageNode
from vo.features.detectors import build_detector
from vo.features.draw import draw_keypoints
# from vo.ros.converters import cam_info_to_K

class MonoVONode(BaseImageNode):
    def __init__(self):
        super().__init__(
            node_name="mono_vo",
            img_topic="/camera/image_raw",
            info_topic="/camera/camera_info")

        self.det = build_detector("orb")

    # just algorithm + debug image
    def handle_frame(self, cv_img, info_msg, header):
        kps = self.det.detect(cv_img, None)
        return draw_keypoints(cv_img, kps)

if __name__ == "__main__":
    MonoVONode()
    import rospy; rospy.spin()