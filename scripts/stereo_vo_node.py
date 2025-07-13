#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vo.ros.base_image_node import BaseImageNode
# from vo.ros.converters import cam_info_to_K

class StereoVONode(BaseImageNode):
    def __init__(self):
        super().__init__(
            node_name="stereo_vo",
            img_topic=("/left_camera/image_raw", "/right_camera/image_raw"),
            info_topic=("/left_camera/camera_info", "/right_camera/camera_info"),
            stereo=True)  # Enable stereo mode

        # The base class now handles stereo feature detection by default
        # No need to override handle_stereo_frame unless custom processing is needed

if __name__ == "__main__":
    StereoVONode()
    import rospy; rospy.spin()
