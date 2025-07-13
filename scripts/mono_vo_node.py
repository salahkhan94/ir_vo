#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vo.ros.base_image_node import BaseImageNode
# from vo.ros.converters import cam_info_to_K

class MonoVONode(BaseImageNode):
    def __init__(self):
        super().__init__(
            node_name="mono_vo",
            img_topic="/camera/image_raw",
            info_topic="/camera/camera_info",
            stereo=False)  # Explicitly set stereo=False for clarity

        # The base class now handles feature detection by default
        # No need to override handle_frame unless custom processing is needed

if __name__ == "__main__":
    MonoVONode()
    import rospy; rospy.spin()