#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
class BaseImageNode(object):
    """
    Handles:  (1) init_node,
              (2) sync Image + CameraInfo,
              (3) cv_bridge conversion,
              (4) debug image publisher stub.
    Child just overrides handle_frame(cv_img, info_msg)
    """
    def __init__(self, node_name, img_topic, info_topic, queue=5, slop=0.02):
        rospy.init_node(node_name)
        self.bridge = CvBridge()

        sub_img  = Subscriber(img_topic,  Image)
        sub_info = Subscriber(info_topic, CameraInfo)
        self.sync = ApproximateTimeSynchronizer([sub_img, sub_info],
                                                queue, slop)
        self.sync.registerCallback(self._cb)

        self.pub_dbg = rospy.Publisher("~debug", Image, queue_size=1)

    def _cb(self, img_msg, info_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        dbg = self.handle_frame(cv_img, info_msg, img_msg.header)
        if dbg is not None:
            self.pub_dbg.publish(
                self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8",
                                          header=img_msg.header))

    # override me
    def handle_frame(self, cv_img, info_msg, header):
        raise NotImplementedError
