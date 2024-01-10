#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import time
import cv2

def image_callback(img_sub):
    try:
        # Convert the ROS Image message to a OpenCV image

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img_sub, desired_encoding="bgr8") #

        # # Save the image to the specified directory with a unique timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"image_{timestamp}.jpg"
        directory = "/home/plomenuc/plome_ws/src/stereo_plome/images"
        filepath = os.path.join(directory, filename)

        # Save the image
        cv2.imwrite(filepath, cv_image)
        rospy.loginfo(f"Image saved: {filepath}")

    except Exception as e:
        rospy.logerr(f"Error processing the image: {e}")

def image_subscriber():
    rospy.init_node('image_saver', anonymous=True)
    # rospy.Subscriber("/stereo_ch3/scaled_x2/right/image_rect_color", Image, image_callback)
    rospy.Subscriber("/stereo_ch3/left/image_raw", Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    image_subscriber()