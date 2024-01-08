#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import torch

from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib
import cv2
import time
import sys
print(sys.path)
import numpy as np

import copy
import rospy
import sensor_msgs.msg
from sensor_msgs.msg import Image, CameraInfo

import message_filters
# from sea_cucumber_detection.msg import det
# from sea_cucumber_detection.msg import holo_detections
from stereo_plome.msg import det_bb
from stereo_plome.msg import mask_point
from stereo_plome.msg import yolov8_BB_latlon
FRAME_SKIP=1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from cv_bridge import CvBridge, CvBridgeError

from math import pi,tan
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header, String
from cola2_msgs.msg import NavSts
# from shapely.geometry import Polygon


from ultralytics import YOLO
import imageio

import shutil
import PIL

class Object_detection:

  def __init__(self, name): # clase constructor

    self.name = name
    # Params
    self.init = False

    self.detections = yolov8_BB_latlon()
    self.det = det_bb()

	  # Params
    self.model_path = rospy.get_param("pathWeights")
    self.detected_class = rospy.get_param("detected_class","urchin")
    self.confidenceThreshold = rospy.get_param("confidenceThreshold")
    self.writeFileResults= rospy.get_param("writeFileResults")
    self.robot_name = rospy.get_param('robot_name')
    self.period = rospy.get_param("period")
    self.project_path = rospy.get_param("project_path")
    self.project_name = rospy.get_param("project_name")
    self.out_folder=os.path.join(self.project_path,self.project_name)
    self.inference_finished=True

    #LOAD MODEL:
    self.model = YOLO(self.model_path)
    image_sub = message_filters.Subscriber('image_rect_color', Image)
    info_sub = message_filters.Subscriber('camera_info', CameraInfo)
    latlon_sub = message_filters.Subscriber(self.robot_name+'/navigator/navigation', NavSts)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub,latlon_sub], queue_size=10, slop=0.1)
    ts.registerCallback(self.cb_image)

    self.inf_image_pub = rospy.Publisher('inf_image', Image, queue_size=10)

	  # Set publishers
    self.pub_OD_predictions = rospy.Publisher(self.detected_class+"_predictions", yolov8_BB_latlon, queue_size=4)

  	# CvBridge for image conversion
    self.bridge = CvBridge()

  def cb_image(self, image, info,navstatus):
    # Callback for synchronized image and camera info and navstatus
    print("Processing a new image!!")
    if self.inference_finished:
      self.image = image
      self.info = info

      self.width = image.width
      self.height = image.height
      self.img_data = image.data
      self.lat=navstatus.global_position.latitude
      self.lat=navstatus.global_position.latitude
      self.run()
      time.sleep(self.period)
    else:
      print("inference not finished, just skipping this image!")


  def run(self):
    print("Sigue nadandoooo!")
    print("Looking for",self.detected_class," !")

    print("NEW IMAGE!! :)")

    self.inference_finished=False

    if not self.init:
      rospy.loginfo('[%s]: Start image segmentation', self.name)

      #Instance segmentation
      # Convert the raw image data to a NumPy array
      img_array = np.frombuffer(self.img_data, dtype=np.uint8)
      # Reshape the NumPy array to the image dimensions
      # img_array = img_array.reshape((self.height, self.width))  # RAW image
      img_array = img_array.reshape((self.height, self.width,3))  # COLOR image
      # Convert BGR to RGB
      img_array = img_array[..., ::-1]

      # Create a PIL Image from the NumPy array
      # pil_image = PIL.Image.fromarray(img_array, mode='L') # RAW IMAGE
      pil_image = PIL.Image.fromarray(img_array,mode='RGB') # COLOR IMAGE
      #inference
      results=self.model.predict(pil_image, save=True,save_conf=True,project=self.project_path,name=self.project_name,exist_ok=True ,conf=self.confidenceThreshold)
      # print("RESULTS, ",results)

      detected_boxes=results[0].boxes
      # print("pred!",results[0])

      #dictionary with model_classes:
      model_classes=results[0].names
      print(model_classes)
      if detected_boxes is not None:
        number_of_detections=len(detected_boxes)
        print(f"I found {number_of_detections} urchins !!!")
        self.detections.num_detections=number_of_detections
        self.detections.dets=[]

        try:
          for i in range(number_of_detections):
            object_cls=model_classes[int(detected_boxes.cls[i])]
            detection_conf=float(detected_boxes.conf[i])
            print("This object is an ",object_cls," with a confidence of a ", detection_conf)
            fish_bb=detected_boxes[i].xywh.numpy().flatten()

            fish_bb=[float(item) for item in fish_bb]
            print("fish_bb ",fish_bb)
            print("fish_bb type",type(fish_bb))
            self.det.bbox=fish_bb
            self.det.confidence=detection_conf
            self.det.object_class=object_cls
            self.det.lat=self.lat
            self.det.lon=self.lon

            self.detections.dets.append(self.det)
          print("-----------------------------------------------------------------")
        except Exception as e:
          self.inference_finished=True
          rospy.logwarn('[%s]: An exception happened!', e)

        res_plotted = results[0].plot()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_name = f"image_{timestamp}.jpg"
        self.detections.imageName=image_name
        self.detections.original_image=self.image
        self.detections.camera_info=self.info
        if detection_conf>0.9:
          cv2.imwrite(os.path.join(self.out_folder,image_name), res_plotted)
          print("Image saved to ",os.path.join(self.out_folder,image_name))

        # Convert the OpenCV image to a ROS Image message
        # Create a ROS Image message
        ros_infered_image = Image()
        ros_infered_image.header = Header()
        ros_infered_image.header.stamp = rospy.Time.now()
        ros_infered_image.height = self.height
        ros_infered_image.width = self.width
        ros_infered_image.encoding = "rgb8"  # Adjust the encoding based on your image format
        # ros_infered_image.is_bigendian = False
        ros_infered_image.step = 3 * self.width  # Assuming 3 channels (RGB)

        # Flatten the NumPy array and assign it to the data field of the ROS Image message
        ros_infered_image.data = res_plotted.flatten().tobytes()
        self.detections.infered_image=ros_infered_image

        self.inf_image_pub.publish(ros_infered_image)
        self.pub_OD_predictions.publish(self.detections)
        self.inference_finished=True


      else:
        print("NOTHING FOUND :(")

if __name__ == '__main__':
  try:
    rospy.init_node('object_detection')
    Object_detection(rospy.get_name())
    rospy.spin()
    cv2.destroyAllWindows()
  except rospy.ROSInterruptException:
    pass
