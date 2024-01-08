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
from stereo_plome.msg import det
from stereo_plome.msg import mask_point
from stereo_plome.msg import yolov8_IS
FRAME_SKIP=1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from cv_bridge import CvBridge, CvBridgeError

from math import pi,tan
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header, String
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
    self.new_image = False

    self.fish_detections = yolov8_IS()
    self.det = det()

    #deg to rad
    self.FOV_x=34.73*((2*pi)/360.0)
    self.FOV_y=26.84*((2*pi)/360.0)
    # self.listener = tf.TransformListener()

	  # Params
    self.model_path = rospy.get_param("pathWeights")
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
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], queue_size=10, slop=0.1)
    ts.registerCallback(self.cb_image)

    self.inf_image_pub = rospy.Publisher('IS_image', Image, queue_size=10)

	  # Set publishers
    self.pub_fish_predictions = rospy.Publisher("fish_predictions", yolov8_IS, queue_size=4)

    # Set classification timer
    # rospy.Timer(rospy.Duration(self.period), self.run)

	  # CvBridge for image conversion
    self.bridge = CvBridge()

  def cb_image(self, image, info):
    # Callback for synchronized image and camera info
    if self.inference_finished:
      self.image = image
      self.info = info
      self.new_image = True
      self.width = image.width
      self.height = image.height
      self.img_data = image.data
      self.run()
      time.sleep(self.period)
    else:
        pass


  def run(self):
    # rospy.loginfo('[%s]: Running', self.name)
    print("Buscando a Nemo")

    if not self.new_image:
      print("NO images found")
      return
    self.new_image = False
    self.inference_finished=False
    # try:
    image = self.image
    info=self.info
    header = self.image.header
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
      # Process results generator

      # print("pred!",results[0])
      fish_masks=results[0].masks
      fish_boxes=results[0].boxes

      #dictionary with model_classes:
      fish_names=results[0].names
      print(fish_names)
      if fish_masks is not None:
        number_of_fish=len(fish_masks)
        print(f"I found {number_of_fish} fishes !!!")
        self.fish_detections.num_detections=number_of_fish
        fish_classes=[]
        self.fish_detections.dets=[]
        try:
          for i in range(number_of_fish):
            fish_cls=fish_names[int(fish_boxes.cls[i])]
            fish_conf=float(fish_boxes.conf[i])
            print("This fish is a ",fish_cls," with a confidence of a ", fish_conf)
            fish_bb=fish_boxes[i].xywh.numpy().flatten()
            fish_mask=np.array(fish_masks[i].xy)#.flatten()#.numpy().flatten()
            fish_mask_msg = []

            for segment in fish_mask:
              segment_array = np.array(segment[0], dtype=np.float32)
              print(segment_array)

              mask_point_msg=mask_point()
              mask_point_msg.x=float(segment_array[0])
              mask_point_msg.y=float(segment_array[1])

              fish_mask_msg.append(mask_point_msg)


            fish_bb=[float(item) for item in fish_bb]
            # print("fish_mask ",fish_mask)
            print("fish_bb type",type(fish_mask))

            self.det.mask=fish_mask_msg
            self.det.bbox=fish_bb
            self.det.confidence=fish_conf
            self.det.object_class=fish_cls

            self.fish_detections.dets.append(self.det)
          # print("Bounding mask!!: ",fish_mask)
          print("-----------------------------------------------------------------")
        except Exception as e:
          self.inference_finished=True
          rospy.logwarn('[%s]: An exception happened!', e)
          print("WRONG FISH MASK: ",fish_mask)


        res_plotted = results[0].plot()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_name = f"image_{timestamp}.jpg"
        self.fish_detections.imageName=image_name
        self.fish_detections.original_image=self.image
        self.fish_detections.camera_info=self.info
        if fish_conf>0.9:
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
        self.fish_detections.infered_image=ros_infered_image

        self.inf_image_pub.publish(ros_infered_image)
        self.pub_fish_predictions.publish(self.fish_detections)
        self.inference_finished=True


      else:
        print("I FIND NOTHING")
    # except Exception as e:
    #   print("EXCEPTION HAPPENED: ",e)
    #   rospy.logwarn('[%s]: There is no input image to run the segmentation', self.name)
    #   return e



if __name__ == '__main__':
  try:
    rospy.init_node('object_detection')
    Object_detection(rospy.get_name())
    rospy.spin()
    cv2.destroyAllWindows()
  except rospy.ROSInterruptException:
    pass
