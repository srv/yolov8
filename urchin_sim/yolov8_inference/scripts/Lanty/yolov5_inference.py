#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import torch
from skimage.io import imread
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib
import cv2
import time
import sys
print(sys.path)
import numpy as np
import cv2
import copy
import rospy
import sensor_msgs.msg
from sensor_msgs.msg import Image, CameraInfo

import message_filters
from sea_cucumber_detection.msg import det
from sea_cucumber_detection.msg import holo_detections
FRAME_SKIP=1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from cv_bridge import CvBridge, CvBridgeError

from math import pi,tan
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header, String
# from shapely.geometry import Polygon
# import tf
# import tf2_ros
# import tf2_geometry_msgs

class Object_detection:

  def __init__(self, name): # clase constructor

    self.name = name
    # Params
    self.init = False
    self.new_image = False

    self.holo_detections_out = holo_detections()
    self.det = det()
    #deg to rad
    self.FOV_x=34.73*((2*pi)/360.0)
    self.FOV_y=26.84*((2*pi)/360.0)
    # self.listener = tf.TransformListener()

	  # Params
    self.pathWeights = rospy.get_param("pathWeights")
    self.confidenceThreshold = rospy.get_param("confidenceThreshold")
    self.writeFileResults= rospy.get_param("writeFileResults")
    self.robot_name = rospy.get_param('robot_name')
    self.period = rospy.get_param("period")

    # Set subscribers and Callback
    # print(f"params: {self.pathWeights} {self.confidenceThreshold} {self.writeFileResults}")

    image_sub = message_filters.Subscriber('image_rect_color', Image)
    info_sub = message_filters.Subscriber('camera_info', CameraInfo)
    latlon_sub = message_filters.Subscriber('/'+str(self.robot_name)+'/navigator/navigation', NavSts)

    image_sub.registerCallback(self.cb_image)
    info_sub.registerCallback(self.cb_info)
    latlon_sub.registerCallback(self.cb_latlon)

	  # Set publishers
    self.pub_holo_detection = rospy.Publisher("holo_detections", holo_detections, queue_size=4)

    # Set classification timer
    rospy.Timer(rospy.Duration(self.period), self.run)

	  # CvBridge for image conversion
    self.bridge = CvBridge()

  def cb_image(self, image):
    # print(f"Call Back image")
    self.image = image
    self.new_image = True


  def cb_info(self, info):
    #print(f"Call Back camera info")
    self.info = info
    time.sleep(self.period)
    self.run()

  def cb_latlon(self,latlon):
    self.latlon = latlon

  def get_param(self, param_name, default = None):
        if rospy.has_param(param_name):
            param_value = rospy.get_param(param_name)
            return param_value
        elif default is not None:
            return default
        else:
            rospy.logfatal('[%s]: invalid parameters for %s in param server!', self.name, param_name)
            rospy.logfatal('[%s]: shutdown due to invalid config parameters!', self.name)
            exit(0)

  def set_model(self):
    theBackend=plt.get_backend()
    self.model = torch.hub.load('/home/tintin/yolov5', 'custom', path=self.pathWeights, source='local',force_reload = False) # --> load local model throught the
    self.model.to(DEVICE).eval()
    matplotlib.use(theBackend)
    # print("model succesfully load!")

  def get_overlapping(self,img_bounds1,img_bounds2):
    #Calculates the overlap ratio between 2 images
    # img_bounds1: poses of corners of img1 (left up,right up,right down,left down)
    # img_bounds2: poses of corners of img2 (left up,right up,right down,left down)
    # returns [overlap,overlap_proportion]
        # overlap: True if there is more overlap than the threshold, false otherwise
        # overlap proportion: overlap ratio of the 2 images

    bounds1=[]
    bounds2=[]
    # convert to (x,y) points
    for bound1,bound2 in zip(img_bounds1,img_bounds2):
        bounds1.append([bound1.pose.position.x,bound1.pose.position.y])
        bounds2.append([bound2.pose.position.x,bound2.pose.position.y])

    img1=Polygon([bounds1[0],bounds1[1],bounds1[2],bounds1[3]])
    img2=Polygon([bounds2[0],bounds2[1],bounds2[2],bounds2[3]])

    if img1.intersects(img2):
        overlap_area=img1.intersection(img2).area
        if img1.area <= img2.area:
            overlap_proportion=overlap_area/img1.area
        else:
            overlap_proportion=overlap_area/img2.area
        if overlap_proportion >= self.overlap_threshold:
            return True,overlap_proportion
        else:
            return False,None
    else:
        return False,None




  def run(self):
    # rospy.loginfo('[%s]: Running', self.name)
    if not self.new_image:
      return
    self.new_image = False
    try:
      image = self.image
      info=self.info
      header = self.image.header
      if not self.init:
        rospy.loginfo('[%s]: Start image segmentation', self.name)
    except:
      rospy.logwarn('[%s]: There is no input image to run the segmentation', self.name)
      return

	  # Set model
    if not self.init:
      self.set_model()
      self.init = True



	  # Object detection
    image_np = np.array(np.frombuffer(image.data, dtype=np.uint8).reshape(360, 480,3))
    image_byte=img_as_ubyte(image_np) # converts into unsigned by format [0...255]
    inference = self.model([image_byte])

    self.frameCounter = self.frameCounter + 1

    # Plot the labeled image
    lblImage=inference.render()
    pred=inference.pandas().xyxy[0] # get the list of occurrences
    number_of_rows=len(pred.index)
    # print(f"Number of Bounding Boxes {number_of_rows}")
    for index, row in pred.iterrows():
      if str(row['class']) == "0":
        if float(row['confidence']) > float(self.confidenceThreshold):
          # print("HIT")
          confidence = row['confidence']
          xmin=row['xmin']
          ymin=row['ymin']
          xmax=row['xmax']
          ymax=row['ymax']

          image_name=str(self.frameCounter)+'.jpg'
          cv2.imwrite(self.writeFileResults+image_name,lblImage[0]) # just to see the labeled image

          f= open(self.writeFileResults+"inferences.txt","a+")
          #f.write("%s %s\r\n" % image_name % confidence)
          f.write(f"{image_name} {confidence} \r\n")
          f.close()
          # print(f"Trained Model run over the image nbr {self.frameCounter}")

          # list of bounding boxes appended to the image detection message
          self.holo_detections_out.dets = []
          self.det.y1 = int(ymin)
          self.det.x1 = int(xmin)
          self.det.y2 = int(ymax)
          self.det.x2 = int(xmax)
          self.det.score = float(confidence)
          self.det.object_class = str(row['class'])
          det2 = copy.deepcopy(self.det)
          self.holo_detections_out.dets.append(det2)
          self.holo_detections_out.header = header
          self.holo_detections_out.image_rect_color = image
          self.holo_detections_out.lat = self.latlon.global_position.latitude
          self.holo_detections_out.lon = self.latlon.global_position.longitude
          self.holo_detections_out.imageName=image_name
          self.holo_detections_out.camera_info = info
          self.holo_detections_out.num_detections = number_of_rows
          self.pub_holo_detection.publish(self.holo_detections_out)

          print(f"Detection message published")

if __name__ == '__main__':
  try:
    rospy.init_node('object_detection')
    Object_detection(rospy.get_name())
    rospy.spin()
    cv2.destroyAllWindows()
  except rospy.ROSInterruptException:
    pass
