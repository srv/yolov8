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
import imageio.v2 as imageio
import message_filters
from sea_cucumber_detection.msg import det
from sea_cucumber_detection.msg import holo_detections
FRAME_SKIP=1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from cv_bridge import CvBridge, CvBridgeError
import PIL
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

    image_sub = message_filters.Subscriber('image_rect_color', Image)
    info_sub = message_filters.Subscriber('camera_info', CameraInfo)

    image_sub.registerCallback(self.cb_image)
    info_sub.registerCallback(self.cb_info)

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
    self.width = image.width
    self.height = image.height
    self.img_data = image.data


  def cb_info(self, info):
    #print(f"Call Back camera info")
    self.info = info
    time.sleep(self.period)
    self.run()



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
    self.model = torch.hub.load('/home/plomenuc/yolov5', 'custom', path=self.pathWeights, source='local',force_reload = False) # --> load local model throught the
    self.model.to(DEVICE).eval()
    matplotlib.use(theBackend)
    print("model succesfully load!")

  def run(self):
    rospy.loginfo('[%s]: Running', self.name)
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
    # image_np = np.array(np.frombuffer(image.data, dtype=np.uint8).reshape(2048, 1536))

    img_array = np.frombuffer(self.img_data, dtype=np.uint8)
    # Reshape the NumPy array to the image dimensions
    # img_array = img_array.reshape((self.height, self.width))  # RAW image
    img_array = img_array.reshape((self.height, self.width,3))  # COLOR image
    # Convert BGR to RGB
    img_array = img_array[..., ::-1]
    # Create a PIL Image from the NumPy array
    # pil_image = PIL.Image.fromarray(img_array, mode='L') # RAW IMAGE
    pil_image = PIL.Image.fromarray(img_array,mode='RGB') # COLOR IMAGE

    inference = self.model([pil_image],conf=self.confidenceThreshold)
    lblImage=inference.render()
    pred=inference.pandas().xyxy[0] # get the list of occurrences
    print(pred)
    number_of_rows=len(pred.index)
    print(f"Number of before reading from disk: Bounding Boxes {number_of_rows}")

    #Check image:
    # file_path = os.path.join("/home/plomenuc/plome_ws/src/stereo_plome/images", 'saved_image.png')
    # pil_image.save(file_path)
    # image_np = imageio.imread("/home/plomenuc/plome_ws/src/stereo_plome/images/saved_image.png")
    # inference = self.model([image_np])
    # print("POST READING FROM DISK")
    # Plot the labeled image

    lblImage=inference.render()
    pred=inference.pandas().xyxy[0] # get the list of occurrences
    print(pred)
    number_of_rows=len(pred.index)
    print(f"Number of Bounding Boxes {number_of_rows}")
    for index, row in pred.iterrows():

      if float(row['confidence']) > float(self.confidenceThreshold):
        print("I FOUND SOMETHING IN THIS IMAGE!!!")

        confidence = row['confidence']
        xmin=row['xmin']
        ymin=row['ymin']
        xmax=row['xmax']
        ymax=row['ymax']


        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_name = f"image_{timestamp}.jpg"
        cv2.imwrite(self.writeFileResults+image_name,lblImage[0]) # just to see the labeled image

        f= open(self.writeFileResults+"inferences.txt","a+")
        #f.write("%s %s\r\n" % image_name % confidence)
        f.write(f"{image_name} {confidence} \r\n")
        f.close()
        # print(f"Trained Model run over the image nbr {self.frameCounter}")

        # list of bounding boxes appended to the image detection message
        # self.holo_detections_out.dets = []
        # self.det.y1 = int(ymin)
        # self.det.x1 = int(xmin)
        # self.det.y2 = int(ymax)
        # self.det.x2 = int(xmax)
        # self.det.score = float(confidence)
        # self.det.object_class = str(row['class'])


        print(f"Detection message published")

if __name__ == '__main__':
  try:
    rospy.init_node('object_detection')
    Object_detection(rospy.get_name())
    rospy.spin()
    cv2.destroyAllWindows()
  except rospy.ROSInterruptException:
    pass
