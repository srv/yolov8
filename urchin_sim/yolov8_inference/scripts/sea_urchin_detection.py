#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import torch

import matplotlib.pyplot as plt
import cv2
import time
import sys
print(sys.path)
import numpy as np
from math import pi,tan
import PIL

#ROS imports
import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from std_msgs.msg import Header
from cola2_msgs.msg import NavSts
# import tf
# import tf2_ros
# import tf2_geometry_msgs
#ros custom msgs:
from yolov8_inference.msg import det_bb
from yolov8_inference.msg import yolov8_BB_latlon

from cv_bridge import CvBridge

#yolov8:
from ultralytics import YOLO
import csv

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Object_detection:

  def __init__(self, name): # clase constructor

    self.name = name
    # Params
    self.init = False
    self.new_image = False

    self.detections = yolov8_BB_latlon()

    self.detections.alt=-1
    self.det = det_bb()
    self.urchin_images_seq=0
    self.rows=0

	  # Params
    self.model_path = rospy.get_param("pathWeights")
    self.confidenceThreshold = rospy.get_param("confidenceThreshold")
    self.writeFileResults= rospy.get_param("writeFileResults")
    self.robot_name = rospy.get_param('robot_name')
    self.period = rospy.get_param("period")
    self.project_path = rospy.get_param("project_path")
    self.project_name = rospy.get_param("project_name")
    self.dir_save_urchins=rospy.get_param("save_urchins_location_path",default="/home/tintin/yolov8/urchin_sim/found_urchins.csv")
    FOV_x_water=rospy.get_param('~FOV_x_water',default=34.73)
    FOV_y_water=rospy.get_param('~FOV_y_water',default=26.84)

    #deg to rad
    self.FOV_x=FOV_x_water*((2*pi)/360.0)
    self.FOV_y=FOV_y_water*((2*pi)/360.0)
    # self.listener = tf.TransformListener()

    self.out_folder=os.path.join(self.project_path,self.project_name)
    self.inference_finished=True

    #LOAD MODEL:
    self.model = YOLO(self.model_path)
    # image_sub = message_filters.Subscriber('image_rect_color', Image)
    image_sub = message_filters.Subscriber(self.robot_name+'/stereo_down/left/image_color', Image)
    info_sub = message_filters.Subscriber(self.robot_name+'/stereo_down/left/camera_info', CameraInfo)
    latlon_sub = message_filters.Subscriber(self.robot_name+'/navigator/navigation', NavSts)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub,latlon_sub], queue_size=100, slop=0.1)
    ts.registerCallback(self.cb_image)

    self.inf_image_pub = rospy.Publisher('IS_image', Image, queue_size=10)

	  # Set publishers
    self.pub_fish_predictions = rospy.Publisher("urchin_predictions", yolov8_BB_latlon, queue_size=4)


  def cb_image(self, image, info,navstatus):
    # Callback for synchronized image and camera info
    if self.inference_finished:
      self.image = image
      self.info = info
      self.new_image = True
      self.width = image.width
      self.height = image.height
      self.img_data = image.data
      self.lat=navstatus.global_position.latitude
      self.lon=navstatus.global_position.longitude
      self.alt=navstatus.altitude
      self.run()
      time.sleep(self.period)
    else:
        pass

  def export_to_csv(self,seq, num_dets, lat, lon,z):
    print("WRITING_CSV")
    header=["img_name","latitude","longitude","altitude","num_detections"]

    csv_file=os.path.join(self.dir_save_urchins)
    seq=str(seq)+".JPG"
    data_list = [seq,lat,lon,z,num_dets]
    print("witing data: ",data_list,"in file: ",csv_file)

    with open(csv_file, 'a+') as file:
        writer = csv.writer(file, delimiter=';')
        if self.rows==0:
            writer.writerow(header)

        writer.writerow(data_list)
        self.rows+=1
        print("rows: ",self.rows)

    file.close()



  def run(self):

    print("Looking for urchins!")
    print("Looking for urchins!")

    if not self.new_image:
      print("NO images found")
      return

    print("NEW IMAGE!!")
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
      # img_array = img_array[..., ::-1]

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
        if number_of_detections>0:
          print("I found ",number_of_detections,"urchins !!!")
          self.detections.num_detections=number_of_detections
          self.detections.dets=[]
          print("ALTITUDE: ", self.alt)
          self.detections.alt=self.alt

          res_plotted = results[0].plot()
          timestamp = time.strftime("%Y%m%d-%H%M%S")
          # image_name = "image_"+str(timestamp)+".jpg"
          image_name = "image_"+str(self.urchin_images_seq)+".jpg"
          self.urchin_images_seq=self.urchin_images_seq+1
          self.detections.imageName=image_name
          self.detections.original_image=self.image

          # try:
          for i in range(number_of_detections):
            object_cls=model_classes[int(detected_boxes.cls[i])]
            detection_conf=float(detected_boxes.conf[i])
            print("This object is a/an ",object_cls," with a confidence of a ", detection_conf)
            fish_bb=detected_boxes[i].xywh.numpy().flatten()

            fish_bb=[float(item) for item in fish_bb]
            print("fish bb: ",fish_bb)
            # print("fish_mask ",fish_mask)
            print("fish_bb type",type(fish_bb))
            self.det.bbox=fish_bb
            self.det.confidence=detection_conf
            self.det.object_class=object_cls
            self.det.lat=self.lat
            self.det.lon=self.lon
            self.detections.dets.append(self.det)

            self.export_to_csv(image_name,number_of_detections, self.lat, self.lon,self.detections.alt)
          # print("Bounding mask!!: ",fish_mask)
          print("-----------------------------------------------------------------")
          # except Exception as e:
          self.inference_finished=True

          # self.detections.camera_info=self.info
          if detection_conf>0.1:
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
          self.pub_fish_predictions.publish(self.detections)
          self.inference_finished=True
        else:
          print("num of detections is 0 :(")
          self.inference_finished=True

      else:
        print("I FIND NOTHING")
        self.inference_finished=True


if __name__ == '__main__':
  try:
    rospy.init_node('object_detection')
    Object_detection(rospy.get_name())
    rospy.spin()
    cv2.destroyAllWindows()
  except rospy.ROSInterruptException:
    pass
