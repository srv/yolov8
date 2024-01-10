#!/usr/bin/env python

import roslib
import rospy
import os
import shutil #for file management, copy file
import rosbag, sys, csv
from geometry_msgs.msg import Point
from sensor_msgs.msg import NavSatFix
import nav_msgs
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from cola2_msgs.msg import NavSts
import datetime
import csv
import subprocess
import shutil

import cv2
import matplotlib
import numpy
import message_filters
import numpy as np
from cv_bridge import CvBridge
from math import pi,tan
import tf
import geometry_msgs.msg
from cola2_lib.utils.ned import NED
from osgeo import gdal
import osr
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header, String
from shapely.geometry import Polygon
from yolov8_inference.msg import det_bb
from yolov8_inference.msg import yolov8_BB_latlon

class image_georeferencer:

    def __init__(self, name):
        """ Init the class """

        # Initialize some parameters
        self.overlap_threshold=rospy.get_param('~overlap_thrshld', default=0.5)

        scaling = rospy.get_param('~scaling', default=1)
        if scaling==1:
            img_topic="/stereo_down/left/image_color"
        if scaling==2:
            img_topic="/stereo_down/scaled_x2/left/image_color"
        if scaling==8:
            img_topic="/stereo_down/scaled_x8/left/image_color"

        dir = rospy.get_param('~saving_dir', default='/home/tintin/simulation_ws/src/yolov8_inference')
        self.dir_save_bagfiles = dir + 'CSV_TESTS/'

        #create directory
        if not os.path.exists(self.dir_save_bagfiles):
            os.makedirs(self.dir_save_bagfiles)

        self.image_secs=None
        self.counter=0
        self.skip_n_imgs=1    #save img every skip_n_imgs
        self.rows=0
        self.listener = tf.TransformListener()
        self.bridge = CvBridge()

        # FOV de la camera extret de excel de framerate i exposicio -> modificar

        FOV_x_water=rospy.get_param('~FOV_x_water',default=34.73)
        FOV_y_water=rospy.get_param('~FOV_y_water',default=26.84)

        print("Georeferencer started!! FOV_X_water, FOV_Y_water: ",FOV_x_water, FOV_y_water)

        #deg to rad
        self.FOV_x=34.73*((2*pi)/360.0)
        self.FOV_y=26.84*((2*pi)/360.0)

        #Subscribers
        self.bbox_sub=rospy.Subscriber("/urchin_predictions", yolov8_BB_latlon, self.urchin_callback)
        self.latlon_sub = rospy.Subscriber('/robot0/navigator/navigation', NavSts,self.latlon_sub)

    def latlon_sub(self,gps_sub):
        self.ned_origin_lat=gps_sub.origin.latitude
        self.ned_origin_lon=gps_sub.origin.longitude
        self.ned = NED(self.ned_origin_lat, self.ned_origin_lon, 0.0)
        #ned origin never changes we just want to listen to it once
        self.latlon_sub.unregister()

    #navigation and image callback
    #gets image and localization info and saves it to a csv
    def urchin_callback(self,urchin_bb):

        print("URCHIN FOUND!! REFERENCING IT TO WORLD!")
        self.altitude = urchin_bb.alt
        self.image_width=urchin_bb.original_image.width
        self.image_height=urchin_bb.original_image.height
        self.image_name=urchin_bb.imageName

        self.cv_image = self.bridge.imgmsg_to_cv2(urchin_bb.infered_image, desired_encoding="bgr8")

        #Image pixel dimensions (tan function uses rad)
        self.pixel_x_dim_img = (2*self.altitude*tan(self.FOV_x/2))/self.image_width #width
        self.pixel_y_dim_img = (2*self.altitude*tan(self.FOV_y/2))/self.image_height #height
        print("IMG_DIM: ",self.pixel_x_dim_img," y_:",self.pixel_y_dim_img)


        self.urchins_in_image=[]
        self.urchins_in_image_latlon=[]

        for urchin_det in urchin_bb.dets:
            #det_bb x,y,w,h
            x,y,w,h=urchin_det.bbox[0],urchin_det.bbox[1],urchin_det.bbox[2],urchin_det.bbox[3]
            self.latitude = urchin_det.lat
            self.longitude = urchin_det.lon

            pose_corner_r_up = PoseStamped()
            pose_corner_l_up = PoseStamped()
            pose_corner_r_down = PoseStamped()
            pose_corner_l_down = PoseStamped()

            #Corners of img referenced to img_center (restar la mitad porque yolo referencia a la esquina superior izq crec)
            #Corners of img referenced to img_center:
            pose_corner_r_up.pose.position.x= ((-self.image_width/2) +(x+w/2))*self.pixel_x_dim_img
            pose_corner_r_up.pose.position.y= ((-self.image_height/2 )+(y+h/2))*self.pixel_y_dim_img

            pose_corner_l_up.pose.position.x= ((-self.image_width/2) +(x-w/2))*self.pixel_x_dim_img
            pose_corner_l_up.pose.position.y=  ((-self.image_height/2 )+(y+h/2))*self.pixel_y_dim_img

            pose_corner_r_down.pose.position.x=  ((-self.image_width/2) +(x+w/2))*self.pixel_x_dim_img
            pose_corner_r_down.pose.position.y= ((-self.image_height/2 )+(y-h/2))*self.pixel_y_dim_img

            pose_corner_l_down.pose.position.x= ((-self.image_width/2) +(x-w/2))*self.pixel_x_dim_img
            pose_corner_l_down.pose.position.y= ((-self.image_height/2 )+(y-h/2))*self.pixel_y_dim_img

            corner_list=[pose_corner_l_up, pose_corner_r_up, pose_corner_r_down,pose_corner_l_down]

            self.bb_corners=[] #
            self.bb_corners_latlon=[]
            #Convert corner's pose to a pose referenced to world
            for corner in corner_list:
                corner.header=Header(stamp=urchin_bb.header.stamp, frame_id='/robot0/stereo_down/left_optical')
                corner.pose.position.z= 0   #self.altitude
                corner.pose.orientation.x = 0
                corner.pose.orientation.y = 0
                corner.pose.orientation.z = 0
                corner.pose.orientation.w = 0
                corner_transformed=self.listener.transformPose("/world_ned", corner)
                self.bb_corners.append(corner_transformed) #
                print("CORNERS")
                print(corner_transformed.pose)
                point_lat,point_lon,_ =self.ned.ned2geodetic([corner_transformed.pose.position.x, corner_transformed.pose.position.y, 0.0])
                print("LAT: ",point_lat," LON ",point_lon)
                self.bb_corners_latlon.append([point_lat,point_lon])
                self.export_to_csv(self.image_name, point_lat, point_lon,self.altitude,corner_transformed.pose.position.x,corner_transformed.pose.position.y)

            self.urchins_in_image.append(self.bb_corners)
            self.urchins_in_image_latlon.append(self.bb_corners_latlon)

        filename=self.image_name.split(".")[0]+".tiff"
        if self.counter==0:
            self.prev_img_urchins=self.urchins_in_image[:]

            # lat lon pxl right up and pixel left down
            # bounds=[pxl_lat_up, pxl_lon_up,pxl_lat_down, pxl_lon_down]
            #Save the first image urchins
            for urchin_latlon in self.urchins_in_image_latlon:
                bounds=[urchin_latlon[1][0],urchin_latlon[1][1],urchin_latlon[3][0],urchin_latlon[3][1]]
                lat_diff = abs(urchin_latlon[3][0] - urchin_latlon[1][0])
                lon_diff = abs(urchin_latlon[3][1] - urchin_latlon[1][1])
                self.geotiff(self,filename, self.cv_image,bounds,lat_diff,lon_diff)

        else:
            for urchin,urchinlatlon in zip(self.urchins_in_image,self.urchins_in_image_latlon):
                for prev_urchin in self.prev_img_urchins:
                    #the 4 corners in global pose xy
                    is_overlaping,overlap = self.get_overlapping(urchin[0:-1],prev_urchin[0:-1])
                    print("OVERLAP: ",overlap)
                    #The urchins are different so we save them
                    if is_overlaping==False:
                        # lat lon pxl right up and pixel left down (bounds=[pxl_lat_up, pxl_lon_up,pxl_lat_down, pxl_lon_down] )
                        bounds=[urchinlatlon[1][0],urchinlatlon[1][1],urchinlatlon[3][0],urchinlatlon[3][1]]
                        lat_diff = abs(urchinlatlon[3][0] - urchinlatlon[1][0])
                        lon_diff = abs(urchinlatlon[3][1] - urchinlatlon[1][1])
                        self.geotiff(self,filename, self.cv_image,bounds,lat_diff,lon_diff)
                        #guardar los 4
                        for lat_lon,xy in zip(urchin,urchinlatlon):
                            self.export_to_csv(self.image_name, lat_lon[0], lat_lon[1],self.altitude,xy.pose.position.x,xy.pose.position.y,"_not_repeated_urchins.csv")



    def export_to_csv(self, seq, lat, long,z,world_x,world_y,csv_name='_detected_urchins_bboxes.csv'):
        print("WRITING_CSV")
        header=["img_name","latitude","longitude","altitude","global_x","global_y"]

        csv_file=os.path.join(self.dir_save_bagfiles+csv_name)

        data_list = [seq,lat,long,z,world_x,world_y]
        print("witing data: ",data_list,"in file: ",csv_file)

        with open(csv_file, 'a+') as file:
            writer = csv.writer(file, delimiter=';')
            if self.rows==0:
                writer.writerow(header)

            writer.writerow(data_list)
            self.rows+=1
            print("rows: ",self.rows)

        file.close()

    # adfGeoTransform[0] /* top left x */
    # adfGeoTransform[1] /* w-e pixel resolution */ ->xres lon
    # adfGeoTransform[2] /* 0 */
    # adfGeoTransform[3] /* top left y */
    # adfGeoTransform[4] /* 0 */
    # adfGeoTransform[5] /* n-s pixel resolution (negative value) */ ->yres lat

    def geotiff(self,filename, image,bounds,lat_diff,lon_diff):

        print("filename: ",filename)
        ny=image.shape[0] #img height 1440
        nx=image.shape[1] #img width 1920
        xres = lon_diff / float(nx)
        yres = lat_diff / float(ny)

        geotransform = (bounds[0], xres, 0, bounds[1], 0, yres)
        # To work in qgis(lat lon must be lon lat):
        # geotransform = (bounds[1], xres, 0, bounds[0], 0, yres)

        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(4326)                # WGS84 lat/long

        if len(image.shape) == 3:
            dst_ds = gdal.GetDriverByName('GTiff').Create(filename, nx, ny, 3, gdal.GDT_Byte)
            # dst_ds = gdal.GetDriverByName('GTiff').Create(filename, ny, nx, 3, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference()            # establish encoding
            srs.ImportFromEPSG(4326)                # WGS84 lat/long
            dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(image[:, :, 0])   # write r-band to the raster
            dst_ds.GetRasterBand(2).WriteArray(image[:, :, 1])   # write g-band to the raster
            dst_ds.GetRasterBand(3).WriteArray(image[:, :, 2])   # write b-band to the raster
        dst_ds.FlushCache()                     # write to disk
        dst_ds = None

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


if __name__ == '__main__':
    try:
        rospy.init_node('image_georeferencer')
        image_georeferencer = image_georeferencer(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass