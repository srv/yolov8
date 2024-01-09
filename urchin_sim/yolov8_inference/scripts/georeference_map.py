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

class scenario_georeferencer:

    def __init__(self, name):
        """ Init the class """

        # Initialize some parameters:

        self.center_lat=39.533687
        self.center_lon=2.590453

        #terrain looks image dimensions:
        self.image_width=2762
        self.image_heigh=3226

        #Stonefish Documentation: Scale of the terrain is defined in meters per pixel
        self.scalex=0.02 #m/pixel
        self.scaley=0.02 #m/pixel

        #scenario dimensions(m):
        self.scenario_xdim=self.scalex*self.image_width
        self.scenario_ydim=self.scaley*self.image_heigh

        self.image_path="/home/uib/yolov8/urchin_sim/scenario/super_texture_by_antonio.jpeg"

        self.ned = NED(self.center_lat, self.center_lon, 0.0)

    #navigation and image callback
    #gets image and localization info and saves it to a csv
    def georeference_scenario(self):

        center_x, center_y, _  = self.ned.geodetic2ned([self.center_lat,self.center_lon,0.0])

        up_left_x = center_x - (self.scenario_xdim/2)
        up_left_y= center_y +(self.scenario_ydim/2)

        # up_right_x = center_x + (self.scenario_xdim/2)
        # up_right_y= center_y +(self.scenario_ydim/2)

        down_right_x = center_x + (self.scenario_xdim/2)
        down_right_y= center_y -(self.scenario_ydim/2)

        # down_left_x = center_x - (self.scenario_xdim/2)
        # down_left_y= center_y -(self.scenario_ydim/2)

        #lat,lon diff ->used only to georeference img
        pxl_lat_up, pxl_lon_up, _ = self.ned.ned2geodetic([up_left_x, up_left_y, 0.0])

        pxl_lat_down, pxl_lon_down, _ = self.ned.ned2geodetic([down_right_x,down_right_y, 0.0])

        lat_diff = abs(pxl_lat_down - pxl_lat_up)
        lon_diff = abs(pxl_lon_down - pxl_lon_up)

        cv_image=cv2.imread(self.image_path)
        bounds=[pxl_lat_up, pxl_lon_up,pxl_lat_down, pxl_lon_down]

        #georeference:
        self.geotiff(self.image_path,cv_image,bounds,lat_diff,lon_diff)


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

        # geotransform = (bounds[0], xres, 0, bounds[1], 0, yres)
        # To work in qgis(lat lon must be lon lat):
        geotransform = (bounds[1], xres, 0, bounds[0], 0, yres)

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


if __name__ == '__main__':
    try:
        rospy.init_node('scenario_georeferencer')
        image_georeferencer = scenario_georeferencer(rospy.get_name())
        image_georeferencer.georeference_scenario()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass