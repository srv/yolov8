# Custom ROS message for YOLOv8 instance segmentation results

Header header
sensor_msgs/Image infered_image
string imageName #name of the stored image

int64 num_detections

#altitude
float32 alt

#Detections
det_bb[] dets #Detected instances (bounding boxs with corresponding score, class and lat lon)

# Original image
sensor_msgs/Image original_image
