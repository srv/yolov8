# Custom ROS message for YOLOv8 instance segmentation results

Header header
sensor_msgs/Image infered_image
sensor_msgs/CameraInfo camera_info
string imageName #name of the stored image

int64 num_detections

#Detections
det[] dets #Detected instances (bounding boxs and mask with corresponding score and class)

# Original image
sensor_msgs/Image original_image
