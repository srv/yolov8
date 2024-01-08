cmake_minimum_required(VERSION 3.0.2)
project(yolov8_inference)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  rosbridge_server
  std_srvs
  message_generation
  rosbridge_server
  sensor_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  det_bb.msg
  yolov8_BB_latlon.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  RecordRequest.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)