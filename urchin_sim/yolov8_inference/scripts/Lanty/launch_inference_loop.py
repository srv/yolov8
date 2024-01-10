import os
#!/usr/bin/env python

import rospy
import rosnode

def check_node_existence(node_name):
    # Check if the node exists
    node_exists = rosnode.rosnode_ping(node_name)

    if node_exists:
        print(f"Node '{node_name}' exists and is reachable.")
    else:
        print(f"Node '{node_name}' does not exist or is not reachable.")

def main():
    # Initialize the ROS node
    rospy.init_node('node_checker', anonymous=True)

    # Specify the name of the node you want to check
    node_to_check = '/your/target/node'

    # Check if the node exists
    check_node_existence(node_to_check)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

# os.system("roslaunch stereo_plome ROS_inference_Plome.launch")