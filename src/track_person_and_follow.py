#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
from detect_track import tracker
from geometry_msgs.msg import Twist

Pi = 3.1415926535897
 
def straight_line(velocity_publisher, vel_msg, speed, distance):
    vel_msg.linear.x = distance # we only specify x axis since the motion is linear
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0

    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0

    t0 = rospy.Time.now().to_sec()
    current_distance = 0

    while (current_distance < distance):
        velocity_publisher.publish(vel_msg) # publishing the message to move
        t1 = rospy.Time.now().to_sec()
        current_distance = speed*(t1 - t0) # because distance = speed*time

    vel_msg.linear.x = 0

    velocity_publisher.publish(vel_msg)

def rotate(velocity_publisher, vel_msg, clockwise, speed, angle):
    angular_speed = speed*2*Pi/360 # converting rotational speed to angular speed
    relative_angle = angle*2*Pi/360 # converting angle from degree to radians

    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0

    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    if clockwise: # Checking the orientation (clockwise or anti-clockwise)
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)

    t0 = rospy.Time.now().to_sec()
    
    current_angle = 0

    while(current_angle < relative_angle):
        velocity_publisher.publish(vel_msg) # publishing the message to rotate
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed*(t1-t0)

    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg) 

class image_converter: 
    def __init__(self):
        self.image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback)
        self.cmd_vel_pub = rospy.Publisher("/tb3_0/cmd_vel", Twist, queue_size = 1)

    def callback(self, image_data):
        try:
            cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        except Exception as e:
            print(e)

        inferred_image, center_points = tracker(cv_image)

        cv2.imshow("Image window", inferred_image)
        cv2.waitKey(1)

        move_cmd = Twist()
        if center_points != []:
            if center_points[0] > 640: # greater than center of bounding box
                prop = (center_points[0] - 640)/640
                rotate(self.cmd_vel_pub, move_cmd, clockwise = 1, speed = 1.8 + prop, angle = 1.8 + prop)
            else:
                prop = (640 - center_points[0])/640
                rotate(self.cmd_vel_pub, move_cmd, clockwise = 0, speed = 1.8 + prop, angle = 1.8 + prop)
            straight_line(self.cmd_vel_pub, move_cmd, 0.15, 0.15)
      
def main(args):
    ic = image_converter()
    rospy.init_node('tracker_and_follower', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)