#!/usr/bin/env python
from __future__ import print_function
import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
from detect_track import tracker
 
class image_converter: 
    def __init__(self):
        self.image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback)
        
    def callback(self, image_data):
        try:
            cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        except Exception as e:
            print(e)

        inferred_image, center_points = tracker(cv_image)
            
        cv2.imshow("Image window", inferred_image)
        cv2.waitKey(1)
      
def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)