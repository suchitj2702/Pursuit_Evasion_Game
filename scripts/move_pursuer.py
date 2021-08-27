import rospy
import string
import math
import time
import sys

from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseActionResult
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped


class MultiGoals:
    def __init__(self, goalListX, goalListY, retry, map_frame):
        self.ns = rospy.get_namespace()
        topic_1 = '/tb3_0/move_base/result'
        topic_2 = self.ns + '/tb3_0/move_base_simple/goal'
        self.sub = rospy.Subscriber(topic_1, MoveBaseActionResult, self.statusCB, queue_size=10)
        self.pub = rospy.Publisher(topic_2, PoseStamped, queue_size=10)
        # params & variables
        self.goalListX = goalListX
        self.goalListY = goalListY
        self.retry = retry
        self.goalId = 0
        self.goalMsg = PoseStamped()
        self.goalMsg.header.frame_id = map_frame
        self.goalMsg.pose.orientation.z = 0.0
        self.goalMsg.pose.orientation.w = 1.0
        # Publish the first goal
        time.sleep(1)
        self.goalMsg.header.stamp = rospy.Time.now()
        self.goalMsg.pose.position.x = self.goalListX[self.goalId]
        self.goalMsg.pose.position.y = self.goalListY[self.goalId]
        self.pub.publish(self.goalMsg)
        rospy.loginfo("Initial goal published! Goal ID is: %d", self.goalId)
        self.goalId = self.goalId + 1

    def statusCB(self, data):
        if data.status.status == 3:  # reached
            self.goalMsg.header.stamp = rospy.Time.now()
            self.goalMsg.pose.position.x = self.goalListX[self.goalId]
            self.goalMsg.pose.position.y = self.goalListY[self.goalId]
            self.pub.publish(self.goalMsg)
            rospy.loginfo("Reached: Goal ID is: %d", self.goalId)
            if self.goalId < (len(self.goalListX) - 1):
                self.goalId = self.goalId + 1
            else:
                self.goalId = 0


if __name__ == "__main__":
    try:
        env = 0
        rospy.init_node('move_pursuer', anonymous=True)
        if env ==  0:
            goalListX = [2.69, 4.68, 3.6, 1.13, -0.83, -2.47, -5.49, -6.28, -4.24]
            goalListY = [1.34, -1.5, -3.76, -4.48, -2.86, 0.76, 1.85, -1.19, -3.73]
        else:
            goalListX = [2.71, 2.78, 0, 5.69, 8.22, 8.34]
            goalListY = [-1.54, -3.67, -4.09, -1.87, -0.27, -4.78]

        # Get params
        map_frame = "map"
        retry = 1

        print(goalListX)
        print(goalListY)
        if len(goalListX) == len(goalListY) & len(goalListY) >= 2:
            # Constract MultiGoals Obj
            rospy.loginfo("Multi Goals Executing...")
            mg = MultiGoals(goalListX, goalListY, retry, map_frame)
            rospy.spin()
        else:
            rospy.errinfo("Lengths of goal lists are not the same")
    except KeyboardInterrupt:
    	print("shutting down")
