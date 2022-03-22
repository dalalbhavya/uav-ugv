#!/usr/bin/env python
from sys import is_finalizing
import rospy
import numpy as np
import time
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

rospy.init_node('string_generator')
pub = rospy.Publisher('navigate_ugv', String, queue_size=20)

waypoints_x = []
waypoints_y = []
command_string = ""

def callback(msg):
    for i,j in msg :
        waypoints_x.append(i)
        waypoints_y.append(j)

sub = rospy.Subscriber('waypoint_Generator', Float64MultiArray , callback )

rospy.loginfo('start')

def cross_product(j):
    # P1 vector = waypoint[j+1] - waypoint[j]
    # P2 vector = waypoint[j+2] - waypoint[j]
    # cross_product = P1 x P2
    return ( ((waypoints_x[j] - waypoints_x[j+1])*(waypoints_y[j+2] - waypoints_y[j+1])) - ((waypoints_x[j+2] - waypoints_x[j+1])*(waypoints_y[j+1] - waypoints_y[j])) )

for i in range(len(waypoints_x)-3) :
        if cross_product(i) == 0:
            # straight path --> throttle 
            command_string += 'w'                
            
        if cross_product(i) > 0 :
            # left turn 
            command_string += 'a'
            
        if cross_product(i) < 0 :
            # right turn
            command_string += 'd'
            
        if i == (len(waypoints_x)-2) :
            # stop / brake 
            command_string += 'wwfffffff'

pub.publish(command_string)
  