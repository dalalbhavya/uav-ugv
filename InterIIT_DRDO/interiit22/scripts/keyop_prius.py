#!/usr/bin/env python
import rospy
import time
from prius_msgs.msg import Control


rospy.init_node('car_demo')
pub = rospy.Publisher('prius', Control, queue_size=20)
rate = rospy.Rate(2)
command = Control()
rospy.loginfo('start')
command_string = 'wwwwwwwwwwaaaawaaawddaawwddwwdf'

def letter_to_command(command,current_key):
    if(current_key == 'w'):
        command.shift_gears = Control.FORWARD
        command.throttle = 1.0
        command.brake =0.0
        command.steer= 0.0
        # pub.publish(command)
    if(current_key == 's'):
        command.shift_gears = Control.REVERSE
        command.throttle = 1.0
        command.brake =0.0
        command.steer= 0.0
        # pub.publish(command)
    if(current_key == 'a'):
        command.shift_gears = Control.FORWARD
        command.throttle = 0.5
        command.brake =0.0
        command.steer= 0.8
        # pub.publish(command)
    if(current_key == 'd'):
        command.shift_gears = Control.FORWARD
        command.throttle = 0.5
        command.brake =0.0
        command.steer= -0.8
        # pub.publish(command)
    if(current_key == 'f'):
        command.shift_gears = Control.NO_COMMAND
        command.throttle = 0.0
        command.brake =1.0
        command.steer= 0.0
        # pub.publish(command)
    return command    

count = 0
if not rospy.is_shutdown():
    for key in command_string:
        count+=1
        print("key ", key)
        command = letter_to_command(command,key)
        pub.publish(command)
        print("command ",count)
        rate.sleep()


# while not rospy.is_shutdown():

# 	current_key = 'w'
#     # str(input('use keyboard strokes to move car'))
# 	pub.publish(command)     
rospy.spin()
