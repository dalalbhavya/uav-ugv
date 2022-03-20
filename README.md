# UAV guided UGV
To run the world make sure to include the path in $GAZEBO_MODEL_PATH environment variable

```src/InterIIT_DRDO/interiit22/models```

by running the command in terminal
```
echo $GAZEBO_MODEL_PATH
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:<your path>
```

To install PX4-Autopilot use the commands

```
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
```

ROS Topics not publishing for depth camera
Solved by copying the links a new world file ```drdo_world1_cam.world```

P.S. : This isssue is yet to be resolved in a more general way

To perform a general testing of the control code of drone follow
1. Open a new terminal and change the directory to ```~/catkin_ws/src/iq_sim/scripts``` and run
```
./startsitl.sh
```

2. Now open a new terminal and run
```
roslaunch iq_sim apm.launch
```

3. In a new terminal run
```
rosrun iq_gnc drdo_test
```

4. Now make the mode GUIDED by running the command
```
mode GUIDED
```


