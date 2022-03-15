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

