# VisFly + Elastic Tracker Integration Guide

## Quick Start

### 1. Start VisFly (Terminal 1)
```bash
cd /home/lfx-desktop/files/obj_track/catkin_ws
source devel/setup.bash
roslaunch visfly visfly_only.launch
```

### 2. Start Elastic Tracker Planning (Terminal 2)  
```bash
cd /home/lfx-desktop/files/Replication/obj_track/Elastic-Tracker
source devel/setup.bash
rosrun planning test_node
```

### 3. Start Trajectory Server (Terminal 3)
```bash
cd /home/lfx-desktop/files/Replication/obj_track/Elastic-Tracker  
source devel/setup.bash
rosrun planning traj_server
```

### 4. Trigger Planning (Terminal 4)
```bash
rostopic pub /triger geometry_msgs/PoseStamped '{}'
```

## System Architecture

- **VisFly** publishes `/drone0/odom` (drone state) and `/target/odom` (target position)
- **Elastic Tracker** receives states and generates trajectories 
- **Trajectory Server** converts trajectories to `/drone0/position_cmd`
- **VisFly** executes position commands, closing the control loop

## Configuration

- **Target trajectory**: Modify `--traj` parameter in `visfly_only.launch` (A, B, C, etc.)
- **Target velocity**: Modify `--velocity` parameter (default: 2.0 m/s)
- **Planning parameters**: See Elastic Tracker's `planning.launch` for tuning

## Output

- Results saved to: `/home/lfx-desktop/files/obj_track/exps/vary_v/saved/objTracking/test/`
- Video output: `{velocity}_{traj}_elastic/video.mp4`
- Data: `{traj}_{velocity}_elastic.pth`

## Tips

- System auto-triggers planning after 2 seconds
- Use `rqt_graph` to visualize ROS node connections
- Check `/drone0/position_cmd` topic for trajectory commands
- Monitor VisFly terminal for performance metrics