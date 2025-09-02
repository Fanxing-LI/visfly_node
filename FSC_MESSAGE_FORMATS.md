# FSC Message Formats for VisFly Integration

This document contains the exact message formats collected from the official FSC system for implementing the VisFly-FSC integration.

## Overview

The FSC integration requires:
- **VisFly → FSC**: Publish odometry + target point data
- **FSC → VisFly**: Subscribe to control commands

---

## 1. Odometry Message (VisFly → FSC)

**Topic:** `/hummingbird/ground_truth/odometry`  
**Type:** `nav_msgs/Odometry`  
**Purpose:** Drone state information (position, orientation, velocities)

```yaml
header: 
  seq: 7
  stamp: {secs: 0, nsecs: 420000000}
  frame_id: "world"
child_frame_id: "hummingbird/base_link"
pose: 
  pose: 
    position: {x: 2.43e-21, y: -2.58e-25, z: 0.07256}
    orientation: {x: -5.44e-23, y: -3.28e-19, z: -2.28e-24, w: 1.0}  # quaternion (x,y,z,w)
  covariance: [0.0, 0.0, ... 36 zeros]
twist: 
  twist: 
    linear: {x: -3.96e-19, y: 7.10e-23, z: -0.686}     # linear velocity [m/s]
    angular: {x: -3.36e-21, y: -1.44e-17, z: -1.06e-22} # angular velocity [rad/s]
  covariance: [0.0, 0.0, ... 36 zeros]
```

**Key Fields for VisFly:**
- `pose.pose.position.{x,y,z}` ← from `drone_state[0:3]`
- `pose.pose.orientation.{x,y,z,w}` ← from `drone_state[4,5,6,3]` (wxyz→xyzw conversion)
- `twist.twist.linear.{x,y,z}` ← from `drone_state[7:10]`
- `twist.twist.angular.{x,y,z}` ← from `drone_state[10:13]`

---

## 2. Target Point Message (VisFly → FSC)

**Topic:** `/hummingbird/aprilfake/point`  
**Type:** `sensor_msgs/PointCloud`  
**Purpose:** Target position in camera frame

```yaml
header: 
  seq: 187
  stamp: {secs: 6, nsecs: 880000000}
  frame_id: ''
points: 
  - {x: 0.00025, y: 0.13976, z: 22.977}  # Target in camera frame [CRITICAL]
  - {x: 23.0, y: 0.0, z: 3.0}           # Target in world frame [OPTIONAL]
  - {x: 368.70, y: 251.03, z: 1.0}      # Corner 1 (pixel coordinates)
  - {x: 384.31, y: 251.03, z: 1.0}      # Corner 2 
  - {x: 384.31, y: 235.42, z: 1.0}      # Corner 3
  - {x: 368.70, y: 235.42, z: 1.0}      # Corner 4
channels: []
```

**Critical Point:** First point is target in camera frame - this is what FSC uses for IBVS control!

**VisFly Implementation:**
- Use existing `world_to_camera_frame()` function to convert target position
- Create PointCloud with first point as camera-frame target

---

## 3. Control Command Message (FSC → VisFly)

**Topic:** `/hummingbird/autopilot/control_command`  
**Type:** `vision_msgs/ControlCommand`  
**Purpose:** FSC control commands for VisFly to execute

```yaml
header: 
  seq: 242
  stamp: {secs: 12, nsecs: 750000000}
  frame_id: ''
control_mode: 2                      # 2 = BODY_RATES mode
armed: True
expected_execution_time: {secs: 12, nsecs: 750000000}
orientation: {x: 3.08e-08, y: -3.21e-07, z: 1.14e-07, w: 1.0}
bodyrates: {x: 9.97e-05, y: -0.03317, z: -1.17e-07}    # [rad/s]
angular_accelerations: {x: 0.0, y: 0.0, z: 0.0}
collective_thrust: 9.813831329345703                    # [m/s²]
thrust_rate: -0.0007750673103146255
rotor_thrusts: []
```

**Key Fields for VisFly:**
- `collective_thrust` → action_tensor[i, 0] (z-axis thrust)
- `bodyrates.x` → action_tensor[i, 1] (roll rate)
- `bodyrates.y` → action_tensor[i, 2] (pitch rate) 
- `bodyrates.z` → action_tensor[i, 3] (yaw rate)

**Control Mode:** 2 = BODY_RATES (same as BPTT policy action type)

---

## Implementation Notes

### Coordinate Frames
- **World Frame**: ROS standard (x-forward, y-left, z-up)
- **Camera Frame**: FSC expects (x-forward, y-left, z-forward distance)
- **Drone State**: VisFly format [pos(3), quat_wxyz(4), vel(3), angular_vel(3)]

### Units
- **Positions**: meters [m]
- **Velocities**: meters per second [m/s] 
- **Angular velocities**: radians per second [rad/s]
- **Thrust**: meters per second squared [m/s²]

### Message Comparison vs BPTT
| Field | BPTT (mav_msgs/RateThrust) | FSC (vision_msgs/ControlCommand) |
|-------|---------------------------|----------------------------------|
| Thrust | `thrust.z` | `collective_thrust` |
| Roll rate | `angular_rates.x` | `bodyrates.x` |
| Pitch rate | `angular_rates.y` | `bodyrates.y` |
| Yaw rate | `angular_rates.z` | `bodyrates.z` |

**Action tensor mapping is identical!**

---

## Next Steps

1. **Import FSC message type** in `visfly.py`:
   ```python
   from vision_msgs.msg import ControlCommand
   ```

2. **Update FSC topic definition** (line 48):
   ```python
   FSC_CONTROL_TOPIC = "/hummingbird/autopilot/control_command"
   ```

3. **Implement FSC callback** function:
   ```python
   def _make_fsc_callback(self, agent_id):
       def callback(msg):
           with self.action_lock:
               self.action_data[agent_id] = {
                   'collective_thrust': msg.collective_thrust,
                   'bodyrate': [msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z]
               }
       return callback
   ```

4. **Update subscribe_action** for FSC mode:
   ```python
   elif self.comment == "fsc":
       action_tensor = torch.zeros(self.num_agent, 4)
       for i in range(self.num_agent):
           if self.action_data[i] is not None:
               action_tensor[i, 0] = self.action_data[i]['collective_thrust']
               action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])
       self.action_data = [None] * self.num_agent
   ```

The integration should be straightforward since FSC uses the same action format as BPTT!