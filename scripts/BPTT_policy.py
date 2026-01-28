#!/usr/bin/env python3
import sys, os
from dynamics import g

def remove_last_n_folders(path, n=5):
    path = path.rstrip('/\\')  # Remove trailing slashes
    for _ in range(n):
        path = path[:path.rfind('/')] if '/' in path else ''
    return path
add_path = remove_last_n_folders(os.path.dirname(os.path.abspath(__file__)), 4)
sys.path.append(add_path)

# ===================================================================================

# ONNX model filename (will be combined with current file directory)
# ONNX_MODEL_FILENAME = "SHAC_std_3_policy.onnx"
ONNX_MODEL_FILENAME = "SHAC_std_1_policy.onnx"
# ONNX_MODEL_FILENAME = "SHAC_MoreDenseScene_maxRand15_plane30_9_policy.onnx"
DEBUG = False
REAL_WORLD = False  # Set to True when running in real-world environment
VICON = False
cali_g = 10.3

USE_EKF = False

CTRL_FREQ = 30
INFO_PRINT_FREQ = CTRL_FREQ

# Action topic prefix configuration
ACTION_TOPIC_PREFIX = "BPTT/drone_0/action"
ODOM_TOPIC_PREFIX = "visfly/drone_0/odom"
DEPTH_TOPIC_PREFIX = "visfly/drone_0/depth"
TARGET_ODOM_TOPIC = "target/odom"

# for real world 
# TODO: overwrite the ACTION and ODOM topic prefix
if REAL_WORLD:
    ACTION_TOPIC_PREFIX = "/bfctrl/cmd"
    ODOM_TOPIC_PREFIX = "/vins_estimator/imu_propagate"  # TODO: VINS VIO
    OMEGA_TOPIC_PREFIX = "/mavros/imu/data"
    DEPTH_TOPIC_PREFIX = "/camera/depth/image_raw"
    TARGET_ODOM_TOPIC = None # Fixed target velocity in real world environment
    if VICON:
        ODOM_TOPIC_PREFIX = "/vicon/hyx/odom"
        
if USE_EKF:
    TARGET_ODOM_TOPIC = "ekf/odom"

# ===================================================================================

import rospy
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3
from quadrotor_msgs.msg import Command
import threading
import numpy as np
import torch as th
import onnxruntime as ort
import os
import torch.nn.functional as F

flex_max_pool = lambda x, k: F.max_pool2d(th.as_tensor(x), kernel_size=k, stride=k)
flex_avg_pool = lambda x, k: F.avg_pool2d(th.as_tensor(x), kernel_size=k, stride=k)

# Import VisFly quaternion utilities
try:
    from maths import Quaternion
except ImportError as e:
    from VisFly.utils.maths import Quaternion
    
try:
    from dynamics import Dynamics
except ImportError as e:
    from VisFly.envs.base.dynamics import Dynamics
    
    
max_dis = 24.
min_dis = 0.1
scale = 3.

depth_preprocess = lambda x: 1 / (1 + th.as_tensor(x).clamp(min_dis, max_dis) / scale)

class BPTTPolicy:
    """
    BPTT-based policy using bodyrate and z-axis acceleration control
    Subscribes to multiple drone odom topics and publishes PositionCommand
    """
    def __init__(self, indice=0, max_v=3.):
        # Initialize ROS node
        rospy.init_node('bptt_policy', anonymous=True)

        # Load ONNX model
        self.onnx_session = self._load_onnx_model()
        self.indice = indice
        self.max_v = max_v
        self.pre_target_pos = None

        # Create publishers and subscribers
        self.position_cmd_publisher = None
        self.odom_subscriber = None
        self.target_odom_subscriber = None
        self.camera_subscriber = None

        # Thread lock for thread safety
        self.lock = threading.Lock()

        # Store latest odom data
        self.latest_odom = None
        self.latest_target_odom = None  # Store target position information
        self.latest_imu = None
        self.latest_depth = None
        
        self.dynamics = Dynamics(cfg="drone_d435i_jetson_orin_nx")

        # Environment status information
        self.state = {
            'position': np.zeros((1, 3)),
            'velocity': np.zeros((1, 3)),
            'orientation': np.zeros((1, 4)),  # quaternion (x,y,z,w)
            'angular_velocity': np.zeros((1, 3)),
            'target': np.zeros((1, 3)),
        }

        # BPTT policy parameters
        self.device = th.device('cpu')
        self.setup_publishers_and_subscribers()

        self._count = 0

        # Add simple running status reminder timer
        self.alive_timer = rospy.Timer(rospy.Duration(2.0), self._alive_callback)

        # target odom monitoring: record last received time
        self.last_target_odom_time = None
        self.target_watchdog_timer = rospy.Timer(rospy.Duration(0.5), self._target_watchdog_callback)

    def setup_publishers_and_subscribers(self):
        """Set up publishers and subscribers"""
        # Create Action publisher for each drone using new topic format
        self.position_cmd_publisher = rospy.Publisher(ACTION_TOPIC_PREFIX.format(self.indice), Command, queue_size=10)

        # Create Odometry subscriber for each drone
        self.odom_subscriber = rospy.Subscriber(ODOM_TOPIC_PREFIX.format(self.indice), Odometry, self._make_drone_odom_callback)

        # Create global target pose subscriber
        self.target_odom_subscriber = rospy.Subscriber(TARGET_ODOM_TOPIC.format(self.indice), Odometry, self._target_odom_callback)
        
        # create camera subscriber
        self.camera_subscriber = rospy.Subscriber(DEPTH_TOPIC_PREFIX.format(self.indice), Image, self._make_camera_callback)
        
        if REAL_WORLD:
            rospy.loginfo("Real-world mode: Subscribing to real-world topics")
            # If in real-world mode, subscribe to IMU data for angular velocity
            self.imu_subscriber = rospy.Subscriber(OMEGA_TOPIC_PREFIX, Imu, self._make_imu_callback(0))
        
        
    def _load_onnx_model(self):
        """Load ONNX model"""
        # Construct full path from current file directory and model filename
        current_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_model_path = os.path.join(current_dir, ONNX_MODEL_FILENAME)
        
        if not os.path.exists(onnx_model_path):
            rospy.logwarn(f"ONNX model file not found: {onnx_model_path}")
            return None

        # Create ONNX Runtime inference session
        providers = ['CPUExecutionProvider']  # Use CPU inference
        session = ort.InferenceSession(onnx_model_path, providers=providers)

        rospy.loginfo(f"Successfully loaded ONNX model: {onnx_model_path}")
        rospy.loginfo(f"Model inputs: {[input.name for input in session.get_inputs()]}")
        rospy.loginfo(f"Model outputs: {[output.name for output in session.get_outputs()]}")

        return session


    def _run_policy_inference(self, obs):
        """Run ONNX model inference"""
        if self.onnx_session is None:
            # raise error
            rospy.logerr("ONNX session is not initialized")

        try:
            # Prepare input data
            # input_name = self.onnx_session.get_inputs()[0].name
            input_data = obs

            # Run inference
            outputs = self.onnx_session.run(None, input_data)

            # Get output action - BPTT uses bodyrate and z-axis acceleration
            action = outputs[0].flatten()
            # print(action)

            self._count += 1
            if self._count % INFO_PRINT_FREQ == 0:
                rospy.loginfo(f"ONNX inference count: {self._count}")
            return action

        except Exception as e:
            rospy.logwarn(f"ONNX inference failed: {e}")
            return np.zeros(4)  # Return default action

    def de_normalize(self, action):
        return self.dynamics._de_normalize(action)

    def _make_drone_odom_callback(self, odom_msg):
        self.latest_odom = odom_msg

    def _target_odom_callback(self, target_odom_msg):
        self.latest_target_odom = target_odom_msg
        self.last_target_odom_time = rospy.Time.now()

    def _make_imu_callback(self, imu_msg):
        """Create IMU callback function for specific drone"""
        self.latest_imu = imu_msg
        
    def _make_camera_callback(self, depth_msg):
        """Create camera callback function for specific drone"""
        self.latest_depth = depth_msg
        
    def _process_target_odom(self):
        """Process target pose message, update preprocessed target position"""
        # Update max velocity from parameter server
        self.max_v = rospy.get_param('/visfly/max_velocity', self.max_v)

        target_odom_msg = self.latest_target_odom
        if target_odom_msg is not None:
            if target_odom_msg.pose.pose.position.x != 0:
                pos_target = th.tensor([
                    target_odom_msg.pose.pose.position.x,
                    target_odom_msg.pose.pose.position.y,
                    target_odom_msg.pose.pose.position.z
                ], dtype=th.float32)
                rela_target = (pos_target - th.as_tensor(self.state["position"], dtype=th.float32))
                rela_dis = rela_target.norm(dim=0)
                self.target = (
                    (rela_target/ rela_dis)*(rela_dis/1).clamp_max(self.max_v)
                ).squeeze()
            else:
                self.target = th.tensor([
                    target_odom_msg.twist.twist.linear.x,
                    target_odom_msg.twist.twist.linear.y,
                    target_odom_msg.twist.twist.linear.z
                ], dtype=th.float32, device=self.device)
        else:
            rospy.logwarn_throttle(2.0, "No target_odom message received yet")
    def _process_self_odom(self):
        """Update environment status"""
        # for drone_id, odom_msg in enumerate(self.latest_odom):

        odom_msg, imu_msg = self.latest_odom, self.latest_imu
        
        self.state['position'] = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ]


        # Update velocity
        self.state['velocity'] = [
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z
        ]

        # Update angular velocity
        if not REAL_WORLD:
            self.state['angular_velocity'] = [
                odom_msg.twist.twist.angular.x,
                odom_msg.twist.twist.angular.y,
                odom_msg.twist.twist.angular.z
            ]
            self.state['orientation'] = [
                odom_msg.pose.pose.orientation.w,
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
            ]
        else:
            self.state['angular_velocity'] = [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z
            ]
            self.state['orientation'] = [
                odom_msg.pose.pose.orientation.w,
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
            ]
    
    def _process_camera(self):
        if self.latest_depth is not None:
            depth_msg = self.latest_depth
            # Convert depth_msg to numpy array
            depth_width = depth_msg.width
            depth_height = depth_msg.height
            depth_data = np.frombuffer(depth_msg.data, dtype=np.float32).reshape((depth_height, depth_width))
            self.depth_image = depth_data
            # resize image from 360 480, to 12, 16


    def _update_env_status(self):
        self._process_self_odom()
        self._process_target_odom()
        self._process_camera()
        
    def preprocess_input(self):
            
        # Get current state
        # position = th.tensor(self.state['position'], dtype=th.float32, device=self.device)
        velocity = th.tensor(self.state['velocity'], dtype=th.float32, device=self.device)
        orientation = th.tensor(self.state['orientation'], dtype=th.float32, device=self.device)
        angular_velocity = th.tensor(self.state['angular_velocity'], dtype=th.float32, device=self.device)
        orientation = Quaternion(
            w=th.tensor(orientation[0], dtype=th.float32, device=self.device),
            x=th.tensor(orientation[1], dtype=th.float32, device=self.device),
            y=th.tensor(orientation[2], dtype=th.float32, device=self.device),
            z=th.tensor(orientation[3], dtype=th.float32, device=self.device)
        )
        
        # Get target position and velocity from subscribed topic (now using Odometry message)
        head_target_velocity = orientation.world_to_head(self.target.T).T

        head_v = orientation.world_to_head(velocity.T).T

        # Get quaternion's 4 components as orientation features, reference ObjectTrackingEnv
        orientation_vec = orientation.toTensor().T.to(self.device) # Convert to tensor format
        # print all cat variable shape
        # Build state vector, reference ObjectTrackingEnv's get_observation
        state = th.hstack([
            head_target_velocity / 10,  # 3D - target position in head coordinate system
            orientation_vec,  # 4D - quaternion (w,x,y,z)
            head_v / 10,  # 3D - velocity in head coordinate system (normalized)
            angular_velocity / 10,  # 3D - angular velocity (normalized)
        ])  # Total 16 dimensions

        # self.state = state

        dim = 30
        k1 = 6
        obs = {
            "state":state.unsqueeze(0).numpy(),
            # "depth": flex_max_pool(depth_preprocess(flex_max_pool(np.expand_dims(self.depth_image,0), k1)), int(dim / k1)).unsqueeze(0).numpy()
            "depth": flex_max_pool(depth_preprocess(flex_avg_pool(np.expand_dims(self.depth_image,0), k1)), int(dim / k1)).unsqueeze(0).numpy()
            # "depth":depth_preprocess(self.depth_image).unsqueeze(0).unsqueeze(0).numpy()
        }
        return obs

    def _publish_command(self, obs):
        """Publish Command based on BPTT policy, using bodyrate and z-axis acceleration"""
        # Preprocess input
        # state = self.preprocess_input(drone_id)

        # Use ONNX model for inference
        action = self._run_policy_inference(obs)
        action = th.atleast_2d(th.as_tensor(action))
        action = self.de_normalize(action)

        # Create Command message
        cmd = Command()

        # Set message header
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "world"

        # BPTT policy specific: ONNX model output format is [z_acc, bodyrate_x, bodyrate_y, bodyrate_z]
        # Command message fields:
        # - thrust: float64 (single value, not Vector3)
        # - angularVel: geometry_msgs/Vector3 (angular velocity)

        if REAL_WORLD:
            # In real-world, convert z-acceleration to z-thrust using mass
            action[0] = action[0] / self.dynamics.m - g + cali_g # f = m * a_z
        else:
            cmd.thrust = action[0] # / self.dynamics.m # z-axis thrust as single value
        cmd.angularVel.x = action[1]  # x-axis roll rate
        cmd.angularVel.y = action[2]  # y-axis pitch rate
        cmd.angularVel.z = action[3]  # z-axis yaw rate
        
        # Set mode to angular velocity control mode
        cmd.mode = cmd.ANGULAR_MODE

        # Publish message
        self.position_cmd_publisher.publish(cmd)

    def _alive_callback(self, event):
        """Simple running status reminder"""
        rospy.loginfo("🚁 BPTT Policy running...")

    def _target_watchdog_callback(self, event):
        """Monitor if target odom has not been updated for a long time (1s)."""
        with self.lock:
            now = rospy.Time.now()
            if self.last_target_odom_time is None:
                rospy.logwarn_throttle(2.0, "No target_odom message received yet (>=1s)")
                return
            gap = (now - self.last_target_odom_time).to_sec()
            if gap > 1.0:
                rospy.logwarn_throttle(1.0, f"No target_odom received in {gap:.2f}s (threshold 1.0s)")

    def run(self):
        """Run BPTT Policy node"""
        rospy.loginfo("BPTT Policy starting in event-driven mode")
        rate = rospy.Rate(CTRL_FREQ)  # Set rate to 33Hz for any periodic tasks if needed
        while True:
            if rospy.is_shutdown():
                break
            
            if self.status_check():
                self._update_env_status()
                cat_state = self.preprocess_input()
                
                self._publish_command(cat_state)
                rate.sleep()

    def status_check(self):
        """Check if all necessary data has been received"""
        with self.lock:
            if self.latest_odom is None:
                return False
            if self.latest_target_odom is None:
                return False
            if self.latest_depth is None:
                return False
            # if any(imu is None for imu in self.latest_imu):
            #     return False
            return True
        
def main():
    """
    Main function - Create and run BPTTPolicy
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run BPTT policy ROS node")
    parser.add_argument("--indice", type=int, default=0, help="Drone index used in topic formatting")
    parser.add_argument("-v","--velocity", type=float, default=12.0, help="Max target velocity clamp")
    args = parser.parse_args()

    try:
        # Get agent count from parameter server, default is 4
        if not REAL_WORLD:
            num_agent = rospy.get_param('~num_agent', 1)
        else:
            num_agent = 1
            
        policy = BPTTPolicy(num_agent)
        policy.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("BPTT Policy stopped")


if __name__ == '__main__':
    main()
