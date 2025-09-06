#!/usr/bin/env python3
import sys, os

def remove_last_n_folders(path, n=5):
    path = path.rstrip('/\\')  # Remove trailing slashes
    for _ in range(n):
        path = path[:path.rfind('/')] if '/' in path else ''
    return path


add_path = remove_last_n_folders(os.path.dirname(os.path.abspath(__file__)), 4)
sys.path.append(add_path)
print(add_path)


# ONNX model filename (will be combined with current file directory)
ONNX_MODEL_FILENAME = "SHAC_NoCaliHeadV_Pos_Dis3.0_spd3.4_lessNoise_2_policy.onnx"

DEBUG = True
REAL_WORLD = True  # Set to True when running in real-world environment
USE_EKF = True

# Action topic prefix configuration
ACTION_TOPIC_PREFIX = "BPTT/drone_{}/action"
ODOM_TOPIC_PREFIX = "visfly/drone_{}/odom"
TARGET_ODOM_TOPIC = "visfly/target/odom"
CTRL_FREQ = 33
INFO_PRINT_FREQ = 30

# for real world 
# TODO: overwrite the ACTION and ODOM topic prefix
if REAL_WORLD:
    ACTION_TOPIC_PREFIX = "/bfctrl/cmd"
    ODOM_TOPIC_PREFIX = "/vicon/cc1/odom"
    OMEGA_TOPIC_PREFIX = "/mavros/imu/data"
    if USE_EKF:
        TARGET_ODOM_TOPIC = "ekf/odom"


import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3
from quadrotor_msgs.msg import Command
import threading
import numpy as np
import torch as th
import onnxruntime as ort
import os

# Import VisFly quaternion utilities
try:
    from maths import Quaternion
except ImportError as e:
    from VisFly.utils.maths import Quaternion
    
try:
    from dynamics import Dynamics
except ImportError as e:
    from VisFly.envs.base.dynamics import Dynamics
    

class BPTTPolicy:
    """
    BPTT-based policy using bodyrate and z-axis acceleration control
    Subscribes to multiple drone odom topics and publishes PositionCommand
    """
    def __init__(self, num_agent=4):
        self.num_agent = num_agent

        # Initialize ROS node
        rospy.init_node('bptt_policy', anonymous=True)

        # Load ONNX model
        self.onnx_session = self._load_onnx_model()

        self.pre_target_pos = None

        # Create publishers and subscribers
        self.position_cmd_publishers = []
        self.env_status_publisher = None
        self.odom_subscribers = []
        self.target_odom_subscriber = None

        # Thread lock for thread safety
        self.lock = threading.Lock()

        # Store latest odom data
        self.latest_odom = [None] * self.num_agent
        self.latest_target_odom = None  # Store target position information
        self.latest_imu = [None] # For real-world angular velocity
        
        self.dynamics = Dynamics(cfg="drone_state")

        # Environment status information
        self.env_status = {
            'positions': np.zeros((self.num_agent, 3)),
            'velocities': np.zeros((self.num_agent, 3)),
            'orientations': np.zeros((self.num_agent, 4)),  # quaternion (x,y,z,w)
            'angular_velocities': np.zeros((self.num_agent, 3)),
            'head_targets': np.zeros((self.num_agent, 3)),
            'head_targets_v': np.zeros((self.num_agent, 3)),
            'head_v': np.zeros((self.num_agent, 3)),
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

        rospy.loginfo(f"BPTT Policy started, managing {self.num_agent} drones")
    
    def setup_publishers_and_subscribers(self):
        """Set up publishers and subscribers"""

        # Create environment status publisher
        # self.env_status_publisher = rospy.Publisher('visfly/env_status', PoseStamped, queue_size=10)

        for i in range(self.num_agent):
            # Create Action publisher for each drone using new topic format
            pub = rospy.Publisher(ACTION_TOPIC_PREFIX.format(i), Command, queue_size=10)
            self.position_cmd_publishers.append(pub)

            # Create Odometry subscriber for each drone
            odom_sub = rospy.Subscriber(ODOM_TOPIC_PREFIX.format(i), Odometry, self._make_drone_odom_callback(i))
            self.odom_subscribers.append(odom_sub)

        # Create global target pose subscriber
        self.target_odom_subscriber = rospy.Subscriber(TARGET_ODOM_TOPIC, Odometry, self._target_odom_callback)
        
        if REAL_WORLD:
            rospy.loginfo("Real-world mode: Subscribing to real-world topics")
            # If in real-world mode, subscribe to IMU data for angular velocity
            self.imu_subscriber = rospy.Subscriber(OMEGA_TOPIC_PREFIX, Imu, self._make_imu_callback(0))
        
    def _load_onnx_model(self):
        """Load ONNX model"""
        try:
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

        except Exception as e:
            rospy.logerr(f"Failed to load ONNX model: {e}")
            return None

    def _run_policy_inference(self, state):
        """Run ONNX model inference"""
        if self.onnx_session is None:
            rospy.logwarn("ONNX model not loaded, returning zero action")
            return np.zeros(4)  # Return default action [f, wx, wy, wz]

        try:
            # Prepare input data
            input_name = self.onnx_session.get_inputs()[0].name
            input_data = {input_name: state.cpu().numpy().reshape(1, -1)}

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

    def _make_drone_odom_callback(self, drone_id):
        """Create odom callback function for specific drone"""
        def odom_callback(odom_msg):
            with self.lock:
                self.latest_odom[drone_id] = odom_msg
        return odom_callback

    def _target_odom_callback(self, target_odom_msg):
        """Global target pose callback function"""
        with self.lock:
            # Update latest target pose
            self.latest_target_odom = target_odom_msg
            self.last_target_odom_time = rospy.Time.now()

    def _make_imu_callback(self, drone_id):
        """Create IMU callback function for specific drone"""
        def imu_callback(imu_msg):
            with self.lock:
                self.latest_imu[drone_id] = imu_msg
        return imu_callback
        
    def _process_target_odom(self):
        """Process target pose message, update preprocessed target position"""
        target_odom_msg = self.latest_target_odom
        if target_odom_msg is not None:
            self.target_pos = th.tensor([
                target_odom_msg.pose.pose.position.x,
                target_odom_msg.pose.pose.position.y,
                target_odom_msg.pose.pose.position.z
            ], dtype=th.float32, device=self.device)
            self.target_v_world = th.tensor([
                target_odom_msg.twist.twist.linear.x,
                target_odom_msg.twist.twist.linear.y,
                target_odom_msg.twist.twist.linear.z
            ], dtype=th.float32, device=self.device)
        else:
            rospy.logwarn_throttle(2.0, "No target_odom message received yet")
            

    def _process_self_odom(self):
        """Update environment status"""
        # for drone_id, odom_msg in enumerate(self.latest_odom):
        for drone_id, (odom_msg, imu_msg) in enumerate(zip(self.latest_odom, self.latest_imu)):
            # Update position
            self.env_status['positions'][drone_id] = [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z
            ]

            # Update velocity
            self.env_status['velocities'][drone_id] = [
                odom_msg.twist.twist.linear.x,
                odom_msg.twist.twist.linear.y,
                odom_msg.twist.twist.linear.z
            ]

            self.env_status['orientations'][drone_id] = [
                odom_msg.pose.pose.orientation.w,
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
            ]

            # Update angular velocity
            self.env_status['angular_velocities'][drone_id] = [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z
            ]

    def _update_env_status(self):
        self._process_self_odom()
        self._process_target_odom()
        
    def preprocess_input(self):
        """Preprocess policy input, reference ObjectTrackingEnv's update_target and get_observation"""
        for drone_id in range(self.num_agent):
            if self.latest_odom[drone_id] is None:
                return None
            
            # Get current state
            position = th.tensor(self.env_status['positions'][drone_id], dtype=th.float32, device=self.device)
            velocity = th.tensor(self.env_status['velocities'][drone_id], dtype=th.float32, device=self.device)
            orientation_q = self.env_status['orientations'][drone_id]
            angular_velocity = th.tensor(self.env_status['angular_velocities'][drone_id], dtype=th.float32, device=self.device)

            # Convert ROS quaternion format (x,y,z,w) to VisFly format (w,x,y,z)
            orientation = Quaternion(
                w=th.tensor(orientation_q[0], dtype=th.float32, device=self.device),
                x=th.tensor(orientation_q[1], dtype=th.float32, device=self.device),
                y=th.tensor(orientation_q[2], dtype=th.float32, device=self.device),
                z=th.tensor(orientation_q[3], dtype=th.float32, device=self.device)
            )

        # Get target position and velocity from subscribed topic (now using Odometry message)
        target_world = self.target_pos
        target_v_world = self.target_v_world

        # Calculate relative target position and velocity
        rela_tar = target_world - position
        rela_v = target_v_world - velocity

        # Use VisFly's quaternion method to calculate target position and velocity in head coordinate system
        head_targets = orientation.world_to_head(rela_tar.unsqueeze(0).T).T
        # if self._count % 20 == 0:
        #     print(orientation)
        head_targets_v = orientation.world_to_head(rela_v.unsqueeze(0).T).T
        head_v = orientation.world_to_head(velocity.unsqueeze(0).T).T

        # Get quaternion's 4 components as orientation features, reference ObjectTrackingEnv
        orientation_vec = th.atleast_2d(orientation.toTensor().to(self.device))  # Convert to tensor format
        angular_velocity = th.atleast_2d(angular_velocity)
        # print all cat variable shape
        # Build state vector, reference ObjectTrackingEnv's get_observation
        state = th.hstack([
            head_targets,  # 3D - target position in head coordinate system
            head_targets_v,  # 3D - target velocity in head coordinate system
            orientation_vec,  # 4D - quaternion (w,x,y,z)
            head_v / 10,  # 3D - velocity in head coordinate system (normalized)
            angular_velocity / 10,  # 3D - angular velocity (normalized)
        ])  # Total 16 dimensions

        # Update environment status for publishing
        self.env_status['head_targets'][drone_id] = head_targets.numpy()
        self.env_status['head_targets_v'][drone_id] = head_targets_v.numpy()
        self.env_status['head_v'][drone_id] = head_v.numpy()

        self.state = state
        
        return state

    def _publish_command(self, drone_id):
        """Publish Command based on BPTT policy, using bodyrate and z-axis acceleration"""
        # Preprocess input
        # state = self.preprocess_input(drone_id)
        state = self.state[drone_id]

        # Use ONNX model for inference
        action = self._run_policy_inference(state)
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

        cmd.thrust = action[0] / self.dynamics.m # z-axis thrust as single value
        cmd.angularVel.x = action[1]  # x-axis roll rate
        cmd.angularVel.y = action[2]  # y-axis pitch rate
        cmd.angularVel.z = action[3]  # z-axis yaw rate
        
        # Set mode to angular velocity control mode
        cmd.mode = cmd.ANGULAR_MODE

        # Publish message
        self.position_cmd_publishers[drone_id].publish(cmd)

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
                self.preprocess_input()
                
                for drone_id in range(self.num_agent):
                    self._publish_command(drone_id)
                rate.sleep()

    def status_check(self):
        """Check if all necessary data has been received"""
        with self.lock:
            if any(odom is None for odom in self.latest_odom):
                return False
            if self.latest_target_odom is None:
                return False
            if any(imu is None for imu in self.latest_imu):
                return False
            return True
        
def main():
    """
    Main function - Create and run BPTTPolicy
    """
    try:
        # Get agent count from parameter server, default is 4
        if not REAL_WORLD:
            num_agent = rospy.get_param('~num_agent', 4)
        else:
            num_agent = 1
            
        policy = BPTTPolicy(num_agent)
        policy.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("BPTT Policy stopped")


if __name__ == '__main__':
    main()
