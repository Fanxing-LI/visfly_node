#!/usr/bin/env python3

# Set environment variables for headless operation before importing graphics libraries
from math import e
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for headless OpenGL
# os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for MuJoCo if applicable
import torch as th
import rospy
import sys
from dynamics import Dynamics
import copy


def remove_last_n_folders(path, n=5):
    path = path.rstrip('/\\')  # 去除末尾的斜杠
    for _ in range(n):
        path = path[:path.rfind('/')] if '/' in path else ''
    return path


add_path = remove_last_n_folders(os.path.dirname(os.path.abspath(__file__)), 4)
sys.path.append(add_path)
print(add_path)

import numpy as np
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud, PointField, PointCloud2, Image
from tf.transformations import quaternion_from_euler
import threading
import argparse
from quadrotor_msgs.msg import PositionCommand, Command
import torch
from VisFly.utils.type import ACTION_TYPE
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32, Vector3, PoseStamped
from scipy.spatial.transform import Rotation as R
from VisFly.utils.common import load_yaml_config
from saveNode import SaveNode

# Topic name definitions

ODOM_TOPIC_PREFIX = "visfly/drone_{}/odom"
TARGET_ODOM_TOPIC = "visfly/drone_{}_target/odom"
DEPTH_TOPIC_PREFIX = "visfly/drone_{}/depth"

# BPTT
BPTT_CMD_TOPIC_PREFIX = "BPTT/drone_{}/action"
BPTT_ODOM_PREFIX = ODOM_TOPIC_PREFIX
BPTT_TARGET_ODOM_TOPIC_PREFIX = TARGET_ODOM_TOPIC


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT")
    parser.add_argument("--env", "-e", type=str, default="navigation")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--scene", "-sc", type=str, default="0.08tree",)
    parser.add_argument("--velocity", "-v", type=float, default=3.0, )
    return parser


class ROSIndepWrapper:
    def __init__(self, env, path,comment="BPTT"):
        self.env = env
        self.args = args
        self.num_agent = self.env.num_envs
        self.action_type = self.env.envs.dynamics.action_type

        self.comment = comment

        self.dynamics = Dynamics(cfg="drone_d435i_jetson_orin_nx")
        
        # Initialize ROS node
        rospy.init_node('visfly', anonymous=True)

        self._count = 0

        # Action data storage and lock for thread safety
        self.normalized_action = None
        self.action_data = [None] * self.num_agent
        self.state_data = [None] * self.num_agent
        self.action_lock = threading.Lock()
        self.state_lock = threading.Lock()

        # Publishers
        self.drone_odom_pubs = []
        self.drone_camera_pubs = []
        self.drone_target_odom_pubs = []
        
        # Subscribers for action
        self.drone_action_subs = []
        
        self.ex_sim_odom_subs = []
        self.ex_sim_odom_reset_pubs = []

        for i in range(self.num_agent):
            # Publisher for odometry - use different topics for different modes
            odom_prefix = BPTT_ODOM_PREFIX
            action_prefix = BPTT_CMD_TOPIC_PREFIX
            action_sub = rospy.Subscriber(action_prefix.format(i), Command, self._make_action_callback(i))

            drone_odom_pub = rospy.Publisher(odom_prefix.format(i), Odometry, queue_size=1)
            
            # depth publisher
            drone_camera_pub = rospy.Publisher(DEPTH_TOPIC_PREFIX.format(i), Image, queue_size=1)
            
            # target publisher
            drone_target_odom_pub = rospy.Publisher(TARGET_ODOM_TOPIC.format(i), Odometry, queue_size=1)
            
            self.drone_target_odom_pubs.append(drone_target_odom_pub)
            self.drone_odom_pubs.append(drone_odom_pub)
            self.drone_camera_pubs.append(drone_camera_pub)
            self.drone_action_subs.append(action_sub)
            

        self.state = th.zeros((self.num_agent, 13))
    
            
        rospy.loginfo("Calling reset to initialize environment...")
        self.reset()

        # Initialize SaveNode for data collection
        attrs = ['state', 'obs', 'reward', 't',]
        self.save_node = SaveNode(path, attrs, self.env)
        rospy.loginfo(f"SaveNode initialized with save path: {save_path}")

        # Frame IDs
        self.world_frame = "world"
        rospy.loginfo(f"Visfly ROS Environment Wrapper initialized with {self.num_agent} agents in {self.comment} mode")

    def reset(self, *args, **kwargs):
        """
        Reset the environment and clear action data.
        This method can be called to reset the environment state.
        """
        rospy.loginfo("Starting environment reset...")

        r = self.env.reset(*args, **kwargs)
        rospy.loginfo(f"Environment reset successful. Return value type: {type(r)}")
        return r

    def _make_ex_sim_odom_callback(self, agent_id):
        def callback(msg):
            self.state_data[agent_id] = msg
        return callback
    
    def _make_action_callback(self, agent_id):
        def callback(msg):
            with self.action_lock:
                # Extract action data based on comment type
                # 对于Command消息：thrust使用thrust字段，bodyrate使用angularVel
                self.action_data[agent_id] = {
                    'z_acc': msg.thrust,  # 使用thrust字段
                    'bodyrate': [msg.angularVel.x, msg.angularVel.y, msg.angularVel.z]  # 使用角速度
                }

            # if all the action data is received, prepare action for main loop
            if all(a is not None for a in self.action_data):
                action = self.process_action()
                normalized_action = self.normalize(action)
                self.normalized_action = normalized_action.clamp(-1,1)
                
                # self.normalized_action = action
                self.action_ready = True
                
        return callback
    
    def normalize(self, action):
        return self.dynamics._normalize(action=action)
        
    def main_loop(self):
        """Main loop for environment stepping - runs in main thread"""
        rospy.loginfo("Starting main control loop...")

        # 30Hz control loop
        freq = 30
        
        rate = rospy.Rate(freq)
        self.action_ready = False
        self.state_ready = False
        
        # Publish initial state
        
        rospy.loginfo("Entering main loop.")
        
        step_count = 0
        while not rospy.is_shutdown():
            # Check if new action is available
            if self.action_ready:
                # Step the environment
                # obs, reward, done, info = self.env.step(self.normalized_action, is_test=True)

                obs, reward, done, info = self.env.step(self.normalized_action, is_test=True)
                    
                step_count += 1
                
                # Collect and process data for SaveNode
                self.collect_and_process()
                
                # Reset action flag
                self.action_ready = False
                self.state_ready = False
                self.pending_action = None

                # Check if environment is done
                if done.all():
                    rospy.loginfo("All environments done. Exiting...")
                    self.save()
                    break
                
                if step_count % freq == 0:
                    rospy.loginfo(f"Step count: {step_count}")
                    
            # Always publish environment status (even without action)
            self.publish_env_status(freq=freq)
            
            # rosinfo the step count
            rate.sleep()
                
    def collect_and_process(self):
        """
        Collect and process data using SaveNode
        """
        self.save_node.stack(self.env)
    
    def save(self, path=None):
        """
        Save collected data using SaveNode
        """
        self.save_node.save(path)
        
    def process_action(self):
        """
        订阅action并提取position和yaw，组成n*4的tensor并return
        """
        with self.action_lock:
            # 提取z_acc和bodyrate组成n*4的tensor
            action_tensor = torch.zeros(self.num_agent, 4)
            for i in range(self.num_agent):
                if self.action_data[i] is not None:
                    action_tensor[i, 0] = self.action_data[i]['z_acc']
                    action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])
                
        self.action_data = [None] * self.num_agent

        return action_tensor

    def publish_env_status(self, is_count=True, freq=10):
        """
        发布所有环境信息
        """
        try:
            self.publish_drone_state()
        except Exception as e:
            rospy.logerr(f"Error publishing environment status: {e}")
            import traceback
            traceback.print_exc()
        
        if is_count:
            self._count += 1
            if self._count % freq == 0:
                rospy.loginfo(f"Published environment status at count {self._count}")

    def publish_drone_state(self):
        """
        发布无人机状态信息
        从self.envs.state获取状态：num_agent*13 (pos, quaternion, vel, angular_vel)
        """
        # Debug: Check environment state availability
        
        drone_states = self.env.state  # shape: (num_agent, 13)
        targets = self.env.target
        depths = self.env.sensor_obs["depth"]
        
        for i in range(self.num_agent):
            state = drone_states[i]  # shape: (13,) [pos(3), quat_wxyz(4), vel(3), angular_vel(3)]
            depth = depths[i]  # shape: (H, W)
            
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = self.world_frame
            odom_msg.child_frame_id = f"drone_{i}"

            # 位置 (0:3) - Apply FSC offset if needed
            odom_msg.pose.pose.position.x = float(state[0])
            odom_msg.pose.pose.position.y = float(state[1])
            odom_msg.pose.pose.position.z = float(state[2])

            # 四元数姿态 (3:7) - 状态格式是wxyz，ROS格式是xyzw
            odom_msg.pose.pose.orientation.x = float(state[4])  # x from wxyz[1]
            odom_msg.pose.pose.orientation.y = float(state[5])  # y from wxyz[2]
            odom_msg.pose.pose.orientation.z = float(state[6])  # z from wxyz[3]
            odom_msg.pose.pose.orientation.w = float(state[3])  # w from wxyz[0]

            # 线速度 (7:10)
            odom_msg.twist.twist.linear.x = float(state[7])
            odom_msg.twist.twist.linear.y = float(state[8])
            odom_msg.twist.twist.linear.z = float(state[9])

            # 角速度 (10:13)
            odom_msg.twist.twist.angular.x = float(state[10])
            odom_msg.twist.twist.angular.y = float(state[11])
            odom_msg.twist.twist.angular.z = float(state[12])
            
            self.drone_odom_pubs[i].publish(odom_msg)
            
            # Publish depth image
            depth_msg = Image()
            depth_msg.header.stamp = rospy.Time.now()
            depth_msg.header.frame_id = f"drone_{i}/depth_camera"
            depth_msg.height = depth.shape[1]
            depth_msg.width = depth.shape[2]
            depth_msg.encoding = "32FC1"
            depth_msg.is_bigendian = 0
            depth_msg.step = depth.shape[1] * 4  # 4 bytes per
            depth_msg.data = depth.astype(np.float32).tobytes()
            
            self.drone_camera_pubs[i].publish(depth_msg)
            

def get_env(algorithm="BPTT", scene="0.08tree"):
    from exps.ros_env.run import main
    eval_env = main(debug_env=True)
    return eval_env


if __name__ == '__main__':
    try:
        parser = parse_args()
        args = parser.parse_args(rospy.myargv()[1:])
        scene = args.scene
        env = get_env(scene=scene)
        current_abs_path = os.path.abspath(__file__)
        obj_track_path = current_abs_path.split('avoidance')[0] + 'avoidance'
        save_path = obj_track_path + f"/exps/ros_env/saved/scene_{scene}_velocity_{args.velocity}"

        print(f"Environment created: {env.__class__}")
        node = ROSIndepWrapper(env, path=save_path, comment=args.comment)
        
        # Start main loop in the main thread
        node.main_loop() 
        

    except rospy.ROSInterruptException:
        pass
