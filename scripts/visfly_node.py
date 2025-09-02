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
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud, PointField, PointCloud2
from tf.transformations import quaternion_from_euler
import threading
import argparse
from exps.vary_v.run import change_v_in_json, env_alias
from quadrotor_msgs.msg import PositionCommand, Command
from mav_msgs.msg import RateThrust
from vision_msgs.msg import ControlCommand
import torch
from VisFly.utils.type import ACTION_TYPE
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32, Vector3, PoseStamped
from scipy.spatial.transform import Rotation as R
from VisFly.utils.common import load_yaml_config
from saveNode import SaveNode

# Topic name definitions

ODOM_TOPIC_PREFIX = "visfly/drone_{}/odom"
TARGET_ODOM_TOPIC = "visfly/target/odom"
POINTCLOUD_TOPIC = "visfly/env/pointcloud"

# BPTT
BPTT_CMD_TOPIC_PREFIX = "BPTT/drone_{}/action"
BPTT_ODOM_PREFIX = ODOM_TOPIC_PREFIX
BPTT_TARGET_ODOM_TOPIC_PREFIX = TARGET_ODOM_TOPIC

# Elastic
# ELASTIC_TARGET_TOPIC = "/target/odom"
ELASTIC_CMD_TOPIC_PREFIX = "/drone{}/debug"
ELASTIC_ODOM_PREFIX = ODOM_TOPIC_PREFIX # TODO: Finish this section
ELASTIC_TARGET_ODOM_TOPIC_PREFIX = TARGET_ODOM_TOPIC # TODO: Finish this section

# FSC
FSC_CMD_TOPIC_PREFIX = "/hummingbird/autopilot/control_command"
FSC_ODOM_PREFIX = ODOM_TOPIC_PREFIX
FSC_TARGET_ODOM_TOPIC_PREFIX = TARGET_ODOM_TOPIC
# FSC_ODOM_TOPIC = "/hummingbird/ground_truth/odometry"
# FSC_TARGET_TOPIC = "/hummingbird/aprilfake/point"
# FSC_CONTROL_TOPIC = "/hummingbird/autopilot/control_command"
# FSC_MOTOR_TOPIC = "/hummingbird/command/motor_speed"

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="elastic")
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT")
    parser.add_argument("--env", "-e", type=str, default="objTracking")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--traj", "-tr", type=str, default="D", )
    parser.add_argument("--velocity", "-v", type=float, default=3.0, )
    parser.add_argument("--num_agent", "-n", type=int, default=4, )
    return parser


class ROSIndepWrapper:
    def __init__(self, env, path,comment="BPTT"):
        self.env = env
        self.args = args
        self.num_agent = self.env.num_envs
        self.action_type = self.env.envs.dynamics.action_type
        if comment == "elastic":
            self.env.envs.dynamics.action_type = ACTION_TYPE.POSITION
            assert self.action_type == ACTION_TYPE.POSITION, f"current action type is {self.action_type}, but it should be 'position' for elastic"
        elif comment == "BPTT":
            assert self.action_type == ACTION_TYPE.BODYRATE, f"current action type is {self.action_type}, but it should be 'bodyrate' for BPTT"
        elif comment == "FSC":
            assert self.action_type == ACTION_TYPE.BODYRATE, f"current action type is {self.action_type}, but it should be 'bodyrate' for FSC"
        self.comment = comment

        self.dynamics = Dynamics(cfg="drone_state")
        
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
        # Subscribers for action
        self.drone_action_subs = []
        
        self.ex_sim_odom_subs = []
        self.ex_sim_odom_reset_pubs = []

        for i in range(self.num_agent):
            # Publisher for odometry - use different topics for different modes
            if self.comment == "elastic":
                odom_prefix = ELASTIC_ODOM_PREFIX
                action_prefix = ELASTIC_CMD_TOPIC_PREFIX
                action_sub = rospy.Subscriber(action_prefix.format(i), PositionCommand, self._make_action_callback(i))
                
                ex_sim_odom_sub = \
                rospy.Subscriber(f"/drone{i}/odom", Odometry, self._make_ex_sim_odom_callback(i))
                
                ex_sim_odom_reset_pub = \
                    rospy.Publisher(f"/drone{i}/odom_reset", Odometry, queue_size=1)

                self.ex_sim_odom_subs.append(ex_sim_odom_sub)
                self.ex_sim_odom_reset_pubs.append(ex_sim_odom_reset_pub)

            elif self.comment == "FSC":
                odom_prefix = FSC_ODOM_PREFIX
                action_prefix = FSC_CMD_TOPIC_PREFIX  # 使用FSC的正确topic
                action_sub = rospy.Subscriber(action_prefix, ControlCommand, self._make_action_callback(i))  # 使用ControlCommand
    
            elif self.comment == "BPTT":
                odom_prefix = BPTT_ODOM_PREFIX
                action_prefix = BPTT_CMD_TOPIC_PREFIX
                action_sub = rospy.Subscriber(action_prefix.format(i), Command, self._make_action_callback(i))

            drone_odom_pub = rospy.Publisher(odom_prefix.format(i), Odometry, queue_size=1)
            
            self.drone_odom_pubs.append(drone_odom_pub)
            self.drone_action_subs.append(action_sub)

        # Target publisher - use different topic for different modes
        if self.comment == "elastic":
            # TODO: 
            target_prefix = ELASTIC_TARGET_ODOM_TOPIC_PREFIX
            self.target_pub = rospy.Publisher(target_prefix, Odometry, queue_size=1)
            
        elif self.comment == "FSC":
            target_prefix = FSC_TARGET_ODOM_TOPIC_PREFIX
            self.target_pub = rospy.Publisher(target_prefix, PoseStamped, queue_size=1)
            
        elif self.comment == "BPTT":
            target_prefix = BPTT_TARGET_ODOM_TOPIC_PREFIX
            self.target_pub = rospy.Publisher(target_prefix, Odometry, queue_size=1)
            
        # Point cloud publisher: elastic uses PointCloud2, others keep PointCloud for existing consumers
        if self.comment == "elastic":
            self.pointcloud_pub = rospy.Publisher(POINTCLOUD_TOPIC, PointCloud2, queue_size=1)
        else:
            self.pointcloud_pub = rospy.Publisher(POINTCLOUD_TOPIC, PointCloud, queue_size=1)
        
        self.state = th.zeros((self.num_agent, 13))
    
            
        rospy.loginfo("Calling reset to initialize environment...")
        self.reset()

        # Initialize SaveNode for data collection
        attrs = ['state', 'obs', 'reward', 't', 'target', 'target_dis', "box_center"]
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

    def reset_ex_sim_odom(self):
        state = self.env.state
        if len(self.ex_sim_odom_reset_pubs) != 0:
            for i in range(self.num_agent):
                # create Odometry message from state
                odom_msg = Odometry()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = self.world_frame
                odom_msg.child_frame_id = f"drone_{i}"

                    # 位置
                odom_msg.pose.pose.position.x = float(state[i, 0])
                odom_msg.pose.pose.position.y = float(state[i, 1])
                odom_msg.pose.pose.position.z = float(state[i, 2])

                # 姿态 (四元数 wxyz -> xyzw)
                odom_msg.pose.pose.orientation.x = float(state[i, 4])
                odom_msg.pose.pose.orientation.y = float(state[i, 5])
                odom_msg.pose.pose.orientation.z = float(state[i, 6])
                odom_msg.pose.pose.orientation.w = float(state[i, 3])

                # 线速度
                odom_msg.twist.twist.linear.x = float(state[i, 7])
                odom_msg.twist.twist.linear.y = float(state[i, 8])
                odom_msg.twist.twist.linear.z = float(state[i, 9])

                # 角速度
                odom_msg.twist.twist.angular.x = float(state[i, 10])
                odom_msg.twist.twist.angular.y = float(state[i, 11])
                odom_msg.twist.twist.angular.z = float(state[i, 12])

                # 发布到仿真odom_reset topic
                self.ex_sim_odom_reset_pubs[i].publish(odom_msg)

    def _make_ex_sim_odom_callback(self, agent_id):
        def callback(msg):
            self.state_data[agent_id] = msg
        return callback
    
    def _make_action_callback(self, agent_id):
        def callback(msg):
            with self.action_lock:
                # Extract action data based on comment type
                if self.comment == "elastic":
                    # 对于PositionCommand消息：提取position和yaw
                    self.action_data[agent_id] = {
                        'position': [msg.position.x, msg.position.y, msg.position.z],
                        'yaw': msg.yaw
                    }
                elif self.comment == "BPTT":
                    # 对于Command消息：thrust使用thrust字段，bodyrate使用angularVel
                    self.action_data[agent_id] = {
                        'z_acc': msg.thrust,  # 使用thrust字段
                        'bodyrate': [msg.angularVel.x, msg.angularVel.y, msg.angularVel.z]  # 使用角速度
                    }
                elif self.comment == "FSC":
                    # 对于ControlCommand消息：使用collective_thrust和bodyrates
                    self.action_data[agent_id] = {
                        'collective_thrust': msg.collective_thrust,  # 使用collective_thrust字段
                        'bodyrate': [msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z]  # 使用bodyrates
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
        
        # Reset environment
        # obs = self.env._observations
        # rospy.loginfo("Environment reset successful")
        
        if self.comment == "elastic":
            for i in range(200):
                rospy.sleep(0.01)
                self.reset_ex_sim_odom()
            rospy.sleep(0.10)
            self.publish_env_status()
            
            state = self.process_state()
            self.state = state
            self.env.envs.reset(
                    state=(self.state[:,:3], self.state[:,3:7], self.state[:,7:10], self.state[:,10:13])
            )
            self.env.envs.sceneManager.step()
            self.env.envs.sceneManager.set_pose(
                position=self.state[:,:3],
                rotation=self.state[:,3:7]
            )
            self.env.envs.update_observation()
            self.env.get_full_observation()
        
        self.collect_and_process()
        start_obj_pos = copy.deepcopy(env.envs.dynamic_object_position[0].clone())
        # 30Hz control loop
        
        freq = 33   
        
        rate = rospy.Rate(freq)
        self.action_ready = False
        self.state_ready = False
        self.pending_action = None
        
        # Publish initial state
        
        rospy.loginfo("Entering main loop.")
        
        step_count = 0
        prev_len = 0
        n_round = 0
        
        while not rospy.is_shutdown():
            if all(s is not None for s in self.state_data):
                state = self.process_state()
                self.state = state
                self.state_ready = True
            # Check if new action is available
            if (self.state_ready or self.action_ready):
                # Step the environment
                # obs, reward, done, info = self.env.step(self.normalized_action, is_test=True)

                if self.comment == "BPTT":
                    obs, reward, done, info = self.env.step(self.normalized_action, is_test=True)
                elif self.comment == "elastic":
                    # self.env.envs.set_pose(self.state[:,:3], self.state[:,3:7])
                    self.env.envs.reset(
                            state=(self.state[:,:3], self.state[:,3:7], self.state[:,7:10], self.state[:,10:13])
                    )
                    self.env.envs.sceneManager.step()
                    self.env.envs.sceneManager.set_pose(
                        position=self.state[:,:3],
                        rotation=self.state[:,3:7]
                    )
                    self.env.envs.update_observation()
                    self.env.get_full_observation()
                    
                step_count += 1
                
                # Collect and process data for SaveNode
                self.collect_and_process()
                
                # Reset action flag
                self.action_ready = False
                self.state_ready = False
                self.pending_action = None
                
                if (start_obj_pos - env.envs.dynamic_object_position[0]).norm()<=0.2 and len(self.save_node.reward_all) > 30+prev_len:
                    n_round += 1
                    prev_len = len(self.save_node.reward_all)

                # Check if environment is done
                if n_round==1 and step_count  >= 300:
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

    def process_state(self):
        with self.state_lock:
            state_tensor = torch.zeros(self.num_agent, 13)
            for i in range(self.num_agent):
                self.state[i, 0] = self.state_data[i].pose.pose.position.x
                self.state[i, 1] = self.state_data[i].pose.pose.position.y
                self.state[i, 2] = self.state_data[i].pose.pose.position.z
                self.state[i, 3] = self.state_data[i].pose.pose.orientation.w
                self.state[i, 4] = self.state_data[i].pose.pose.orientation.x
                self.state[i, 5] = self.state_data[i].pose.pose.orientation.y
                self.state[i, 6] = self.state_data[i].pose.pose.orientation.z
                self.state[i, 7] = self.state_data[i].twist.twist.linear.x
                self.state[i, 8] = self.state_data[i].twist.twist.linear.y
                self.state[i, 9] = self.state_data[i].twist.twist.linear.z
                self.state[i, 10] = self.state_data[i].twist.twist.angular.x
                self.state[i, 11] = self.state_data[i].twist.twist.angular.y
                self.state[i, 12] = self.state_data[i].twist.twist.angular.z

                self.state[i, 3:7] = self.state[i, 3:7] / self.state[i, 3:7].norm()

        self.state_data = [None] * self.num_agent
        self.state_ready = True
        return self.state
        
    def process_action(self):
        """
        订阅action并提取position和yaw，组成n*4的tensor并return
        """
        with self.action_lock:
            if self.comment == "elastic":
                # 提取position和yaw组成n*4的tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, 0] = self.action_data[i]['yaw']
                        action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['position'])
                return action_tensor
            elif self.comment == "BPTT":
                # 提取z_acc和bodyrate组成n*4的tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, 0] = self.action_data[i]['z_acc']
                        action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])
            elif self.comment == "FSC":
                # Extract collective_thrust and bodyrate components as n*4 tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, 0] = self.action_data[i]['collective_thrust']  # z-thrust
                        action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])  # [roll, pitch, yaw] rates
                
        self.action_data = [None] * self.num_agent

        return action_tensor

    def publish_env_status(self, is_count=True, freq=10):
        """
        发布所有环境信息
        """
        try:
            self.publish_drone_state()
            self.publish_target_odom()
            self.publish_pointcloud()
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

        for i in range(self.num_agent):
            state = drone_states[i]  # shape: (13,) [pos(3), quat_wxyz(4), vel(3), angular_vel(3)]
            
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
                

    def publish_target_odom(self):
        """
        发布目标位姿
        从self.envs.dynamic_object_position获取目标位置
        """
        target_position = self.env.dynamic_object_position[0][0]  # 取第一个作为target位置

        if self.comment == "FSC":
            # FSC mode: publish target as PoseStamped
            if self.target_pub is not None:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = self.world_frame

                pose_msg.pose.position.x = target_position[0]
                pose_msg.pose.position.y = target_position[1]
                pose_msg.pose.position.z = target_position[2]

                # 默认朝向
                pose_msg.pose.orientation.w = 1.0

                self.target_pub.publish(pose_msg)
                
        elif self.comment == "BPTT":
            # Standard mode: publish target as Odometry
            if self.target_pub is not None:
                odom_msg = Odometry()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = self.world_frame
                odom_msg.child_frame_id = "target"

                odom_msg.pose.pose.position.x = target_position[0]
                odom_msg.pose.pose.position.y = target_position[1]
                odom_msg.pose.pose.position.z = target_position[2]

                # 默认朝向
                odom_msg.pose.pose.orientation.w = 1.0

                self.target_pub.publish(odom_msg)
                
        else:
                        # Standard mode: publish target as Odometry
            if self.target_pub is not None:
                odom_msg = Odometry()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = self.world_frame
                odom_msg.child_frame_id = "target"

                odom_msg.pose.pose.position.x = target_position[0]
                odom_msg.pose.pose.position.y = target_position[1]
                odom_msg.pose.pose.position.z = target_position[2]

                # 默认朝向
                odom_msg.pose.pose.orientation.w = 1.0

                self.target_pub.publish(odom_msg)


    def publish_pointcloud(self):
        """
        发布点云数据
        elastic: 使用PointCloud2 (单个演示点)
        BPTT: 使用PointCloud (单个演示点)
        FSC: 自定义格式的PointCloud (目标+角点)
        """
        if self.comment == "elastic":
            # 构造一个包含单点的 PointCloud2
            import struct
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.world_frame

            # 单个示例点
            x, y, z = 100.0, 100.0, 100.0
            points = [(x, y, z)]

            fields = [
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            ]

            point_step = 12  # 3 * float32
            data_bytes = b''.join([struct.pack('fff', *p) for p in points])

            pc2_msg = PointCloud2()
            pc2_msg.header = header
            pc2_msg.height = 1
            pc2_msg.width = len(points)
            pc2_msg.fields = fields
            pc2_msg.is_bigendian = False
            pc2_msg.point_step = point_step
            pc2_msg.row_step = point_step * len(points)
            pc2_msg.data = data_bytes
            pc2_msg.is_dense = True

            self.pointcloud_pub.publish(pc2_msg)

        elif self.comment == "BPTT":
            # BPTT 仍保持使用 PointCloud
            point_cloud = PointCloud()
            point_cloud.header.stamp = rospy.Time.now()
            point_cloud.header.frame_id = self.world_frame

            example_point = Point32()
            example_point.x = 100.0
            example_point.y = 100.0
            example_point.z = 100.0
            point_cloud.points = [example_point]
            self.pointcloud_pub.publish(point_cloud)

        elif self.comment == "FSC":
            # FSC mode: publish target as PointCloud in camera frame
            target_position = self.env.dynamic_object_position[0][0]  # 取第一个作为target位置
            
            if self.env.state is None:
                rospy.logwarn("Drone state not available for camera frame conversion")
                return

            # 获取第一个无人机的状态(FSC只处理单无人机)
            drone_state = self.env.state[0]  # num_agent * 13
            
            # 无人机位置和姿态
            drone_pos = np.array([drone_state[0], drone_state[1], drone_state[2]])
            drone_quat = np.array([drone_state[4], drone_state[5], drone_state[6], drone_state[3]])  # xyzw格式
            
            # 转换到相机坐标系 - Use FSC-specific transformation
            target_camera = self.world_to_camera_frame_fsc(target_position, drone_pos, drone_quat)
            
            # 创建PointCloud消息
            point_cloud = PointCloud()
            point_cloud.header.stamp = rospy.Time.now()
            # FSC AprilFake doesn't set frame_id - explicitly set to empty string to match expected format
            point_cloud.header.frame_id = ""
            
            # 添加目标点 - Match FSC AprilFake format with 2 points minimum
            # Point 1: Camera frame target position
            camera_point = Point32()
            camera_point.x = float(target_camera[0])
            camera_point.y = float(target_camera[1])
            camera_point.z = float(target_camera[2])
            
            # Point 2: Global frame target position (required by FSC)
            global_point = Point32()
            global_point.x = float(target_position[0])
            global_point.y = float(target_position[1])
            global_point.z = float(target_position[2])
            
            # Points 3-6: AprilTag corner points in image coordinates (fake but necessary)
            # Based on official FSC data: corners around center of image
            center_u, center_v = 376.0, 240.0  # Typical image center
            tag_size = 12.0  # Half-width of AprilTag in pixels
            
            corner1 = Point32()
            corner1.x = center_u - tag_size
            corner1.y = center_v + tag_size  
            corner1.z = 1.0
            
            corner2 = Point32()
            corner2.x = center_u + tag_size
            corner2.y = center_v + tag_size
            corner2.z = 1.0
            
            corner3 = Point32()
            corner3.x = center_u + tag_size
            corner3.y = center_v - tag_size
            corner3.z = 1.0
            
            corner4 = Point32()
            corner4.x = center_u - tag_size
            corner4.y = center_v - tag_size
            corner4.z = 1.0
            
            point_cloud.points = [camera_point, global_point, corner1, corner2, corner3, corner4]
            
            # Debug: Log FSC target data for troubleshooting "point unavailable" issue
            # rospy.loginfo_throttle(2.0, f"FSC Target Debug - Camera frame: [{target_camera[0]:.3f}, {target_camera[1]:.3f}, {target_camera[2]:.3f}] | "
            #                             f"Global frame: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] | "
            #                             f"Drone pos: [{drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f}] | "
            #                             f"Points count: {len(point_cloud.points)} | "
            #                             f"Camera Z>0: {target_camera[2] > 0}")

            self.pointcloud_pub.publish(point_cloud)

    def world_to_camera_frame_fsc(self, target_world, drone_pos, drone_quat):
        """
        FSC-specific camera frame transformation using exact AprilFake projToCam logic
        """
        # FSC camera extrinsics from aprilfake_params.yaml
        t_B_C = np.array([0.0, 0.0, 0.15])  # Camera 15cm UP from body center
        q_B_C = R.from_quat([-0.5, 0.5, -0.5, 0.5])  # [x,y,z,w] format, FSC uses [w,x,y,z]=[0.5,-0.5,0.5,-0.5]
        
        # Step 1: Transform target to drone body frame
        # p_lb_b = orient.inverse() * (target - pos)
        drone_rotation = R.from_quat(drone_quat)  # xyzw format
        p_lb_b = drone_rotation.inv().apply(target_world - drone_pos)
        
        # Step 2: Transform from body frame to camera frame 
        # p_lc_c = q_B_C_.inverse() * (p_lb_b - t_B_C_)
        target_camera = q_B_C.inv().apply(p_lb_b - t_B_C)
        
        return target_camera


def get_env(velocity, trajectory, distance=3.0, algorithm="BPTT"):
    proj_path = os.path.dirname(os.path.abspath(__file__)).split("obj_track")[0] + "obj_track/"
    env_config = load_yaml_config(proj_path + f'exps/vary_v/env_cfgs/objTracking.yaml')
    env_config["eval_env"]["scene_kwargs"]["obj_settings"]["path"] = \
        proj_path + 'exps/vary_v/configs/obj/' + trajectory

    # Modify action type for elastic mode
    if algorithm == "elastic":
        env_config["eval_env"]["dynamics_kwargs"] = env_config["eval_env"].get("dynamics_kwargs", {})
        env_config["eval_env"]["dynamics_kwargs"]["action_type"] = "position"
        print(f"[INFO] Modified action_type to 'position' for elastic mode")
    # env_config["eval_env"]["visual"] = False
    change_v_in_json(trajectory, velocity, distance)

    eval_env = env_alias["objTracking"](
        **env_config["eval_env"]
    )

    return eval_env


if __name__ == '__main__':
    try:
        parser = parse_args()
        args = parser.parse_args(rospy.myargv()[1:])

        env = get_env(args.velocity, args.traj, distance=3.0, algorithm=args.comment)
    
        current_abs_path = os.path.abspath(__file__)
        obj_track_path = current_abs_path.split('obj_track')[0] + 'obj_track'
        save_path = f"{obj_track_path}/exps/vary_v/saved/objTracking/test/{args.traj}_{args.velocity}_{args.comment}_Dis3.0"

        assert args.comment in ["BPTT", "elastic", "FSC"]
        print(f"Environment created: {env.__class__}")
        node = ROSIndepWrapper(env, path=save_path, comment=args.comment)
        
        # Start main loop in the main thread
        node.main_loop() 
        

    except rospy.ROSInterruptException:
        pass
