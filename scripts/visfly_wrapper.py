#!/usr/bin/env python3
import rospy
import sys, os

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
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import quaternion_from_euler
import threading
import argparse
from exps.vary_v.run import main
from quadrotor_msgs.msg import PositionCommand, Command
from vision_msgs.msg import ControlCommand
import torch
from VisFly.utils.type import ACTION_TYPE
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32, Vector3
from scipy.spatial.transform import Rotation as R

# Topic name definitions

ODOM_TOPIC_PREFIX = "visfly/drone_{}/odom"
TARGET_ODOM_TOPIC = "visfly/target/odom"
POINTCLOUD_TOPIC = "visfly/env/pointcloud"

# Elastic Tracker topic names (when in elastic mode)
ELASTIC_ODOM_PREFIX = "/drone{}/odom"
ELASTIC_TARGET_TOPIC = "/target/odom"
ELASTIC_CMD_PREFIX = "/drone{}/position_cmd"

# FSC topic names (when in fsc mode)
FSC_ODOM_TOPIC = "/hummingbird/ground_truth/odometry"
FSC_TARGET_TOPIC = "/hummingbird/aprilfake/point"
FSC_CONTROL_TOPIC = "/hummingbird/autopilot/control_command"
FSC_MOTOR_TOPIC = "/hummingbird/command/motor_speed"

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="objTracking")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--traj", "-tr", type=str, default="1", )
    parser.add_argument("--velocity", "-v", type=float, default=3.0, )
    parser.add_argument("--num_agent", "-n", type=int, default=4, )
    return parser


class ROSEnvWrapper:
    def __init__(self, env, comment="elastic"):
        self.envs = env
        self.num_agent = self.envs.num_envs
        self.action_type = self.envs.envs.dynamics.action_type
        if comment == "elastic":
            ACTION_TOPIC_PREFIX = ELASTIC_CMD_PREFIX
            assert self.action_type == ACTION_TYPE.POSITION, f"current action type is {self.action_type}, but it should be 'position' for elastic"
        elif comment == "BPTT":
            assert self.action_type == ACTION_TYPE.BODYRATE, f"current action type is {self.action_type}, but it should be 'bodyrate' for BPTT"
            ACTION_TOPIC_PREFIX = "BPTT/drone_{}/action"
        elif comment == "fsc":
            ACTION_TOPIC_PREFIX = FSC_CONTROL_TOPIC
            assert self.action_type == ACTION_TYPE.BODYRATE, f"current action type is {self.action_type}, but it should be 'bodyrate' for fsc"
        self.comment = comment

        # Initialize ROS node
        rospy.init_node('visfly', anonymous=True)

        self._count = 0

        # Action data storage and lock for thread safety
        self.action_data = [None] * self.num_agent
        self.action_lock = threading.Lock()

        # Publishers
        self.drone_odom_pubs = []
        # Subscribers for action
        self.drone_action_subs = []

        for i in range(self.num_agent):
            # Publisher for odometry - use different topics for different modes
            if self.comment == "elastic":
                drone_odom_pub = rospy.Publisher(ELASTIC_ODOM_PREFIX.format(i), Odometry, queue_size=1)
            elif self.comment == "fsc":
                # FSC expects single drone odometry (only use first drone for now)
                # TODO: Handle multiple drones in FSC if needed
                if i == 0:
                    drone_odom_pub = rospy.Publisher(FSC_ODOM_TOPIC, Odometry, queue_size=1)
                else:
                    drone_odom_pub = None  # Only first drone for FSC
            else:
                drone_odom_pub = rospy.Publisher(ODOM_TOPIC_PREFIX.format(i), Odometry, queue_size=1)
            self.drone_odom_pubs.append(drone_odom_pub)

            # 添加action订阅者
            if self.comment == "elastic":
                action_sub = rospy.Subscriber(ACTION_TOPIC_PREFIX.format(i), PositionCommand, self._make_elastic_callback(i))
            elif self.comment == "BPTT":
                action_sub = rospy.Subscriber(ACTION_TOPIC_PREFIX.format(i), Command, self._make_action_callback(i))
            elif self.comment == "fsc":
                # FSC uses single drone, only subscribe for agent 0
                if i == 0:
                    action_sub = rospy.Subscriber(FSC_CONTROL_TOPIC, ControlCommand, self._make_fsc_callback(i))
                else:
                    action_sub = None  # FSC only supports single drone
            self.drone_action_subs.append(action_sub)

        # Target publisher - use different topic for different modes
        if self.comment == "elastic":
            self.target_odom_pub = rospy.Publisher(ELASTIC_TARGET_TOPIC, Odometry, queue_size=1)
        elif self.comment == "fsc":
            self.target_point_pub = rospy.Publisher(FSC_TARGET_TOPIC, PointCloud, queue_size=1)
            self.target_odom_pub = None  # FSC doesn't use target odometry
        else:
            self.target_odom_pub = rospy.Publisher(TARGET_ODOM_TOPIC, Odometry, queue_size=1)
        
        self.pointcloud_pub = rospy.Publisher(POINTCLOUD_TOPIC, PointCloud2, queue_size=1)
        
        # Elastic Tracker trigger publisher (for elastic mode)
        if self.comment == "elastic":
            self.trigger_pub = rospy.Publisher('/triger', PoseStamped, queue_size=1)
            self.planning_triggered = False
            # Timer to auto-trigger planning after initialization
            rospy.Timer(rospy.Duration(2.0), self._auto_trigger_planning, oneshot=True)

        # Frame IDs
        self.world_frame = "world"
        print("===========================================================================")
        rospy.loginfo(f"Visfly ROS Environment Wrapper initialized with {self.num_agent} agents in {self.comment} mode")
        if self.comment == "fsc":
            rospy.loginfo(f"FSC mode: Publishing drone odometry to: {FSC_ODOM_TOPIC}")
            rospy.loginfo(f"FSC mode: Publishing target points to: {FSC_TARGET_TOPIC}")
            rospy.loginfo(f"FSC mode: Subscribing to control commands from: {FSC_CONTROL_TOPIC}")
            
            # Delay IBVS configuration until after first points are sent
            self.fsc_ibvs_configured = False
        else:
            rospy.loginfo(f"Subscribing to action topics: {[ACTION_TOPIC_PREFIX.format(i) for i in range(self.num_agent)]}")
            rospy.loginfo(f"Publishing to topics: {[ODOM_TOPIC_PREFIX.format(i) for i in range(self.num_agent)]}")

    def reset(self, *args, **kwargs):
        """
        Reset the environment and clear action data.
        This method can be called to reset the environment state.
        """
        print("---------------------------debug:enter node reset")

        r=self.envs.reset(*args, **kwargs)
        # publish initial environment status
        print("---------------------------debug:leave node reset and finish env reset")
        self.publish_env_status()
        print("---------------------------debug:leave node reset")
        return r

    def predict(self, obs, deterministic=True):
        """
        publish current action
        """
        # 先检查一次是否已有数据
        with self.action_lock:
            if self.comment == "fsc":
                # FSC mode: only check agent 0 (single drone)
                all_actions_available = self.action_data[0] is not None
            else:
                # Multi-agent mode: check all agents
                all_actions_available = all(self.action_data[i] is not None for i in range(self.num_agent))

        publish_count = 0
        max_publish_count = 100
        # print("===================================debug")
        # print("===================================predicting")
        if not all_actions_available:
            print(publish_count)
            while True:
                rospy.sleep(0.1)
                # 在等待期间持续发布当前状态
                if publish_count < max_publish_count:
                    self.publish_env_status()
                    publish_count += 1
                    # import pdb; pdb.set_trace()
                else:
                    
                    raise RuntimeError("Maximum publish count reached without receiving all actions")

                rospy.loginfo("Waiting for action data...")

                # 重新检查是否所有agent都有有效的action数据（不持有锁）
                with self.action_lock:
                    if self.comment == "fsc":
                        # FSC mode: only check agent 0 (single drone)
                        all_actions_available = self.action_data[0] is not None
                    else:
                        # Multi-agent mode: check all agents
                        all_actions_available = all(self.action_data[i] is not None for i in range(self.num_agent))
                    # print(self.action_data)

                if all_actions_available:
                    break
            rospy.loginfo("All action data received, proceeding...")

        return self.subscribe_action()
    def _auto_trigger_planning(self, event):
        """Auto-trigger Elastic Tracker planning (only for elastic mode)"""
        if self.comment == "elastic" and not self.planning_triggered:
            rospy.loginfo("Auto-triggering Elastic Tracker planning...")
            trigger_msg = PoseStamped()
            trigger_msg.header.stamp = rospy.Time.now()
            trigger_msg.header.frame_id = "world"
            trigger_msg.pose.position.x = 0.0
            trigger_msg.pose.position.y = 0.0
            trigger_msg.pose.position.z = 0.0
            trigger_msg.pose.orientation.w = 1.0
            self.trigger_pub.publish(trigger_msg)
            self.planning_triggered = True
            rospy.loginfo("Planning trigger sent to Elastic Tracker!")

    def _make_action_callback(self, agent_id):
        def callback(msg):
            with self.action_lock:
                # Extract action data based on comment type
                if self.comment == "elastic":
                    # This shouldn't be called anymore since we use _make_elastic_callback
                    raise RuntimeError("Use _make_elastic_callback for elastic mode")
                elif self.comment == "BPTT":
                    # 对于Command消息：thrust使用thrust字段，bodyrate使用angularVel
                    self.action_data[agent_id] = {
                        'z_acc': msg.thrust,  # 使用thrust字段（float64）
                        'bodyrate': [msg.angularVel.x, msg.angularVel.y, msg.angularVel.z]  # 使用角速度
                    }
                elif self.comment == "fsc":
                    # FSC uses dedicated _make_fsc_callback, this shouldn't be called
                    raise RuntimeError("Use _make_fsc_callback for FSC mode")
        return callback

    def _make_fsc_callback(self, agent_id):
        """Create FSC-specific action callback for vision_msgs/ControlCommand"""
        def callback(msg):
            with self.action_lock:
                # FSC control limits from rotors_control_params.yaml
                MIN_THRUST = 1.0   # N
                MAX_THRUST = 20.0  # N
                MAX_BODYRATE_XY = 3.0  # rad/s
                MAX_BODYRATE_Z = 1.5   # rad/s
                
                # Normalize thrust: [1.0, 20.0] -> [-1, 1]
                normalized_thrust = 2 * (msg.collective_thrust - MIN_THRUST) / (MAX_THRUST - MIN_THRUST) - 1
                normalized_thrust = max(-1.0, min(1.0, normalized_thrust))  # Clamp to [-1, 1]
                # TODO: 带上质量，归一化到[-1, 1]
                # Normalize bodyrates: [-max_rate, max_rate] -> [-1, 1]
                normalized_roll_rate = max(-1.0, min(1.0, msg.bodyrates.x / MAX_BODYRATE_XY))
                normalized_pitch_rate = max(-1.0, min(1.0, msg.bodyrates.y / MAX_BODYRATE_XY))
                normalized_yaw_rate = max(-1.0, min(1.0, msg.bodyrates.z / MAX_BODYRATE_Z))
                
                # Store normalized values
                self.action_data[agent_id] = {
                    'collective_thrust': normalized_thrust,
                    'bodyrate': [normalized_roll_rate, normalized_pitch_rate, normalized_yaw_rate]
                }
                
                # Debug: Log when FSC commands are received (raw and normalized)
                if agent_id == 0:  # Only log for first agent to avoid spam
                    rospy.loginfo_throttle(2.0, f"FSC raw: thrust={msg.collective_thrust:.3f}, "
                                                f"rates=[{msg.bodyrates.x:.3f}, {msg.bodyrates.y:.3f}, {msg.bodyrates.z:.3f}] | "
                                                f"normalized: thrust={normalized_thrust:.3f}, "
                                                f"rates=[{normalized_roll_rate:.3f}, {normalized_pitch_rate:.3f}, {normalized_yaw_rate:.3f}]")
        return callback

    def subscribe_action(self):
        """
        订阅action并提取position和yaw，组成n*4的tensor并return
        """
        with self.action_lock:
            if self.comment == "elastic":
                raise NotImplementedError
                # 提取position和yaw组成n*4的tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, :3] = torch.tensor(self.action_data[i]['position'])
                        action_tensor[i, 3] = self.action_data[i]['yaw']  # yaw的dim是0而不是3
                # 清空action_data
                self.action_data = [None] * self.num_agent
                return action_tensor
            elif self.comment == "BPTT":
                # 提取z_acc和bodyrate组成n*4的tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, 0] = self.action_data[i]['z_acc']
                        action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])
                # print("===================================predicting")
                self.action_data = [None] * self.num_agent
            elif self.comment == "fsc":
                # Extract collective_thrust and bodyrate components as n*4 tensor
                action_tensor = torch.zeros(self.num_agent, 4)
                for i in range(self.num_agent):
                    if self.action_data[i] is not None:
                        action_tensor[i, 0] = self.action_data[i]['collective_thrust']  # z-thrust
                        action_tensor[i, 1:4] = torch.tensor(self.action_data[i]['bodyrate'])  # [roll, pitch, yaw] rates
                # Clear action data
                self.action_data = [None] * self.num_agent

        return action_tensor

    def publish_env_status(self, is_count=True):
        """
        发布所有环境信息
        """
        self.publish_drone_state()
        self.publish_target_odom()
        self.publish_pointcloud()
        # print(f"debug pushlished environment status.{self._count}")
        # wait 0.03 s
        if is_count:
            # rospy.sleep(0.03)
            self._count += 1
            if self._count % 10 == 0:
                rospy.loginfo(f"Published environment status at count {self._count}")
            # print("Published environment status.")
        # print("debug pushlished environment status. leaving")

    def publish_drone_state(self):
        """
        发布无人机状态信息
        从self.envs.state获取状态：num_agent*13 (pos, quaternion, vel, angular_vel)
        """
        # 获取当前环境状态
        if not hasattr(self.envs, 'state') or self.envs.state is None:
            rospy.logwarn("Environment state not available")
            return

        # Remove FSC position offset - it was causing wrong camera coordinates
        FSC_POSITION_OFFSET = [0.0, 0.0, 0.0]

        drone_states = self.envs.state  # shape: (num_agent, 13)

        for i in range(self.num_agent):
            if i < len(drone_states):
                state = drone_states[i]  # shape: (13,) [pos(3), quat_wxyz(4), vel(3), angular_vel(3)]
                odom_msg = Odometry()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = self.world_frame
                odom_msg.child_frame_id = f"drone_{i}"

                # 位置 (0:3) - Apply FSC offset if needed
                odom_msg.pose.pose.position.x = state[0] - FSC_POSITION_OFFSET[0]
                odom_msg.pose.pose.position.y = state[1] - FSC_POSITION_OFFSET[1]
                odom_msg.pose.pose.position.z = state[2] - FSC_POSITION_OFFSET[2]

                # 四元数姿态 (3:7) - 状态格式是wxyz，ROS格式是xyzw
                odom_msg.pose.pose.orientation.x = state[4]  # x from wxyz[1]
                odom_msg.pose.pose.orientation.y = state[5]  # y from wxyz[2]
                odom_msg.pose.pose.orientation.z = state[6]  # z from wxyz[3]
                odom_msg.pose.pose.orientation.w = state[3]  # w from wxyz[0]

                # 线速度 (7:10)
                odom_msg.twist.twist.linear.x = state[7]
                odom_msg.twist.twist.linear.y = state[8]
                odom_msg.twist.twist.linear.z = state[9]

                # 角速度 (10:13)
                odom_msg.twist.twist.angular.x = state[10]
                odom_msg.twist.twist.angular.y = state[11]
                odom_msg.twist.twist.angular.z = state[12]

                # Only publish if publisher exists (FSC mode only uses first drone)
                if self.drone_odom_pubs[i] is not None:
                    self.drone_odom_pubs[i].publish(odom_msg)

    def publish_target_odom(self):
        """
        发布目标位姿
        从self.envs.dynamic_object_position获取目标位置
        """
        # 获取目标位置
        if not hasattr(self.envs, 'dynamic_object_position') or self.envs.dynamic_object_position is None:
            rospy.logwarn("Dynamic object position not available")
            return

        # dynamic_object_position是一个len=num_agent的list，取第一个作为target位置
        if len(self.envs.dynamic_object_position) == 0:
            rospy.logwarn("Dynamic object position list is empty")
            return

        target_position = self.envs.dynamic_object_position[0][0]  # 取第一个作为target位置

        if self.comment == "fsc":
            # FSC mode: publish target as PointCloud in camera frame
            self.publish_target_pointcloud(target_position)
        else:
            # Standard mode: publish target as Odometry
            if self.target_odom_pub is not None:
                odom_msg = Odometry()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = self.world_frame
                odom_msg.child_frame_id = "target"

                odom_msg.pose.pose.position.x = target_position[0]
                odom_msg.pose.pose.position.y = target_position[1]
                odom_msg.pose.pose.position.z = target_position[2]

                # 默认朝向
                odom_msg.pose.pose.orientation.w = 1.0

                self.target_odom_pub.publish(odom_msg)

    def publish_target_pointcloud(self, target_position):
        """
        发布FSC格式的目标点云
        将世界坐标系中的目标位置转换为相机坐标系的点云
        """
        if not hasattr(self.envs, 'state') or self.envs.state is None:
            rospy.logwarn("Drone state not available for camera frame conversion")
            return

        # 获取第一个无人机的状态(FSC只处理单无人机)
        drone_state = self.envs.state[0]  # num_agent * 13
        
        # No position offset needed - was causing wrong camera coordinates
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
        rospy.loginfo_throttle(2.0, f"FSC Target Debug - Camera frame: [{target_camera[0]:.3f}, {target_camera[1]:.3f}, {target_camera[2]:.3f}] | "
                                    f"Global frame: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] | "
                                    f"Drone pos: [{drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f}] | "
                                    f"Points count: {len(point_cloud.points)} | "
                                    f"Camera Z>0: {target_camera[2] > 0}")
        
        self.target_point_pub.publish(point_cloud)
        
        # Configure FSC IBVS mode after first point is published (delayed activation)
        if not self.fsc_ibvs_configured:
            rospy.loginfo("First target point published - now configuring FSC for IBVS mode")
            self._configure_fsc_ibvs_mode()
            self.fsc_ibvs_configured = True

    def world_to_camera_frame(self, target_world, drone_pos, drone_quat):
        """
        将世界坐标系中的目标点转换为相机坐标系
        相机安装在无人机前方，朝前看
        """
        # 相机相对于无人机的偏移(前方15cm)
        camera_offset = np.array([0.15, 0.0, 0.0])
        
        # 相机相对于无人机的旋转(相机坐标系: X前 Y左 Z上)
        # 无人机坐标系: X前 Y左 Z上
        # 相机坐标系相对于无人机: 绕Z轴转-90度，然后绕Y轴转-90度
        camera_rot_relative = R.from_euler('xyz', [0, -90, -90], degrees=True)
        
        # 无人机姿态旋转
        drone_rotation = R.from_quat(drone_quat)  # xyzw格式
        
        # 计算相机在世界坐标系中的位置和姿态
        camera_pos_world = drone_pos + drone_rotation.apply(camera_offset)
        camera_rotation_world = drone_rotation * camera_rot_relative
        
        # 将目标从世界坐标系转换为相机坐标系
        target_relative = target_world - camera_pos_world
        target_camera = camera_rotation_world.inv().apply(target_relative)
        
        return target_camera

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

    def _configure_fsc_ibvs_mode(self):
        """
        Configure FSC autopilot for IBVS mode automatically
        This replaces the manual rqt_reconfigure step
        """
        rospy.loginfo("Configuring FSC for IBVS mode...")
        
        try:
            # Wait a moment for FSC to fully initialize
            # rospy.sleep(1.0)
            
            # Set FSC parameters for IBVS tracking
            rospy.set_param('/hummingbird/fsc_rotors_node/enable_moving_target_track', True)
            rospy.set_param('/hummingbird/param_tuner_node/feature_track', True)  
            rospy.set_param('/hummingbird/param_tuner_node/reference_distance', 3.0)
            
            rospy.loginfo("FSC parameters set: enable_moving_target_track=True, reference_distance=3.0")
            
            # Import required service types
            from dynamic_reconfigure.srv import Reconfigure
            from dynamic_reconfigure.msg import Config, DoubleParameter, BoolParameter
            
            # Wait for param_tuner service to be available
            service_name = '/hummingbird/param_tuner_node/set_parameters'
            rospy.loginfo(f"Waiting for service: {service_name}")
            rospy.wait_for_service(service_name, timeout=10.0)
            
            # Create service client and configure parameters
            set_params = rospy.ServiceProxy(service_name, Reconfigure)
            
            # Create parameter configuration
            config = Config()
            config.doubles = [DoubleParameter(name='reference_distance', value=3.0)]
            config.bools = [BoolParameter(name='feature_track', value=True)]
            
            # Send parameter update
            result = set_params(config)
            rospy.loginfo("FSC IBVS mode activated successfully via param_tuner service")
            
        except Exception as e:
            rospy.logwarn(f"Failed to configure FSC IBVS mode: {e}")
            rospy.logwarn("FSC may remain in position control mode - manual rqt_reconfigure needed")

    def publish_pointcloud(self):
        """
        发布点云数据
        发布一个包含单个远点的示例点云
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        # 创建一个包含单个远点的示例点云
        example_points = [[100.0, 100.0, 100.0]]  # 一个很远的点作为示例

        pc2_msg = pc2.create_cloud(header, fields, example_points)
        self.pointcloud_pub.publish(pc2_msg)

    @property
    def num_agent(self):
        return self._num_agent

    @num_agent.setter
    def num_agent(self, value):
        self._num_agent = value


if __name__ == '__main__':
    try:
        parser = parse_args()
        args = parser.parse_args(rospy.myargv()[1:])

        env_kwargs = {
            'traj': args.traj,
            'velocity': args.velocity,
            "comment": args.comment,
        }
        assert args.comment in ["BPTT", "elastic", "fsc"]
        print("------------------debug:enter the node")
        env = main(traj=env_kwargs["traj"],
                   velocity=env_kwargs["velocity"],
                   comment=env_kwargs["comment"],
                   ROS_wrapper=ROSEnvWrapper,
                   debug=False,
                   )

        print(f"Environment created: {env.__class__}")
        # node = ROSEnvWrapper(env_kwargs)

        # Keep the node running
        # while True:
        #     rospy.sleep(0.01)
        #     env._publish_ros_data()

    except rospy.ROSInterruptException:
        pass
