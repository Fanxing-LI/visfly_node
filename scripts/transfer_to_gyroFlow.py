#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
import json
import os
import argparse
from sensor_msgs.msg import Image, Imu, CameraInfo
from cv_bridge import CvBridge
from tqdm import tqdm
import struct


def extract_rosbag_to_gyroflow(bag_path, output_dir, image_topic="/camera/color/image_raw", 
                              imu_topic="/mavros/imu/data", camera_info_topic="/camera/color/camera_info"):
    """
    从ROS bag中提取图像和IMU数据，转换为GyroFlow格式
    
    Args:
        bag_path: ROS bag文件路径
        output_dir: 输出目录
        image_topic: 图像topic名称
        imu_topic: IMU topic名称
        camera_info_topic: 相机信息topic名称
    """
    if not os.path.exists(bag_path):
        print(f"Error: Bag file {bag_path} not found!")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建cv_bridge
    bridge = CvBridge()
    
    # 用于存储数据
    images = []
    imu_data = []
    image_timestamps = []
    imu_timestamps = []
    camera_info = None
    
    print(f"Opening bag file: {bag_path}")
    
    # 读取bag文件
    with rosbag.Bag(bag_path, 'r') as bag:
        # 获取bag信息
        info = bag.get_type_and_topic_info()
        topics = info.topics
        
        print(f"Available topics:")
        for topic, info in topics.items():
            print(f"  {topic}: {info.msg_type} ({info.message_count} messages)")
        
        # 检查required topics是否存在
        if image_topic not in topics:
            print(f"Warning: Image topic {image_topic} not found in bag!")
            return
        if imu_topic not in topics:
            print(f"Warning: IMU topic {imu_topic} not found in bag!")
            return
        
        # 计算总消息数用于进度条
        total_msgs = topics[image_topic].message_count + topics[imu_topic].message_count
        
        print(f"Extracting data from bag...")
        
        # 读取消息
        with tqdm(total=total_msgs, desc="Processing messages") as pbar:
            for topic, msg, t in bag.read_messages(topics=[image_topic, imu_topic, camera_info_topic]):
                timestamp = t.to_sec()
                
                if topic == image_topic:
                    try:
                        # 转换ROS图像到OpenCV格式
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                        images.append(cv_image)
                        image_timestamps.append(timestamp)
                    except Exception as e:
                        print(f"Error converting image: {e}")
                        continue
                
                elif topic == imu_topic:
                    # 提取IMU数据 (角速度和加速度)
                    imu_sample = {
                        'timestamp': timestamp,
                        'gyro_x': msg.angular_velocity.x,
                        'gyro_y': msg.angular_velocity.y,
                        'gyro_z': msg.angular_velocity.z,
                        'accel_x': msg.linear_acceleration.x,
                        'accel_y': msg.linear_acceleration.y,
                        'accel_z': msg.linear_acceleration.z
                    }
                    imu_data.append(imu_sample)
                    imu_timestamps.append(timestamp)
                
                elif topic == camera_info_topic and camera_info is None:
                    # 只需要一条camera_info消息
                    camera_info = msg
                
                pbar.update(1)
    
    print(f"Extracted {len(images)} images and {len(imu_data)} IMU samples")
    
    if len(images) == 0:
        print("No images found!")
        return
    if len(imu_data) == 0:
        print("No IMU data found!")
        return
    
    # 保存视频文件 (GyroFlow支持的格式)
    video_path = os.path.join(output_dir, "video.mp4")
    save_video(images, image_timestamps, video_path)
    
    # 保存IMU数据为GyroFlow格式
    imu_path = os.path.join(output_dir, "gyro_data.csv")
    save_imu_gyroflow_format(imu_data, imu_path)
    
    # 生成镜头配置文件
    lens_profile_path = os.path.join(output_dir, "lens_profile.json")
    save_lens_profile(camera_info, images[0] if images else None, lens_profile_path)
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    save_metadata(image_timestamps, imu_timestamps, len(images), len(imu_data), metadata_path)
    
    print(f"Conversion complete!")
    print(f"Video saved to: {video_path}")
    print(f"IMU data saved to: {imu_path}")
    print(f"Lens profile saved to: {lens_profile_path}")
    print(f"Metadata saved to: {metadata_path}")


def save_video(images, timestamps, output_path, fps=30):
    """保存图像序列为视频文件"""
    if len(images) == 0:
        return
    
    height, width = images[0].shape[:2]
    
    # 使用mp4格式，GyroFlow兼容性更好
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Saving video ({len(images)} frames) to {output_path}")
    
    for img in tqdm(images, desc="Writing video"):
        out.write(img)
    
    out.release()


def save_imu_gyroflow_format(imu_data, output_path):
    """保存IMU数据为GyroFlow兼容的CSV格式"""
    print(f"Saving IMU data ({len(imu_data)} samples) to {output_path}")
    
    with open(output_path, 'w') as f:
        # GyroFlow CSV header
        f.write("timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n")
        
        for sample in tqdm(imu_data, desc="Writing IMU data"):
            # 时间戳转换为微秒 (GyroFlow格式)
            timestamp_us = int(sample['timestamp'] * 1000000)
            f.write(f"{timestamp_us},{sample['gyro_x']},{sample['gyro_y']},{sample['gyro_z']},"
                   f"{sample['accel_x']},{sample['accel_y']},{sample['accel_z']}\n")


def save_lens_profile(camera_info, sample_image, output_path):
    """生成GyroFlow兼容的镜头配置文件"""
    if camera_info is None:
        print("Warning: No camera info found, creating default lens profile")
        # 创建默认配置
        if sample_image is not None:
            height, width = sample_image.shape[:2]
        else:
            width, height = 640, 480
        
        lens_profile = {
            "name": "Unknown Camera",
            "note": "Generated from ROS bag without camera_info",
            "calibrated_by": "transfer_to_gyroFlow.py",
            "camera_brand": "Unknown",
            "camera_model": "Unknown",
            "lens_model": "Unknown",
            "camera_setting": {
                "fps": 30,
                "width": width,
                "height": height
            },
            "calibration_dimension": {
                "w": width,
                "h": height
            },
            "orig_dimension": {
                "w": width,
                "h": height
            },
            "output_dimension": {
                "w": width,
                "h": height
            },
            "identifier": f"unknown_{width}x{height}",
            "calibrator_version": "1.0",
            "date": "2024-01-01",
            "fisheye_params": {
                "camera_matrix": [
                    [width * 0.8, 0, width / 2],
                    [0, height * 0.8, height / 2],
                    [0, 0, 1]
                ],
                "distortion_coeffs": [0, 0, 0, 0],
                "radial_distortion_limit": None
            }
        }
    else:
        # 从camera_info生成配置
        width = camera_info.width
        height = camera_info.height
        
        # 提取相机矩阵
        K = camera_info.K
        camera_matrix = [
            [K[0], K[1], K[2]],
            [K[3], K[4], K[5]],
            [K[6], K[7], K[8]]
        ]
        
        # 提取畸变系数
        D = camera_info.D
        distortion_coeffs = list(D) if len(D) > 0 else [0, 0, 0, 0, 0]
        
        lens_profile = {
            "name": f"ROS Camera {width}x{height}",
            "note": f"Generated from ROS camera_info topic",
            "calibrated_by": "transfer_to_gyroFlow.py",
            "camera_brand": "ROS",
            "camera_model": camera_info.distortion_model if hasattr(camera_info, 'distortion_model') else "Unknown",
            "lens_model": "ROS Camera",
            "camera_setting": {
                "fps": 30,
                "width": width,
                "height": height
            },
            "calibration_dimension": {
                "w": width,
                "h": height
            },
            "orig_dimension": {
                "w": width,
                "h": height
            },
            "output_dimension": {
                "w": width,
                "h": height
            },
            "identifier": f"ros_camera_{width}x{height}",
            "calibrator_version": "1.0",
            "date": "2024-01-01",
            "fisheye_params": {
                "camera_matrix": camera_matrix,
                "distortion_coeffs": distortion_coeffs,
                "radial_distortion_limit": None
            }
        }
    
    print(f"Saving lens profile to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(lens_profile, f, indent=2)


def save_metadata(image_timestamps, imu_timestamps, num_images, num_imu_samples, output_path):
    """保存元数据信息"""
    metadata = {
        "video": {
            "frame_count": num_images,
            "start_timestamp": image_timestamps[0] if image_timestamps else 0,
            "end_timestamp": image_timestamps[-1] if image_timestamps else 0,
            "duration_sec": (image_timestamps[-1] - image_timestamps[0]) if len(image_timestamps) > 1 else 0
        },
        "imu": {
            "sample_count": num_imu_samples,
            "start_timestamp": imu_timestamps[0] if imu_timestamps else 0,
            "end_timestamp": imu_timestamps[-1] if imu_timestamps else 0,
            "duration_sec": (imu_timestamps[-1] - imu_timestamps[0]) if len(imu_timestamps) > 1 else 0
        },
        "sync_info": {
            "time_offset_sec": 0,  # 可根据需要调整
            "notes": "Extracted from ROS bag using transfer_to_gyroFlow.py"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert ROS bag to GyroFlow format")
    parser.add_argument("--bag_path", type=str,help="Path to ROS bag file", 
                        default="plots/lfx_data2042_withInfo.bag"
                        )
    parser.add_argument("--output_dir", type=str, help="Output directory for GyroFlow files",
                        default="plots/gyroflow_output"
                        )
    parser.add_argument("--image-topic", default="/camera/color/image_raw", 
                       help="Image topic name (default: /camera/color/image_raw)")
    parser.add_argument("--imu-topic", default="/mavros/imu/data",
                       help="IMU topic name (default: /mavros/imu/data)")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info",
                       help="Camera info topic name (default: /camera/color/camera_info)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Output video FPS (default: 30)")
    
    args = parser.parse_args()
    
    extract_rosbag_to_gyroflow(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        image_topic=args.image_topic,
        imu_topic=args.imu_topic,
        camera_info_topic=args.camera_info_topic
    )


if __name__ == "__main__":
    main()