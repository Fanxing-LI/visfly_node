#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry

# Added configuration variables
use_pos_target = False
target_vel = [8.0, 0.0, -2.0]
target_pos = [60.0, 0.0, -10.0]
max_velocity = 8.0
    
def commander():
    # Initialize the ROS node
    rospy.init_node('commander_node', anonymous=True)

    # Set parameters to share with other nodes (e.g., BPTT_policy)
    rospy.set_param('/visfly/max_velocity', max_velocity)

    # Create a publisher for the target odometry
    # Topic name inferred from context, typically 'target_odom' or similar
    pub = rospy.Publisher('/target/odom', Odometry, queue_size=10)

    # Set the loop rate to 100Hz
    rate = rospy.Rate(100) 

    # Watchdog timer initialization
    last_watchdog_time = rospy.Time.now()

    while not rospy.is_shutdown():
        # Create the Odometry message
        odom_msg = Odometry()
        
        # Populate the header
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "world"  # Or appropriate frame
        odom_msg.child_frame_id = "target_base_link"

        # Populate pose and twist with data here
        if use_pos_target:
            odom_msg.pose.pose.position.x = target_pos[0]
            odom_msg.pose.pose.position.y = target_pos[1]
            odom_msg.pose.pose.position.z = target_pos[2]
        else:
            # Fallback or alternative mode (e.g. keeping 0 or integrating velocity)
            odom_msg.pose.pose.position.x = 0.0
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.position.z = 0.0

        odom_msg.twist.twist.linear.x = target_vel[0]
        odom_msg.twist.twist.linear.y = target_vel[1]
        odom_msg.twist.twist.linear.z = target_vel[2]
        
        odom_msg.pose.pose.orientation.w = 1.0

        # Publish the message
        pub.publish(odom_msg)

        # Watchdog: Log every 1 second to verify operation
        if (rospy.Time.now() - last_watchdog_time).to_sec() > 1.0:
            mode_str = "Position" if use_pos_target else "Velocity"
            target_str = str(target_pos) if use_pos_target else str(target_vel)
            rospy.loginfo("Commander Watchdog: Active. Mode: %s. Target: %s", mode_str, target_str)
            last_watchdog_time = rospy.Time.now()

        # Sleep to maintain the loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        commander()
    except rospy.ROSInterruptException:
        pass
