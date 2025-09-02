#!/usr/bin/env python3

import rospy
import numpy as np
import threading
import os
import yaml
from nav_msgs.msg import Odometry
from collections import deque

TARGET_ODOM_TOPIC = "vicon/objtarget/odom"

class EKF9D:
    """9D (p,v,a) constant-acceleration EKF optimized for low-power processors like N100."""

    def __init__(self, q_pos=0.01, q_vel=0.1, q_acc=1.0, r_pos=0.1, r_vel=0.2):
        # Force CPU usage to avoid GPU overhead on N100
        self.device = 'cpu'
        self.state_dim = 9              # [px,py,pz,vx,vy,vz,ax,ay,az]
        self.obs_dim = 6               # [px,py,pz,vx,vy,vz]
        
        # Use numpy for better performance on CPU
        self.x = np.zeros(self.state_dim, dtype=np.float32)
        self.P = np.eye(self.state_dim, dtype=np.float32) * 10
        
        # Pre-compute process noise matrix
        self.Q_base = np.diag(np.array([
            q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel,
            q_acc, q_acc, q_acc
        ], dtype=np.float32))
        
        # Pre-compute measurement noise matrix
        self.R = np.diag(np.array([
            r_pos, r_pos, r_pos,
            r_vel, r_vel, r_vel
        ], dtype=np.float32))
        
        # Pre-allocate matrices to avoid memory allocation during runtime
        self.F = np.eye(self.state_dim, dtype=np.float32)
        self.H = np.zeros((self.obs_dim, self.state_dim), dtype=np.float32)
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        self.H[3,3] = self.H[4,4] = self.H[5,5] = 1.0
        
        # Pre-allocate temporary arrays for calculations
        self.temp_state = np.zeros(self.state_dim, dtype=np.float32)
        self.temp_P = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        self.temp_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self.temp_y = np.zeros(self.obs_dim, dtype=np.float32)
        self.temp_S = np.zeros((self.obs_dim, self.obs_dim), dtype=np.float32)
        self.temp_K = np.zeros((self.state_dim, self.obs_dim), dtype=np.float32)
        self.temp_HP = np.zeros((self.obs_dim, self.state_dim), dtype=np.float32)  # For H*P
        self.I = np.eye(self.state_dim, dtype=np.float32)
        
        self.is_initialized = False

    def _build_F(self, dt):
        """Build state transition matrix F in-place to avoid memory allocation."""
        # Reset to identity
        self.F.fill(0.0)
        np.fill_diagonal(self.F, 1.0)
        
        # Fill non-diagonal elements
        self.F[0,3] = self.F[1,4] = self.F[2,5] = dt
        half_dt2 = 0.5 * dt * dt
        self.F[0,6] = self.F[1,7] = self.F[2,8] = half_dt2
        self.F[3,6] = self.F[4,7] = self.F[5,8] = dt

    def predict(self, dt):
        """Optimized prediction step using in-place operations."""
        if not self.is_initialized or dt <= 0.0 or dt > 1.0:
            return
            
        self._build_F(dt)
        
        # x = F @ x (in-place)
        np.dot(self.F, self.x, out=self.temp_state)
        self.x[:] = self.temp_state
        
        # P = F @ P @ F.T + Q (optimized)
        np.dot(self.F, self.P, out=self.temp_P)
        np.dot(self.temp_P, self.F.T, out=self.P)
        self.P += self.Q_base

    def update(self, odom_vec):
        """Optimized update step using in-place operations."""
        # Convert to numpy array if needed
        if not isinstance(odom_vec, np.ndarray):
            odom_vec = np.array(odom_vec, dtype=np.float32)
            
        # Extract measurement
        if odom_vec.size == 13:  # full odom style
            self.temp_obs[0:3] = odom_vec[0:3]   # position
            self.temp_obs[3:6] = odom_vec[7:10]  # velocity
        elif odom_vec.size == 6:
            self.temp_obs[:] = odom_vec
        else:
            return  # ignore malformed input
            
        if not self.is_initialized:
            self.x[0:3] = self.temp_obs[0:3]
            self.x[3:6] = self.temp_obs[3:6]
            self.is_initialized = True
            return
            
        # Innovation: y = z - H*x
        np.dot(self.H, self.x, out=self.temp_y)
        self.temp_y = self.temp_obs - self.temp_y
        
        # Innovation covariance: S = H*P*H.T + R
        np.dot(self.H, self.P, out=self.temp_HP)  # H*P -> (6,9)
        np.dot(self.temp_HP, self.H.T, out=self.temp_S)  # H*P*H.T -> (6,6)
        self.temp_S += self.R
        
        # Kalman gain: K = P*H.T*inv(S) - simple and fast
        np.dot(self.P, self.H.T, out=self.temp_K)  # P*H.T -> (9,6)
        try:
            # Direct solve: K = solve(S, (P*H.T).T).T = solve(S, temp_K.T).T
            self.temp_K[:] = np.linalg.solve(self.temp_S, self.temp_K.T).T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            self.temp_K[:] = np.dot(self.temp_K, np.linalg.pinv(self.temp_S))
        
        # State update: x = x + K*y
        self.x += np.dot(self.temp_K, self.temp_y)
        
        # Covariance update: P = (I - K*H)*P
        np.dot(self.temp_K, self.H, out=self.temp_P)
        self.temp_P = self.I - self.temp_P
        self.P = np.dot(self.temp_P, self.P)

    def get_position(self):
        """Return position as numpy array (no copy needed)."""
        return self.x[0:3]

    def get_velocity(self):
        """Return velocity as numpy array (no copy needed)."""
        return self.x[3:6]


class EKFNode:
    """ROS node optimized for low-power processors like N100."""

    def __init__(self):
        rospy.init_node('ekf_node', anonymous=True)

        # Load configuration
        script_dir = os.path.dirname(os.path.realpath(__file__))
        pkg_root = os.path.abspath(os.path.join(script_dir, '..'))
        cfg_path = os.path.join(pkg_root, 'config', 'ekf_cfg.yaml')
        cfg_path = "ekf_cfg.yaml"  # Use local config file for faster loading
        
        config = {}
        try:
            with open(cfg_path, 'r') as f:
                raw = yaml.safe_load(f) or {}
                config = raw.get('ekf', {})
        except Exception as e:
            rospy.logwarn(f"Failed to load EKF config file '{cfg_path}': {e}. Using defaults.")
            config = {}

        # Extract parameters with optimized defaults for N100
        max_dt = float(config.get('max_dt', 0.5))  # Reduced from 1.0
        min_dt = float(config.get('min_dt', 0.001))  # Increased from 0.0005
        q_cfg = config.get('q', {}) or {}
        r_cfg = config.get('r', {}) or {}
        q_pos = float(q_cfg.get('pos', 0.01))
        q_vel = float(q_cfg.get('vel', 0.10))
        q_acc = float(q_cfg.get('acc', 1.0))
        r_pos = float(r_cfg.get('pos', 0.10))
        r_vel = float(r_cfg.get('vel', 0.20))

        self.max_dt = max_dt
        self.min_dt = min_dt
        self.ekf = EKF9D(q_pos=q_pos, q_vel=q_vel, q_acc=q_acc, r_pos=r_pos, r_vel=r_vel)
        # 可选互斥锁（单线程 spinner 时可关闭减少开销）
        self.use_lock = bool(config.get('use_lock', False))
        self.data_lock = threading.Lock() if self.use_lock else None
        self.last_odom_time = None
        self.last_pub_time = None

        # Filtering settings
        self.ma_window_pos = int(config.get('ma_window_pos', config.get('ma_window', 3)))
        self.ma_window_vel = int(config.get('ma_window_vel', config.get('ma_window', 3)))
        self.ema_alpha_pos = float(config.get('ema_alpha_pos', 0.3))
        self.ema_alpha_vel = float(config.get('ema_alpha_vel', 0.3))
        self._ema_pos = None
        self._ema_vel = None
        self._pos_hist = deque(maxlen=max(self.ma_window_pos, 1))
        self._vel_hist = deque(maxlen=max(self.ma_window_vel, 1))
        if self.ma_window_pos > 1:
            self._pos_array = np.zeros((self.ma_window_pos, 3), dtype=np.float32)
        if self.ma_window_vel > 1:
            self._vel_array = np.zeros((self.ma_window_vel, 3), dtype=np.float32)

        # Output message
        self.output_odom = Odometry()
        self.output_odom.header.frame_id = 'world'
        self.output_odom.child_frame_id = 'base_link'

        # Preallocated measurement vector
        self._odom_vec = np.zeros(13, dtype=np.float32)

        # Pub/Sub
        self.odom_pub = rospy.Publisher('/ekf/odom', Odometry, queue_size=5)
        self.odom_sub = rospy.Subscriber(TARGET_ODOM_TOPIC, Odometry, self.odom_callback, queue_size=5)

        # Prediction timer
        self.predict_rate = float(config.get('predict_rate', 0.0))
        self.publish_on_predict = bool(config.get('publish_on_predict', True))
        if self.predict_rate > 0:
            self.predict_timer = rospy.Timer(rospy.Duration(1.0 / self.predict_rate), self._predict_timer_cb)
        else:
            self.predict_timer = None

        # Watchdog
        self.watchdog_timer = rospy.Timer(rospy.Duration(2.0), self.watchdog_callback)

        rospy.loginfo(
            f"EKF node started device={self.ekf.device} q_pos={q_pos} q_vel={q_vel} q_acc={q_acc} "
            f"r_pos={r_pos} r_vel={r_vel} max_dt={max_dt} min_dt={min_dt} ema_pos={self.ema_alpha_pos} "
            f"ema_vel={self.ema_alpha_vel} predict_rate={self.predict_rate} use_lock={self.use_lock}"
        )

    # ---------------- internal helpers ----------------
    def _publish_filtered(self, stamp, predicted=False):
        pos = self.ekf.get_position()
        vel = self.ekf.get_velocity()
        # EMA -> MA -> raw
        if 0.0 < self.ema_alpha_pos <= 1.0:
            if self._ema_pos is None:
                self._ema_pos = pos.copy()
            else:
                self._ema_pos *= (1.0 - self.ema_alpha_pos)
                self._ema_pos += self.ema_alpha_pos * pos
            pos_f = self._ema_pos
        elif self.ma_window_pos > 1:
            self._pos_hist.append(pos.copy())
            pos_f = np.mean(self._pos_hist, axis=0)
        else:
            pos_f = pos
        if 0.0 < self.ema_alpha_vel <= 1.0:
            if self._ema_vel is None:
                self._ema_vel = vel.copy()
            else:
                self._ema_vel *= (1.0 - self.ema_alpha_vel)
                self._ema_vel += self.ema_alpha_vel * vel
            vel_f = self._ema_vel
        elif self.ma_window_vel > 1:
            self._vel_hist.append(vel.copy())
            vel_f = np.mean(self._vel_hist, axis=0)
        else:
            vel_f = vel
        P = self.ekf.P
        self.output_odom.header.stamp = stamp
        self.output_odom.child_frame_id = 'base_link_pred' if predicted else 'base_link'
        self.output_odom.pose.pose.position.x = float(pos_f[0])
        self.output_odom.pose.pose.position.y = float(pos_f[1])
        self.output_odom.pose.pose.position.z = float(pos_f[2])
        self.output_odom.twist.twist.linear.x = float(vel_f[0])
        self.output_odom.twist.twist.linear.y = float(vel_f[1])
        self.output_odom.twist.twist.linear.z = float(vel_f[2])
        self.output_odom.pose.covariance[0] = P[0,0]
        self.output_odom.pose.covariance[7] = P[1,1]
        self.output_odom.pose.covariance[14] = P[2,2]
        self.output_odom.twist.covariance[0] = P[3,3]
        self.output_odom.twist.covariance[7] = P[4,4]
        self.output_odom.twist.covariance[14] = P[5,5]
        self.odom_pub.publish(self.output_odom)
        self.last_pub_time = stamp

    def _predict_timer_cb(self, event):
        if not self.ekf.is_initialized or self.last_odom_time is None:
            return
        now = rospy.Time.now()
        dt = (now - self.last_odom_time).to_sec()
        if dt <= 0 or dt > self.max_dt:
            return
        if self.use_lock:
            with self.data_lock:
                self.ekf.predict(dt)
                self.last_odom_time = now
        else:
            self.ekf.predict(dt)
            self.last_odom_time = now
        if self.publish_on_predict:
            self._publish_filtered(now, predicted=True)
        
    def watchdog_callback(self, event):
        """Lightweight watchdog with reduced frequency."""
        now = rospy.Time.now()
        if not self.last_odom_time:
            rospy.logwarn_throttle(10, "EKF Watchdog: no first odom received yet.")
            return
        gap = (now - self.last_odom_time).to_sec()
        if gap > 2.0:  # Increased threshold to reduce log spam
            rospy.logwarn_throttle(5, f"EKF Watchdog: no odom for {gap:.2f}s")
        else:
            rospy.loginfo_throttle(30, "EKF Watchdog: normal running!")

    def odom_callback(self, msg: Odometry):
        """Optimized callback with minimal memory allocation."""
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        dt = None
        
        if self.last_odom_time is not None:
            dt = (stamp - self.last_odom_time).to_sec()
            # clamp / ignore dt outside configured thresholds
            if dt < self.min_dt or dt > self.max_dt:
                rospy.logwarn_throttle(5, f"Clamping dt: {dt} (min: {self.min_dt}, max: {self.max_dt})")
                dt = None

        # 填充预分配测量向量 self._odom_vec
        v = self._odom_vec
        v[0] = msg.pose.pose.position.x
        v[1] = msg.pose.pose.position.y
        v[2] = msg.pose.pose.position.z
        v[3] = msg.pose.pose.orientation.x
        v[4] = msg.pose.pose.orientation.y
        v[5] = msg.pose.pose.orientation.z
        v[6] = msg.pose.pose.orientation.w
        v[7] = msg.twist.twist.linear.x
        v[8] = msg.twist.twist.linear.y
        v[9] = msg.twist.twist.linear.z
        v[10] = msg.twist.twist.angular.x
        v[11] = msg.twist.twist.angular.y
        v[12] = msg.twist.twist.angular.z

        if self.use_lock:
            with self.data_lock:
                if dt is not None:
                    self.ekf.predict(dt)
                self.ekf.update(v)
                self.last_odom_time = stamp
        else:
            if dt is not None:
                self.ekf.predict(dt)
            self.ekf.update(v)
            self.last_odom_time = stamp

        # Publish
        if self.ekf.is_initialized:
            self._publish_filtered(stamp, predicted=False)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = EKFNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("EKF Node shutting down")
    except Exception:
        rospy.logerr("EKF Node error occurred")
        import traceback
        traceback.print_exc()
