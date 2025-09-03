#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <deque>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>

// 9D constant-acceleration EKF (p,v,a)
class EKF9D {
public:
  using Matrix = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>;
  using Vector = Eigen::Matrix<float,Eigen::Dynamic,1>;
  EKF9D(float q_pos, float q_vel, float q_acc, float r_pos, float r_vel) {
    state_dim_ = 9; obs_dim_ = 6;
    x_.setZero(state_dim_);
    P_.setIdentity(state_dim_,state_dim_); P_ *= 10.f;
    Q_.setZero(state_dim_,state_dim_);
    R_.setZero(obs_dim_,obs_dim_);
    for(int i=0;i<3;++i){
      Q_(i,i)=q_pos; Q_(i+3,i+3)=q_vel; Q_(i+6,i+6)=q_acc;
      R_(i,i)=r_pos; R_(i+3,i+3)=r_vel;
    }
    F_.setIdentity(state_dim_,state_dim_);
    H_.setZero(obs_dim_,state_dim_);
    for(int i=0;i<3;++i){ H_(i,i)=1.f; H_(i+3,i+3)=1.f; }
    is_initialized_=false;
  }
  void predict(float dt){
    if(!is_initialized_ || dt<=0.f || dt>1.f) return;
    buildF(dt);
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
  }
  void update(const Eigen::Matrix<float,13,1>& z){
    Eigen::Matrix<float,6,1> meas;
    // pos 0..2, vel 7..9
    meas.segment<3>(0)=z.segment<3>(0);
    meas.segment<3>(3)=z.segment<3>(7);
    if(!is_initialized_){
      x_.segment<3>(0)=meas.segment<3>(0);
      x_.segment<3>(3)=meas.segment<3>(3);
      is_initialized_=true; return;
    }
    Eigen::Matrix<float,6,1> y = meas - H_*x_;
    Eigen::Matrix<float,6,6> S = H_*P_*H_.transpose() + R_;
    Eigen::Matrix<float,9,6> K = P_*H_.transpose() * S.inverse();
    x_ = x_ + K*y;
    Eigen::Matrix<float,9,9> I = Eigen::Matrix<float,9,9>::Identity();
    P_ = (I - K*H_) * P_;
  }
  Eigen::Vector3f pos() const { return x_.segment<3>(0); }
  Eigen::Vector3f vel() const { return x_.segment<3>(3); }
  const Eigen::Matrix<float,9,9>& cov() const { return reinterpret_cast<const Eigen::Matrix<float,9,9>&>(P_); }
  bool initialized() const { return is_initialized_; }
private:
  void buildF(float dt){
    F_.setIdentity();
    F_.block<3,3>(0,3).setIdentity(); F_.block<3,3>(0,3)*=dt;
    float half_dt2 = 0.5f*dt*dt;
    F_(0,6)=half_dt2; F_(1,7)=half_dt2; F_(2,8)=half_dt2;
    F_(3,6)=dt; F_(4,7)=dt; F_(5,8)=dt;
  }
  int state_dim_, obs_dim_;
  Vector x_; Matrix P_, Q_, R_, F_, H_;
  bool is_initialized_;
};

class EKFNode {
public:
  EKFNode(ros::NodeHandle& nh): nh_(nh), nh_global_(){
    std::string target_topic = "vicon/objtarget/odom";
    nh_.param<std::string>("target_topic", target_topic, target_topic);
    ROS_INFO_STREAM("Parameter target_topic: " << target_topic);
    // Load optional YAML config path
    std::string cfg_path; nh_.param<std::string>("cfg_path", cfg_path, "");
    ROS_INFO_STREAM("Parameter cfg_path: " << cfg_path);
    float qpos=0.01f,qvel=0.1f,qacc=1.f,rpos=0.1f,rvel=0.2f; 
    float ema_pos=0.f, ema_vel=0.f; int ma_pos=1, ma_vel=1; 
    double max_dt=1.0, min_dt=0.0005;
    double target_hz=0.0;  // Target EKF update frequency (0 = no limit)
    if(!cfg_path.empty()){
      try{
        YAML::Node root = YAML::LoadFile(cfg_path);
        auto ekf = root["ekf"];
        if(ekf){
          qpos = ekf["q"]["pos"].as<float>(qpos);
            qvel = ekf["q"]["vel"].as<float>(qvel);
            qacc = ekf["q"]["acc"].as<float>(qacc);
            rpos = ekf["r"]["pos"].as<float>(rpos);
            rvel = ekf["r"]["vel"].as<float>(rvel);
            ema_pos = ekf["ema_alpha_pos"].as<float>(ema_pos);
            ema_vel = ekf["ema_alpha_vel"].as<float>(ema_vel);
            ma_pos = ekf["ma_window_pos"].as<int>(ma_pos);
            ma_vel = ekf["ma_window_vel"].as<int>(ma_vel);
            max_dt = ekf["max_dt"].as<double>(max_dt);
            min_dt = ekf["min_dt"].as<double>(min_dt);
            target_hz = ekf["target_hz"].as<double>(target_hz);
        }
      }catch(const std::exception& e){
        ROS_WARN_STREAM("Failed to load YAML config: "<<e.what());
      }
    }
    ekf_.reset(new EKF9D(qpos,qvel,qacc,rpos,rvel));
    ema_alpha_pos_=ema_pos; ema_alpha_vel_=ema_vel; ma_window_pos_=ma_pos; ma_window_vel_=ma_vel;
    max_dt_=max_dt; min_dt_=min_dt; target_hz_=target_hz;
    if(ma_window_pos_>1) pos_hist_.set_capacity(ma_window_pos_);
    if(ma_window_vel_>1) vel_hist_.set_capacity(ma_window_vel_);

    odom_sub_ = nh_global_.subscribe(target_topic,10,&EKFNode::odomCb,this);
    odom_pub_ = nh_global_.advertise<nav_msgs::Odometry>("/ekf/odom",10);

    ROS_INFO_STREAM("Subscribing to topic: " << target_topic);
    ROS_INFO_STREAM("Publishing to topic: /ekf/odom");

    // 检查topic是否存在
    ros::master::V_TopicInfo topics;
    ros::master::getTopics(topics);
    bool topic_found = false;
    for(const auto& topic : topics) {
      if(topic.name.find("vicon/objtarget/odom") != std::string::npos) {
        topic_found = true;
        ROS_INFO_STREAM("Found input topic: " << topic.name);
      }
    }
    if(!topic_found) {
      ROS_WARN("Input topic vicon/objtarget/odom not found! Available topics:");
      ros::master::getTopics(topics);
      for(const auto& topic : topics) {
        ROS_WARN_STREAM("  " << topic.name);
      }
    }

    ROS_INFO_STREAM("EKF C++ node started q_pos="<<qpos<<" q_vel="<<qvel<<" q_acc="<<qacc
                    <<" r_pos="<<rpos<<" r_vel="<<rvel<<" ema_pos="<<ema_alpha_pos_<<" ema_vel="<<ema_alpha_vel_
                    <<" target_hz="<<target_hz_);
  }
private:
  template<typename T>
  class Ring {
  public:
    void set_capacity(size_t c){cap_=c; buf_.clear();}
    void push(const Eigen::Vector3f& v){ if(cap_==0) return; if(buf_.size()==cap_) buf_.pop_front(); buf_.push_back(v);}    
    Eigen::Vector3f mean() const{ Eigen::Vector3f m=Eigen::Vector3f::Zero(); if(buf_.empty()) return m; for(auto &v:buf_) m+=v; return m/static_cast<float>(buf_.size()); }
  private:
    size_t cap_{0}; std::deque<Eigen::Vector3f> buf_;
  };

  void odomCb(const nav_msgs::OdometryConstPtr& msg){
    ros::Time stamp = msg->header.stamp.isZero()? ros::Time::now(): msg->header.stamp;
    
    // Frequency limiting: skip if not enough time has passed
    if(target_hz_ > 0.0 && !last_ekf_time_.isZero()) {
      double min_interval = 1.0 / target_hz_;
      double time_since_last = (stamp - last_ekf_time_).toSec();
      if(time_since_last < min_interval) {
        ROS_DEBUG_THROTTLE(5.0, "Skipping EKF update: %.3fs < %.3fs", time_since_last, min_interval);
        return;  // Skip this update to maintain target frequency
      }
    }
    
    ROS_INFO_THROTTLE(1.0, "Received odom message, pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f)", 
                      msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
                      msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
    if(last_meas_time_.isZero()) last_meas_time_=stamp;
    double dt = (stamp - last_meas_time_).toSec();
    if(dt>0) ekf_->predict(static_cast<float>(dt));
    last_meas_time_ = stamp;
    last_ekf_time_ = stamp;  // Update EKF processing time
    
    // Fill measurement vector
    Eigen::Matrix<float,13,1> z; z.setZero();
    z(0)=msg->pose.pose.position.x; z(1)=msg->pose.pose.position.y; z(2)=msg->pose.pose.position.z;
    z(3)=msg->pose.pose.orientation.x; z(4)=msg->pose.pose.orientation.y; z(5)=msg->pose.pose.orientation.z; z(6)=msg->pose.pose.orientation.w;
    z(7)=msg->twist.twist.linear.x; z(8)=msg->twist.twist.linear.y; z(9)=msg->twist.twist.linear.z;
    z(10)=msg->twist.twist.angular.x; z(11)=msg->twist.twist.angular.y; z(12)=msg->twist.twist.angular.z;
    ekf_->update(z);
    publishState(stamp,false);
  }

  void publishState(const ros::Time& stamp, bool predicted){
    nav_msgs::Odometry odom; odom.header.stamp=stamp; odom.header.frame_id="world"; odom.child_frame_id="base_link";
    Eigen::Vector3f p = ekf_->pos(); Eigen::Vector3f v = ekf_->vel();
    // Filtering
    Eigen::Vector3f pf=p, vf=v;
    if(ema_alpha_pos_>0.f && ema_alpha_pos_<=1.f){ if(!ema_pos_init_){ema_pos_=p; ema_pos_init_=true;} else ema_pos_ = (1.f-ema_alpha_pos_)*ema_pos_ + ema_alpha_pos_*p; pf=ema_pos_; }
    else if(ma_window_pos_>1){ pos_hist_.push(p); pf = pos_hist_.mean(); }
    if(ema_alpha_vel_>0.f && ema_alpha_vel_<=1.f){ if(!ema_vel_init_){ema_vel_=v; ema_vel_init_=true;} else ema_vel_ = (1.f-ema_alpha_vel_)*ema_vel_ + ema_alpha_vel_*v; vf=ema_vel_; }
    else if(ma_window_vel_>1){ vel_hist_.push(v); vf = vel_hist_.mean(); }
    odom.pose.pose.position.x=pf.x(); odom.pose.pose.position.y=pf.y(); odom.pose.pose.position.z=pf.z();
    odom.twist.twist.linear.x=vf.x(); odom.twist.twist.linear.y=vf.y(); odom.twist.twist.linear.z=vf.z();
    const auto &P = ekf_->cov();
    odom.pose.covariance[0]=P(0,0); odom.pose.covariance[7]=P(1,1); odom.pose.covariance[14]=P(2,2);
    odom.twist.covariance[0]=P(3,3); odom.twist.covariance[7]=P(4,4); odom.twist.covariance[14]=P(5,5);
    ROS_INFO_THROTTLE(1.0, "Publishing %s: pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f)", 
                      predicted?"predicted":"filtered", pf.x(), pf.y(), pf.z(), vf.x(), vf.y(), vf.z());
    odom_pub_.publish(odom);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle nh_global_;
  std::unique_ptr<EKF9D> ekf_;
  ros::Subscriber odom_sub_;
  ros::Publisher odom_pub_;
  ros::Time last_meas_time_;
  ros::Time last_ekf_time_;  // Time of last EKF processing

  float ema_alpha_pos_{0.f}, ema_alpha_vel_{0.f};
  int ma_window_pos_{1}, ma_window_vel_{1};
  double max_dt_{1.0}, min_dt_{0.0005};
  double target_hz_{0.0};  // Target EKF update frequency

  // Filtering state
  Eigen::Vector3f ema_pos_; bool ema_pos_init_{false};
  Eigen::Vector3f ema_vel_; bool ema_vel_init_{false};
  Ring<float> pos_hist_, vel_hist_;
};

int main(int argc, char** argv){
  ros::init(argc, argv, "ekf_cpp_node");
  ros::NodeHandle nh("~");
  EKFNode node(nh);
  ros::spin();
  return 0;
}
