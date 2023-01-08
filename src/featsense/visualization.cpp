#include <Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include "featsense/visualization.h"
#include "featsense/mypcl.h"
#include "util/publish.h"

Visualization::Visualization(ros::NodeHandle &nh, const Params &params, Buffers &buffers)
    : path_pub_(nh.advertise<sensor_msgs::PointCloud2>("/path", 100)),
      br_(),
      odom_buf_(buffers.odom_buffer),
      header_buf_(buffers.header_buffer)
{

}

void Visualization::thread_run()
{
  while (ros::ok() && running)
  {
    if (odom_buf_ && !odom_buf_->empty() && header_buf_ && !header_buf_->empty())
    {
      std_msgs::Header header;
      header_buf_->pop_nb(&header);
      ros::Time pointcloud_time = header.stamp;

      Eigen::Isometry3d odom;
      odom_buf_->pop_nb(&odom);

      Eigen::Quaterniond q_current(odom.rotation());
      q_current.normalize();
      Eigen::Vector3d t_current = odom.translation();

      // publish tf transform
      tf::Transform transform;
      transform.setOrigin(tf::Vector3(t_current.x(), t_current.y(), t_current.z()));
      tf::Quaternion q(q_current.x(), q_current.y(), q_current.z(), q_current.w());
      transform.setRotation(q);
      br_.sendTransform(tf::StampedTransform(transform, pointcloud_time, "map", "base_link"));

      publish_path(odom, path_pub_, header);
    }

    //sleep 2 ms every time
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}
