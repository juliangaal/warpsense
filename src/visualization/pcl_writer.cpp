// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

// C++
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

class SubscribeWriter
{
private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  fs::path dir_;
  std::unique_ptr<pcl::PLYWriter> ply_writer_;
  std::unique_ptr<pcl::PCDWriter> pcd_writer_;
  size_t n_cloud_;

public:
  SubscribeWriter(ros::NodeHandle& nh) : nh_(nh), sub_(), dir_(), ply_writer_(), pcd_writer_(), n_cloud_(0)
  {
    std::string temp;
    nh.param<std::string>("dir", temp, "/tmp/plys/");

    bool ply;
    nh.param<bool>("ply", ply, false);
    if (ply)
    {
      ROS_INFO_STREAM("Will write ply files");
      ply_writer_.reset(new pcl::PLYWriter);
    }

    bool pcd;
    nh.param<bool>("pcd", pcd, false);
    if (pcd)
    {
      ROS_INFO_STREAM("Will write pcd files");
      pcd_writer_.reset(new pcl::PCDWriter);
    }

    dir_ = fs::path(temp);
    if (!fs::exists(dir_))
    {
      ROS_WARN_STREAM(dir_ << " does not exist, creating...");
      if (!fs::create_directory(dir_))
      {
        ROS_ERROR_STREAM("couldn't create " << dir_ << ", exiting.");
        ros::shutdown();
      }
    }
    else
    {
      ROS_ERROR_STREAM(dir_ << " already exists. Please backup data. Warpsense will not overwrite");
      ros::shutdown();
    }

    std::string topic;
    nh.param<std::string>("topic", topic, "/pointcloud_in");

    sub_ = nh_.subscribe(topic, 1, &SubscribeWriter::cloud_callback, this);
    ROS_INFO_STREAM("subscribing to " << topic << ", saving in " << dir_);
  }

  ~SubscribeWriter() = default;

  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud)
  {
    ROS_INFO_STREAM("Writer received pointcloud");

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << std::to_string(n_cloud_);
    fs::path basename = dir_ / ss.str();

    pcl::PCLPointCloud2 pcl;
    pcl_conversions::toPCL(*cloud, pcl);
    if (ply_writer_)
    {
      auto filename = basename.replace_extension(".ply");
      ply_writer_->writeBinary(filename, pcl);
      ROS_INFO_STREAM("saved ply " << filename);
    }

    if (pcd_writer_)
    {
      auto filename = basename.replace_extension(".pcd");
      pcd_writer_->writeBinary(filename, pcl);
      ROS_INFO_STREAM("saved pcd " << filename);
    }

    ++n_cloud_;
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "open3d_conversions_ex_repub_downsampled");
  ros::NodeHandle nh("~");
  SubscribeWriter s(nh);
  ros::spin();
}