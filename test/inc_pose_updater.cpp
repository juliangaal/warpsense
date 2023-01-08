/**
  * @file inc_pose_updater.cpp
  * @author julian 
  * @date 7/6/22
 */

#include <ros/ros.h>
#include <util/publish.h>
#include <sensor_msgs/PointCloud.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud", 1000);
    ros::Publisher opub = nh.advertise<nav_msgs::Odometry>("/odom", 1000);
    ros::Publisher lopub = nh.advertise<nav_msgs::Odometry>("/lodom", 1000);
    
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    float theta = M_PI/12; // The angle of rotation in radians
    mat(0, 0) = std::cos (theta);
    mat(0, 1) = -sin(theta);
    mat(1, 0) = sin (theta);
    mat(1, 1) = std::cos (theta);
    mat(0, 3) = 0.5;

    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();

    // constant pose change increment
    Eigen::Quaterniond rot_inc = Eigen::Quaterniond(mat.block<3, 3>(0, 0));
    Eigen::Vector3d trans_inc = Eigen::Vector3d(mat.block<3, 1>(0, 3));

    Eigen::Isometry3d last_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d corr_pose = last_pose;
    Eigen::Quaterniond rot_diff = Eigen::Quaterniond::Identity();
    Eigen::Vector3d trans_diff = Eigen::Vector3d::Zero();

    ros::Rate rate(1.0);
    size_t i = 0;
    while (ros::ok())
    { 
        // update absolute pose
        rotation *= rot_inc;
        translation = rot_inc * translation + trans_inc;

        // update absolute difference to last pose
        rot_diff = rotation * last_pose.rotation().inverse();
        trans_diff = translation - last_pose.translation();

        // simulate lower frequency of vgicp
        if (i % 5 == 0 && i != 0)
        {
          Eigen::Quaterniond q(corr_pose.rotation() * rot_diff);
          Eigen::Vector3d t(corr_pose.translation() + trans_diff);
          publish_odometry(q, t, lopub);
          
          // update last pose with absolute pose estimate
          last_pose = Eigen::Isometry3d::Identity();
          last_pose.rotate(rotation);
          last_pose.pretranslate(translation);

          // translate to simulate drift correction
          corr_pose = last_pose;
          // corr_pose.pretranslate(trans_inc);

          // reset differences
          rot_diff = Eigen::Quaterniond::Identity();
          trans_diff = Eigen::Vector3d::Zero();
        }

        std_msgs::Header header;
        header.stamp = ros::Time::now();
        header.frame_id = "map";
        publish_path(rotation, translation, pub, header);
        publish_odometry(rotation, translation, opub);

        rate.sleep();
        ++i;
    }
}