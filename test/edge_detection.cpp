/*
 * OpenCV Example using ROS and CPP
 */

// Include the ROS library
#include <ros/ros.h>

// Include opencv2
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// Topics
static const std::string IMAGE_TOPIC = "/ouster/signal_image";
static const std::string PUBLISH_TOPIC = "/ouster/reflec_image_edges";

ros::Publisher edge_pub;
ros::Publisher gauss_pub;
ros::Publisher bilateral_pub;

void image_cb(const sensor_msgs::ImageConstPtr& msg)
{
  std_msgs::Header header = msg->header;

  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_BGR2GRAY);
  cv::Mat blurred;
  cv::Mat edges;
  cv::Mat bilateral;
  cv::Mat eq;

  cv::GaussianBlur(cv_ptr->image, blurred, cv::Size(5, 5), 0);
  //cv::equalizeHist(blurred, eq);
  cv::bilateralFilter(blurred, bilateral, 10, 20, 20, cv::BORDER_WRAP);
  cv::Canny(bilateral, edges, 10, 250);

  sensor_msgs::Image img;
  cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
  auto img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, edges);
  img_bridge.toImageMsg(img); // from cv_bridge to sensor_msgs::Image
  // Output modified video stream
  edge_pub.publish(img);

  cv::cvtColor(blurred, blurred, cv::COLOR_GRAY2BGR);
  img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, blurred);
  img_bridge.toImageMsg(img); // from cv_bridge to sensor_msgs::Image
  // Output modified video stream
  gauss_pub.publish(img);

  cv::cvtColor(bilateral, bilateral, cv::COLOR_GRAY2BGR);
  img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, bilateral);
  img_bridge.toImageMsg(img); // from cv_bridge to sensor_msgs::Image
  // Output modified video stream
  bilateral_pub.publish(img);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_compare");
  ros::NodeHandle nh("~");

  edge_pub = nh.advertise<sensor_msgs::Image>("ouster/edge", 100);
  gauss_pub = nh.advertise<sensor_msgs::Image>("ouster/gauss", 100);
  bilateral_pub = nh.advertise<sensor_msgs::Image>("ouster/bilateral", 100);
  ros::Subscriber img_sub =  nh.subscribe<sensor_msgs::Image>(IMAGE_TOPIC, 1000, image_cb);

  ros::spin();
}