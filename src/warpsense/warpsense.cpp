#include <ros/ros.h>
#include "warpsense/app.h"
#include <csignal>

std::function<void(int)> sigint_callback;
void sigint_handler(int value)
{
  sigint_callback(value);
}

void mySigintHandler(int sig)
{
  ROS_ERROR("SHIT\n");
  ros::shutdown();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "warpsense_new");
  ros::NodeHandle nh("~");

  Params params(nh);

  warpsense::App app(nh, params);
  //signal(SIGINT, mySigintHandler);
  //signal(SIGTERM, [](int signal) { return terminate_or_interrupt_handler(app, signal); });

  sigint_callback = std::bind(&warpsense::App::terminate,
                              &app,
                              std::placeholders::_1);

  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = sigint_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  ros::spin();
  return 0;
}
