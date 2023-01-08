#pragma once

#include "featsense/buffers.h"
#include "params/params.h"
#include "util/background_thread.h"
#include <tf/transform_broadcaster.h>

class Visualization : public BackgroundThread
{
public:
  Visualization(ros::NodeHandle& nh, const Params& params, Buffers& buffers);
  ~Visualization() = default;

  void thread_run() final;
private:
  ros::Publisher path_pub_;
  tf::TransformBroadcaster br_;
  OdomBuffer::Ptr& odom_buf_;
  HeaderBuffer::Ptr& header_buf_;
};

