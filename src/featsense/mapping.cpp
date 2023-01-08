#include "util/publish.h"
#include "featsense/mapping.h"
#include "warpsense/cuda/cleanup.h"

Mapping::Mapping(ros::NodeHandle &nh, const Params &params, Buffers &buffers)
    : hdf5_global_map_{new HDF5GlobalMap(params.map)},
      hdf5_local_map_{new HDF5LocalMap(params.map.size.x(), params.map.size.y(), params.map.size.z(), hdf5_global_map_)},
      map_shift_pose_buffer_{std::make_shared<ConcurrentRingBuffer<Eigen::Matrix4f>>(1)},
      gpu_{nh, params, map_shift_pose_buffer_, hdf5_local_map_},
      map_pub_(nh.advertise<sensor_msgs::PointCloud2>("/gicp_map", 100)),
      path_pub_(nh.advertise<sensor_msgs::PointCloud2>("/gicp_path", 100)),
      pre_path_pub_(nh.advertise<sensor_msgs::PointCloud2>("/gicp_pre_path", 100)),
      params_(params),
      cloud_buf_(buffers.vgicp_cloud_buffer),
      odom_buf_(buffers.vgicp_odom_buffer),
      header_buf_(buffers.vgicp_header_buffer),
      initialized_(false),
      terminated_(false)
{
}

pcl::PointCloud<pcl::PointXYZI>::Ptr enrich_from_queue(std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr>& queue, int n_scans)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr enriched = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());

  for (const auto &element: queue)
  {
    *enriched += *element;
  }

  if (queue.size() == n_scans)
  {
    queue.pop_back();
  }

  return enriched;
}

void Mapping::thread_run()
{
  std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> last_pcls;
  pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_pcl_during_shift(new pcl::PointCloud<pcl::PointXYZI>());
  Eigen::Isometry3d last_gicp_pose;
  Eigen::Isometry3d last_floam_pose;
  double total_time = 0;
  int total_frame = 0;
  float resolution = static_cast<float>(params_.map.resolution) / 1000.f;

  while (ros::ok() && running)
  {
    if (cloud_buf_ && !cloud_buf_->empty() && odom_buf_ && !odom_buf_->empty() && header_buf_ && !header_buf_->empty())
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());
      cloud_buf_->pop_nb(&pointcloud_in);

      Eigen::Isometry3d floam_pose;
      odom_buf_->pop_nb(&floam_pose);

      std_msgs::Header header;
      header_buf_->pop_nb(&header);

      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();

      if (!initialized_)
      {
        last_pcls.push_front(pointcloud_in);
        last_gicp_pose = floam_pose;
        last_floam_pose = floam_pose;
        subsample(pointcloud_in, resolution);
        gpu_.update_tsdf_from_ros(pointcloud_in, floam_pose);
        initialized_ = true;
        continue;
      }

      // apply vgicp refinement

      double distance = (last_floam_pose.translation() - floam_pose.translation()).norm();

      if (distance > params_.map.update_distance)
      {
        Eigen::Quaterniond old_rot(last_floam_pose.rotation());
        Eigen::Quaterniond curr_rot(floam_pose.rotation());
        Eigen::Quaterniond diff_rot = curr_rot * old_rot.inverse();

        Eigen::Vector3d old_translation = last_floam_pose.translation();
        Eigen::Vector3d curr_translation = floam_pose.translation();
        Eigen::Vector3d diff_trans = curr_translation - old_translation;

        Eigen::Isometry3d initial_transform = last_gicp_pose;
        initial_transform.rotate(diff_rot);
        initial_transform.pretranslate(diff_trans);

        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_pcl(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::transformPointCloud(*pointcloud_in, *transformed_pcl, initial_transform.cast<float>());

        auto last_pcl = enrich_from_queue(last_pcls, params_.floam.enrich);
        auto transform = vgicp_transform(transformed_pcl, last_pcl, resolution, params_.floam.vgicp_fitness_score);
        Eigen::Isometry3d gicp_pose = initial_transform;
        auto rotation = Eigen::Quaterniond(transform.block<3, 3>(0, 0).cast<double>());
        gicp_pose.prerotate(rotation);
        auto translation = Eigen::Vector3d(transform.block<3, 1>(0, 3).cast<double>());
        gicp_pose.pretranslate(translation);

        end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        total_frame++;
        float time_temp = elapsed_seconds.count() * 1000;
        total_time += time_temp;
        ROS_WARN("average vicp mapping time %f ms", total_time / total_frame);

        publish_cloud(*transformed_pcl, map_pub_, header);

        // update tsdf with gicp pose
        if (gpu_.is_shifting())
        {
          *accumulated_pcl_during_shift += *transformed_pcl;
          ROS_WARN("Map is shifting. Accumulated %ld points", accumulated_pcl_during_shift->size());
        }
        else
        {
          if (!accumulated_pcl_during_shift->empty())
          {
            *transformed_pcl += *accumulated_pcl_during_shift;
            accumulated_pcl_during_shift->clear();
            subsample(transformed_pcl, resolution);
          }
          gpu_.update_tsdf_from_ros(transformed_pcl, gicp_pose);
        }

        // update map
        last_gicp_pose = gicp_pose;
        last_floam_pose = floam_pose;
        last_pcls.push_front(transformed_pcl);

        // save pose
        hdf5_global_map_->write_pose(gicp_pose, 1);

        // push pose to map shift thread
        update_map_shift(gicp_pose);

        publish_path(gicp_pose, path_pub_, header);
        publish_path(initial_transform, pre_path_pub_, header);
      }
    }
  }
}

void Mapping::update_map_shift(const Eigen::Isometry3d &gicp_pose)
{
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3,3>(0,0) = gicp_pose.rotation();
  pose.block<3,1>(0,3) = gicp_pose.translation() * 1000.f; // to scale for fastsense backend
  map_shift_pose_buffer_->push_nb(pose.cast<float>());
}

void Mapping::terminate(int signal)
{
  // force emptying of queues, so that no more runtime_evaluator form is active and can be written to csv
  running = false;
  cloud_buf_.reset();
  odom_buf_.reset();
  header_buf_.reset();

  gpu_.join_mapping_thread();
  ROS_INFO_STREAM("Joined mapping thread");

  ROS_INFO_STREAM("Waiting for all callbacks to finish...");
  std::this_thread::sleep_for(std::chrono::seconds(1));

  ROS_INFO_STREAM("Initialized " << std::boolalpha << initialized_);

  if (initialized_)
  {
#ifdef TIME_MEASUREMENT
    fs::path filename = (fs::path(DATA_PATH) / params_.map.identifier()).replace_extension("csv");
  ROS_INFO_STREAM("Saving runtime evaluator data in " << filename);
  auto &runtime_eval_ = RuntimeEvaluator::get_instance();
  if (!runtime_eval_.export_results(filename))
  {
    ROS_ERROR_STREAM("Couldn't save runtime evaluator stats");
  }
#endif

    ROS_INFO_STREAM("Writing map into " << hdf5_global_map_->filename());
    cuda::pause();
    cuda::DeviceMap existing_cuda_map(hdf5_local_map_);
    gpu_.tsdf()->avg_map().to_host(existing_cuda_map);
    hdf5_local_map_->write_back();
    ROS_INFO_STREAM("Completed");
  }

  ros::shutdown();
}

Mapping::~Mapping()
{
  // can't call terminate() directly, unfortunately
  // ROS may call destructor, too and therefore call terminate() is called twice (ctrl+c, RAII)
  // without checking
  if (!terminated_)
  {
    terminate();
    terminated_ = true;
  }
}

