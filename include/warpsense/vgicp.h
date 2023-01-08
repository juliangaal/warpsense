#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>

#ifdef USE_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

inline std::string vgicp_description()
{
  #ifdef USE_CUDA
  return "(gpu)";
  #else
  return "(cpu)";
  #endif
}

inline Eigen::Matrix4f apply_registration(pcl::PointCloud<pcl::PointXYZI>::Ptr& source, pcl::PointCloud<pcl::PointXYZI>::Ptr& target, float fitness_score_threshold)
{

  #ifdef USE_CUDA 
  // assumption: weak cpu, strong gpu -> bruteforce covariance estimation
  // TODO compare with rbf_kernel and cpu-based parallel kdtree
  fast_gicp::FastVGICPCuda<pcl::PointXYZI, pcl::PointXYZI> vgicp;
  vgicp.setResolution(1.0);
  vgicp.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  // kernel width (and distance threshold) need to be tuned
  vgicp.setKernelWidth(0.5);
  #else
  fast_gicp::FastVGICP<pcl::PointXYZI, pcl::PointXYZI> vgicp;
  vgicp.setResolution(1.0);
  vgicp.setNumThreads(omp_get_max_threads());
  #endif
  double fitness_score = 0.0;
  static int n_calls = 0;
  static float vgicp_sum = 0.0f;

  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  // fast_gicp reuses calculated covariances if an input cloud is the same as the previous one
  // to prevent this for benchmarking, force clear source and target clouds
  vgicp.clearTarget();
  vgicp.clearSource();
  vgicp.setInputTarget(target);
  vgicp.setInputSource(source);
  vgicp.align(*source);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = vgicp.getFitnessScore();
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

  n_calls += 1;
  vgicp_sum += single;
  //ROS_INFO_STREAM("vgicp " << vgicp_description() << ": " << single << "[msec] | health: " << std::setw(3) << fitness_score << " | avg: " << vgicp_sum / n_calls << "[msec]");

  if (fitness_score > fitness_score_threshold)
  {
    ROS_WARN_STREAM("vgicp " << vgicp_description() << " fitness score " << fitness_score << " is shit");
    return Eigen::Matrix4f::Identity();
  }
  
  return vgicp.getFinalTransformation(); 
}

inline void subsample(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, float resolution)
{
  static pcl::ApproximateVoxelGrid<pcl::PointXYZI> voxelgrid;
  voxelgrid.setLeafSize(resolution, resolution, resolution);
  voxelgrid.setInputCloud(cloud);
  voxelgrid.filter(*cloud);
}

inline Eigen::Matrix4f vgicp_transform(pcl::PointCloud<pcl::PointXYZI>::Ptr& source, pcl::PointCloud<pcl::PointXYZI>::Ptr& target, float resolution, float fitness_score_threshold)
{
  subsample(source, resolution);
  subsample(target, resolution);
  return apply_registration(source, target, fitness_score_threshold);
}