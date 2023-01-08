// Original Copyright:
// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
//
// Modification Copyright: Julian

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include "atan.h"

int N_SCANS = 128;
float min_distance = 2;
float max_distance = 60;

namespace original
{

class Double2d
{
public:
  int id;
  double value;

  Double2d(int id_in, double value_in) : id(id_in), value(value_in) {}
};

inline void
featureExtractionFromSector(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in, std::vector<Double2d> &cloudCurvature,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_edge,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf)
{

  std::sort(cloudCurvature.begin(), cloudCurvature.end(), [](const Double2d &a, const Double2d &b)
  {
    return a.value < b.value;
  });


  int largestPickedNum = 0;
  std::vector<int> picked_points;
  int point_info_count = 0;
  for (int i = cloudCurvature.size() - 1; i >= 0; i--)
  {
    int ind = cloudCurvature[i].id;
    if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end())
    {
      if (cloudCurvature[i].value <= 0.1)
      {
        break;
      }

      largestPickedNum++;
      picked_points.push_back(ind);

      if (largestPickedNum <= 20)
      {
        pc_out_edge->push_back(pc_in->points[ind]);
        point_info_count++;
      } else
      {
        break;
      }

      for (int k = 1; k <= 5; k++)
      {
        double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
        double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
        double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
        {
          break;
        }
        picked_points.push_back(ind + k);
      }
      for (int k = -1; k >= -5; k--)
      {
        double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
        double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
        double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
        {
          break;
        }
        picked_points.push_back(ind + k);
      }

    }
  }

  //find flat points
  // point_info_count =0;
  // int smallestPickedNum = 0;

  // for (int i = 0; i <= (int)cloudCurvature.size()-1; i++)
  // {
  //     int ind = cloudCurvature[i].id;

  //     if( std::find(picked_points.begin(), picked_points.end(), ind)==picked_points.end()){
  //         if(cloudCurvature[i].value > 0.1){
  //             //ROS_WARN("extracted feature not qualified, please check lidar");
  //             break;
  //         }
  //         smallestPickedNum++;
  //         picked_points.push_back(ind);

  //         if(smallestPickedNum <= 4){
  //             //find all points
  //             pc_surf_flat->push_back(pc_in->points[ind]);
  //             pc_surf_lessFlat->push_back(pc_in->points[ind]);
  //             point_info_count++;
  //         }
  //         else{
  //             break;
  //         }

  //         for(int k=1;k<=5;k++){
  //             double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
  //             double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
  //             double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
  //             if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05){
  //                 break;
  //             }
  //             picked_points.push_back(ind+k);
  //         }
  //         for(int k=-1;k>=-5;k--){
  //             double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
  //             double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
  //             double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
  //             if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05){
  //                 break;
  //             }
  //             picked_points.push_back(ind+k);
  //         }

  //     }
  // }

  for (int i = 0; i <= (int) cloudCurvature.size() - 1; i++)
  {
    int ind = cloudCurvature[i].id;
    if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end())
    {
      pc_out_surf->push_back(pc_in->points[ind]);
    }
  }
}

inline void
featureExtraction(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_edge,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf)
{
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*pc_in, indices);

  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> laserCloudScans;
  for (int i = 0; i < N_SCANS; i++)
  {
    laserCloudScans.push_back(pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>()));
  }

  for (int i = 0; i < (int) pc_in->points.size(); i++)
  {
    int scanID = 0;
    double distance = sqrt(pc_in->points[i].x * pc_in->points[i].x + pc_in->points[i].y * pc_in->points[i].y);
    if (distance < min_distance || distance > max_distance)
      continue;
    double angle = atan(pc_in->points[i].z / distance) * 180 / M_PI;

    if (N_SCANS == 128)
    {
      scanID = int((angle + 22.5) / 2 + 0.5);
      if (scanID > (N_SCANS - 1) || scanID < 0)
      {
        continue;
      }
    } else
    {
      printf("wrong scan number\n");
    }
    laserCloudScans[scanID]->push_back(pc_in->points[i]);

  }

  for (int i = 0; i < N_SCANS; i++)
  {
    if (laserCloudScans[i]->points.size() < 131)
    {
      continue;
    }

    std::vector<Double2d> cloudCurvature;
    int total_points = laserCloudScans[i]->points.size() - 10;
    for (int j = 5; j < (int) laserCloudScans[i]->points.size() - 5; j++)
    {
      double diffX = laserCloudScans[i]->points[j - 5].x + laserCloudScans[i]->points[j - 4].x +
                     laserCloudScans[i]->points[j - 3].x + laserCloudScans[i]->points[j - 2].x +
                     laserCloudScans[i]->points[j - 1].x - 10 * laserCloudScans[i]->points[j].x +
                     laserCloudScans[i]->points[j + 1].x + laserCloudScans[i]->points[j + 2].x +
                     laserCloudScans[i]->points[j + 3].x + laserCloudScans[i]->points[j + 4].x +
                     laserCloudScans[i]->points[j + 5].x;
      double diffY = laserCloudScans[i]->points[j - 5].y + laserCloudScans[i]->points[j - 4].y +
                     laserCloudScans[i]->points[j - 3].y + laserCloudScans[i]->points[j - 2].y +
                     laserCloudScans[i]->points[j - 1].y - 10 * laserCloudScans[i]->points[j].y +
                     laserCloudScans[i]->points[j + 1].y + laserCloudScans[i]->points[j + 2].y +
                     laserCloudScans[i]->points[j + 3].y + laserCloudScans[i]->points[j + 4].y +
                     laserCloudScans[i]->points[j + 5].y;
      double diffZ = laserCloudScans[i]->points[j - 5].z + laserCloudScans[i]->points[j - 4].z +
                     laserCloudScans[i]->points[j - 3].z + laserCloudScans[i]->points[j - 2].z +
                     laserCloudScans[i]->points[j - 1].z - 10 * laserCloudScans[i]->points[j].z +
                     laserCloudScans[i]->points[j + 1].z + laserCloudScans[i]->points[j + 2].z +
                     laserCloudScans[i]->points[j + 3].z + laserCloudScans[i]->points[j + 4].z +
                     laserCloudScans[i]->points[j + 5].z;
      Double2d distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
      cloudCurvature.push_back(distance);

    }
    for (int j = 0; j < 6; j++)
    {
      int sector_length = (int) (total_points / 6);
      int sector_start = sector_length * j;
      int sector_end = sector_length * (j + 1) - 1;
      if (j == 5)
      {
        sector_end = total_points - 1;
      }
      std::vector<Double2d> subCloudCurvature(cloudCurvature.begin() + sector_start,
                                              cloudCurvature.begin() + sector_end);

      featureExtractionFromSector(laserCloudScans[i], subCloudCurvature, pc_out_edge, pc_out_surf);
    }

  }
}

} // end namespace original

namespace custom
{

//points covariance class
struct Curvature
{
public:
  int id;
  double value;
  Curvature() : id(0), value(0) {}
  Curvature(int id_, double value_)
      : id(id_), value(value_)
  {}
};

struct Feature
{
  Feature() : idx(0), curvature(0) {}
  Feature(size_t idx, float curvature) : idx(idx), curvature(curvature) {}
  size_t idx;
  float curvature;
};

struct FeatureI
{
  FeatureI() : idx(0), curvature(0) {}
  FeatureI(size_t idx, float curvature) : idx(idx), curvature(curvature) {}
  size_t idx;
  float curvature;
  float intensity;
};

inline void extract_features_from_line(int j, const pcl::PointCloud<pcl::PointXYZI> &pc_in,
                                       std::vector<Curvature> &cloud_curvature,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_edge,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf)
{
  std::sort(cloud_curvature.begin(), cloud_curvature.end(), [](const Curvature &a, const Curvature &b)
  {
    return a.value < b.value;
  });


  int largestPickedNum = 0;
  std::vector<int> picked_points;
  picked_points.reserve(pc_in.size());
  int point_info_count = 0;
  for (int i = cloud_curvature.size() - 1; i >= 0; i--)
  {
    int ind = cloud_curvature[i].id;
    if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end())
    {
      if (cloud_curvature[i].value <= 2.0)
      {
        break;
      }

      largestPickedNum++;
      picked_points.push_back(ind);

      if (largestPickedNum <= 100)
      {
        pc_out_edge->push_back(pc_in.points[ind]);
        point_info_count++;
      }
      else
      {
        break;
      }

      for (int k = 1; k <= 5; k++)
      {
        double diffX = pc_in.points[ind + k].x - pc_in.points[ind + k - 1].x;
        double diffY = pc_in.points[ind + k].y - pc_in.points[ind + k - 1].y;
        double diffZ = pc_in.points[ind + k].z - pc_in.points[ind + k - 1].z;
        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
        {
          break;
        }
        picked_points.push_back(ind + k);
      }
      for (int k = -1; k >= -5; k--)
      {
        double diffX = pc_in.points[ind + k].x - pc_in.points[ind + k + 1].x;
        double diffY = pc_in.points[ind + k].y - pc_in.points[ind + k + 1].y;
        double diffZ = pc_in.points[ind + k].z - pc_in.points[ind + k + 1].z;
        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
        {
          break;
        }
        picked_points.push_back(ind + k);
      }

    }
  }

  for (int i = 0; i <= (int) cloud_curvature.size() - 1; i++)
  {
    int ind = cloud_curvature[i].id;
    if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end())
    {
      pc_out_surf->push_back(pc_in.points[ind]);
    }
  }
}

inline void extract_features(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_edge,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf)
{
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*pc_in, indices);

  const int& pcl_height = N_SCANS;
  const int& n_scanlines = N_SCANS;

  std::vector<pcl::PointCloud<pcl::PointXYZI>> scan_lines(pcl_height);

  for (int i = 0; i < (int) pc_in->points.size(); i++)
  {
    double distance = sqrt(pc_in->points[i].x * pc_in->points[i].x + pc_in->points[i].y * pc_in->points[i].y);
    if (distance < min_distance || distance > max_distance)
    {
      continue;
    }
    double angle = atan(pc_in->points[i].z / distance) * 180 / M_PI;
    auto scanID = lround((angle + (45.0 / 2.)) / 2);
    if (scanID > (pcl_height - 1) || scanID < 0)
    {
      continue;
    }
    scan_lines[scanID].push_back(pc_in->points[i]);
  }

  for (int i = 0; i < n_scanlines; i++)
  {
    if (scan_lines[i].points.size() < 10)
    {
      continue;
    }

    std::vector<Curvature> cloudCurvature;
    size_t total_points = scan_lines[i].points.size() - 10;
    cloudCurvature.reserve(total_points);
    for (int j = 5; j < (int) scan_lines[i].points.size() - 5; j++)
    {
      double diffX = scan_lines[i].points[j - 5].x + scan_lines[i].points[j - 4].x +
                     scan_lines[i].points[j - 3].x + scan_lines[i].points[j - 2].x +
                     scan_lines[i].points[j - 1].x - 10 * scan_lines[i].points[j].x +
                     scan_lines[i].points[j + 1].x + scan_lines[i].points[j + 2].x +
                     scan_lines[i].points[j + 3].x + scan_lines[i].points[j + 4].x +
                     scan_lines[i].points[j + 5].x;
      double diffY = scan_lines[i].points[j - 5].y + scan_lines[i].points[j - 4].y +
                     scan_lines[i].points[j - 3].y + scan_lines[i].points[j - 2].y +
                     scan_lines[i].points[j - 1].y - 10 * scan_lines[i].points[j].y +
                     scan_lines[i].points[j + 1].y + scan_lines[i].points[j + 2].y +
                     scan_lines[i].points[j + 3].y + scan_lines[i].points[j + 4].y +
                     scan_lines[i].points[j + 5].y;
      double diffZ = scan_lines[i].points[j - 5].z + scan_lines[i].points[j - 4].z +
                     scan_lines[i].points[j - 3].z + scan_lines[i].points[j - 2].z +
                     scan_lines[i].points[j - 1].z - 10 * scan_lines[i].points[j].z +
                     scan_lines[i].points[j + 1].z + scan_lines[i].points[j + 2].z +
                     scan_lines[i].points[j + 3].z + scan_lines[i].points[j + 4].z +
                     scan_lines[i].points[j + 5].z;
      Curvature curv(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
      cloudCurvature.push_back(curv);
    }

    for (int j = 0; j < 6; j++)
    {
      int sector_length = (int) (total_points / 6);
      int sector_start = sector_length * j;
      int sector_end = sector_length * (j + 1) - 1;
      if (j == 5)
      {
        sector_end = total_points - 1;
      }
      std::vector<Curvature> subCloudCurvature(cloudCurvature.begin() + sector_start,
                                               cloudCurvature.begin() + sector_end);
      extract_features_from_line(i, scan_lines[i], subCloudCurvature, pc_out_edge, pc_out_surf);
    }
  }
}

inline float range(const pcl::PointXYZI& p)
{
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

inline void extract_features_lio_sam(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_edge,
                                               pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf)
{
  auto width = 1024;
  auto height = 128;
  float edge_threshold = 2.0;
  float surf_threshold = 0.001;

  std::vector<Feature> features;
  features.resize(width * height);

  std::vector<int> label;
  label.resize(width * height);

  std::vector<bool> neighbor_picked;
  neighbor_picked.resize(width * height);

  // calculate curvature
  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      size_t i = u * width + v;
      float diffX = cloud->points[i - 5].x + cloud->points[i - 4].x + cloud->points[i - 3].x + cloud->points[i - 2].x + cloud->points[i - 1].x - 10 * cloud->points[i].x + cloud->points[i + 1].x + cloud->points[i + 2].x + cloud->points[i + 3].x + cloud->points[i + 4].x + cloud->points[i + 5].x;
      float diffY = cloud->points[i - 5].y + cloud->points[i - 4].y + cloud->points[i - 3].y + cloud->points[i - 2].y + cloud->points[i - 1].y - 10 * cloud->points[i].y + cloud->points[i + 1].y + cloud->points[i + 2].y + cloud->points[i + 3].y + cloud->points[i + 4].y + cloud->points[i + 5].y;
      float diffZ = cloud->points[i - 5].z + cloud->points[i - 4].z + cloud->points[i - 3].z + cloud->points[i - 2].z + cloud->points[i - 1].z - 10 * cloud->points[i].z + cloud->points[i + 1].z + cloud->points[i + 2].z + cloud->points[i + 3].z + cloud->points[i + 4].z + cloud->points[i + 5].z;

      features[i] = Feature(i, diffX * diffX + diffY * diffY + diffZ * diffZ);
      neighbor_picked[i] = false;
      label[i] = 0;
    }
  }

  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      // mark occluded points
      size_t i = u * width + v;
      float depth1 = range(cloud->points[i]);
      float depth2 = range(cloud->points[i + 1]);

      if (depth1 < min_distance || depth1 > max_distance)
      {
        neighbor_picked[i] = true;
      }

      if (depth1 - depth2 > 0.3)
      {
        neighbor_picked[i - 5] = true;
        neighbor_picked[i - 4] = true;
        neighbor_picked[i - 3] = true;
        neighbor_picked[i - 2] = true;
        neighbor_picked[i - 1] = true;
        neighbor_picked[i] = true;

      }

      if (depth2 - depth1 > 0.3)
      {
        neighbor_picked[i + 1] = true;
        neighbor_picked[i + 2] = true;
        neighbor_picked[i + 3] = true;
        neighbor_picked[i + 4] = true;
        neighbor_picked[i + 5] = true;
        neighbor_picked[i + 6] = true;
      }

      // parallel beam
      float diff1 = std::abs(range(cloud->points[i - 1]) - range(cloud->points[i]));
      float diff2 = std::abs(range(cloud->points[i + 1]) - range(cloud->points[i]));

      if (diff1 > 0.02 * range(cloud->points[i]) && diff2 > 0.02 * range(cloud->points[i]))
      {
        neighbor_picked[i] = true;
      }
    }
  }

  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; v += (width / 6))
    {
      auto sp = u * width + v;
      auto ep = sp + width / 6;
      if (ep >= (u+1) * width)
      {
        ep = (u+1) * width - 6;
      }

      std::sort(features.begin()+sp, features.begin()+ep, [&](const auto& f, const auto& f2)
      {
        return f.curvature < f2.curvature;
      });

      int max_edge_features = 20;
      for (int k = ep; k >= sp; --k)
      {
        size_t idx = features[k].idx;
        float curvature = features[k].curvature;

        if (curvature > edge_threshold && !neighbor_picked[idx] && max_edge_features-- != 0)
        {
          label[idx] = 1;
          pc_out_edge->push_back(cloud->points[idx]);

          for (int l = 0; l < 5; l++)
          {
            neighbor_picked[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked[idx + l] = true;
          }
        }
      }

      int max_plane_features = 20;
      for (int k = sp; k < ep; ++k)
      {
        size_t idx = features[k].idx;
        float curvature = features[k].curvature;
        if (curvature < surf_threshold && !neighbor_picked[idx] && max_plane_features-- != 0)
        {
          label[idx] = -1;

          for (int l = 0; l <= 5; l++)
          {
            neighbor_picked[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked[idx + l] = true;
          }
        }

        if (label[idx] < 0)
        {
          pc_out_surf->push_back(cloud->points[idx]);
        }
      }
    }
  }
}

inline void extract_features_lio_sam_intensity(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_edge,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf)
{
  auto width = 1024;
  auto height = 128;
  float edge_threshold = 2.0;
  float surf_threshold = 0.001;

  std::vector<Feature> features;
  features.resize(width * height);

  std::vector<int> label;
  label.resize(width * height);

  std::vector<bool> neighbor_picked;
  neighbor_picked.resize(width * height);

  // calculate curvature
  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      size_t i = u * width + v;
      float diffX = cloud->points[i - 5].x + cloud->points[i - 4].x + cloud->points[i - 3].x + cloud->points[i - 2].x + cloud->points[i - 1].x - 10 * cloud->points[i].x + cloud->points[i + 1].x + cloud->points[i + 2].x + cloud->points[i + 3].x + cloud->points[i + 4].x + cloud->points[i + 5].x;
      float diffY = cloud->points[i - 5].y + cloud->points[i - 4].y + cloud->points[i - 3].y + cloud->points[i - 2].y + cloud->points[i - 1].y - 10 * cloud->points[i].y + cloud->points[i + 1].y + cloud->points[i + 2].y + cloud->points[i + 3].y + cloud->points[i + 4].y + cloud->points[i + 5].y;
      float diffZ = cloud->points[i - 5].z + cloud->points[i - 4].z + cloud->points[i - 3].z + cloud->points[i - 2].z + cloud->points[i - 1].z - 10 * cloud->points[i].z + cloud->points[i + 1].z + cloud->points[i + 2].z + cloud->points[i + 3].z + cloud->points[i + 4].z + cloud->points[i + 5].z;

      features[i] = Feature(i, diffX * diffX + diffY * diffY + diffZ * diffZ);
      neighbor_picked[i] = false;
      label[i] = 0;
    }
  }

  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      // mark occluded points
      size_t i = u * width + v;
      float depth1 = range(cloud->points[i]);
      float depth2 = range(cloud->points[i + 1]);
      float intensity = cloud->points[i].intensity;

      if (depth1 < min_distance || depth1 > max_distance || intensity < 50)
      {
        neighbor_picked[i] = true;
      }

      if (depth1 - depth2 > 0.3)
      {
        neighbor_picked[i - 5] = true;
        neighbor_picked[i - 4] = true;
        neighbor_picked[i - 3] = true;
        neighbor_picked[i - 2] = true;
        neighbor_picked[i - 1] = true;
        neighbor_picked[i] = true;

      }

      if (depth2 - depth1 > 0.3)
      {
        neighbor_picked[i + 1] = true;
        neighbor_picked[i + 2] = true;
        neighbor_picked[i + 3] = true;
        neighbor_picked[i + 4] = true;
        neighbor_picked[i + 5] = true;
        neighbor_picked[i + 6] = true;
      }

      // parallel beam
      float diff1 = std::abs(range(cloud->points[i - 1]) - range(cloud->points[i]));
      float diff2 = std::abs(range(cloud->points[i + 1]) - range(cloud->points[i]));

      if (diff1 > 0.02 * range(cloud->points[i]) && diff2 > 0.02 * range(cloud->points[i]))
      {
        neighbor_picked[i] = true;
      }
    }
  }

  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; v += (width / 6))
    {
      auto sp = u * width + v;
      auto ep = sp + width / 6;
      if (ep >= (u+1) * width)
      {
        ep = (u+1) * width - 6;
      }

      std::sort(features.begin()+sp, features.begin()+ep, [&](const auto& f, const auto& f2)
      {
        return f.curvature < f2.curvature;
      });

      int max_edge_features = 20;
      for (int k = ep; k >= sp; --k)
      {
        size_t idx = features[k].idx;
        float curvature = features[k].curvature;
        //float intensity = features[k].intensity;

        if (curvature > edge_threshold && !neighbor_picked[idx] && max_edge_features-- != 0)
        {
          label[idx] = 1;
          pc_out_edge->push_back(cloud->points[idx]);

          for (int l = 0; l < 5; l++)
          {
            neighbor_picked[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked[idx + l] = true;
          }
        }
      }

      int max_plane_features = 20;
      for (int k = sp; k < ep; ++k)
      {
        size_t idx = features[k].idx;
        float curvature = features[k].curvature;
        if (curvature < surf_threshold && !neighbor_picked[idx] && max_plane_features-- != 0)
        {
          label[idx] = -1;

          for (int l = 0; l <= 5; l++)
          {
            neighbor_picked[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked[idx + l] = true;
          }
        }

        if (label[idx] < 0)
        {
          pc_out_surf->push_back(cloud->points[idx]);
        }
      }
    }
  }
}

inline void extract_features_lio_sam_custom(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_edge,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_out_surf)
{
  auto width = 1024;
  auto height = 128;
  float edge_threshold = 2.0;
  float surf_threshold = 0.1;

  static std::vector<Feature> features;
  features.resize(width * height);

  static std::vector<float> ranges;
  ranges.resize(width * height);

  static std::vector<bool> neighbor_picked;
  neighbor_picked.resize(width * height);

  // calculate curvature
  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      size_t i = u * width + v;
      float diffX = cloud->points[i - 5].x + cloud->points[i - 4].x + cloud->points[i - 3].x + cloud->points[i - 2].x + cloud->points[i - 1].x - 10 * cloud->points[i].x + cloud->points[i + 1].x + cloud->points[i + 2].x + cloud->points[i + 3].x + cloud->points[i + 4].x + cloud->points[i + 5].x;
      float diffY = cloud->points[i - 5].y + cloud->points[i - 4].y + cloud->points[i - 3].y + cloud->points[i - 2].y + cloud->points[i - 1].y - 10 * cloud->points[i].y + cloud->points[i + 1].y + cloud->points[i + 2].y + cloud->points[i + 3].y + cloud->points[i + 4].y + cloud->points[i + 5].y;
      float diffZ = cloud->points[i - 5].z + cloud->points[i - 4].z + cloud->points[i - 3].z + cloud->points[i - 2].z + cloud->points[i - 1].z - 10 * cloud->points[i].z + cloud->points[i + 1].z + cloud->points[i + 2].z + cloud->points[i + 3].z + cloud->points[i + 4].z + cloud->points[i + 5].z;

      ranges[i] = range(cloud->points[i]);
      features[i] = Feature(i, diffX * diffX + diffY * diffY + diffZ * diffZ);
      neighbor_picked[i] = false;
    }
  }

  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      // mark occluded points
      size_t i = u * width + v;
      float distance = ranges[i];
      float next_distance = ranges[i + 1];
      float prev_distance = ranges[i - 1];

      if (distance < min_distance || distance > max_distance)
      {
        neighbor_picked[i] = true;
      }

      if (distance - next_distance > 0.3)
      {
        neighbor_picked[i - 5] = true;
        neighbor_picked[i - 4] = true;
        neighbor_picked[i - 3] = true;
        neighbor_picked[i - 2] = true;
        neighbor_picked[i - 1] = true;
        neighbor_picked[i] = true;

      }

      float dist_temp = next_distance - distance;
      if (dist_temp > 0.3)
      {
        neighbor_picked[i + 1] = true;
        neighbor_picked[i + 2] = true;
        neighbor_picked[i + 3] = true;
        neighbor_picked[i + 4] = true;
        neighbor_picked[i + 5] = true;
        neighbor_picked[i + 6] = true;
      }

      // parallel beam
      float diff1 = std::abs(prev_distance - distance);
      float diff2 = std::abs(dist_temp);

      if (diff1 > 0.02 * distance && diff2 > 0.02 * distance)
      {
        neighbor_picked[i] = true;
      }
    }
  }

  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; v += (width / 6))
    {
      auto sp = u * width + v;
      auto ep = sp + width / 6;
      if (ep >= (u+1) * width)
      {
        ep = (u+1) * width - 6;
      }

      std::sort(features.begin()+sp, features.begin()+ep, [&](const auto& f, const auto& f2)
      {
        return f.curvature < f2.curvature;
      });

      int max_edge_features = 20;
      for (int k = ep; k >= sp; --k)
      {
        size_t idx = features[k].idx;
        float curvature = features[k].curvature;

        if (curvature >= edge_threshold && !neighbor_picked[idx] && max_edge_features-- != 0)
        {
          pc_out_edge->push_back(cloud->points[idx]);

          for (int l = 0; l < 5; l++)
          {
            neighbor_picked[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked[idx + l] = true;
          }
        }
      }

      int max_plane_features = 20;
      for (int k = sp; k < ep; ++k)
      {
        size_t idx = features[k].idx;
        float curvature = features[k].curvature;
        if (curvature <= surf_threshold && !neighbor_picked[idx] && max_plane_features-- != 0)
        {
          for (int l = 0; l <= 5; l++)
          {
            neighbor_picked[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked[idx + l] = true;
          }

          pc_out_surf->push_back(cloud->points[idx]);
        }
      }
    }
  }
}

} // end namespace custom