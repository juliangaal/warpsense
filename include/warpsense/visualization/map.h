#pragma once
// ROS stuff
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Header.h>
#include <vector>
#include <omp.h>

// local lib
#include "warpsense/types.h"
#include "map/hdf5_local_map.h"

inline void publish_local_map(ros::Publisher &pub, const HDF5LocalMap &values, int tau, int map_resolution)
{
  int thread_count = omp_get_max_threads();
  std::vector<std::pair<geometry_msgs::Point, std_msgs::ColorRGBA>> results[thread_count];

  int left[3], right[3];
  auto size = values.get_size();
  auto pos = values.get_pos();
  for (int i = 0; i < 3; i++)
  {
    left[i] = pos[i] - size[i] / 2;
    right[i] = pos[i] + size[i] / 2;
  }

#pragma omp parallel num_threads(thread_count)
  {
    auto& result = results[omp_get_thread_num()];
    std_msgs::ColorRGBA color;
    color.a = 1;
    color.b = 0;

#pragma omp for collapse(3) schedule(static)
    for (int x = left[0]; x <= right[0]; x++)
    {
      for (int y = left[1]; y <= right[1]; y++)
      {
        for (int z = left[2]; z <= right[2]; z++)
        {
          try
          {
            auto& val = values.value(x, y, z);
            if (val.weight() <= 0 || abs(val.value()) >= tau)
            {
              continue;
            }

            geometry_msgs::Point point;
            point.x = (float)x * (float)map_resolution / 1000.f;
            point.y = (float)y * (float)map_resolution / 1000.f;
            point.z = (float)z * (float)map_resolution / 1000.f;
            // color.a = std::min(val.weight(), 1.0f);
            if (val.value() >= 0)
            {
              color.r = val.value() / (float)tau;
              color.g = 0;
            }
            else
            {
              color.r = 0;
              color.g = -val.value() / (float)tau;
            }

            result.emplace_back(point, color);
          }
          catch(std::out_of_range&)
          {
           ROS_ERROR_STREAM("VIS OUT OF RANGE");
          }

        }
      }
    }
  }

  std::vector<int> offsets(thread_count, 0);
  size_t total_results = 0;
  for (int i = 0; i < thread_count; i++)
  {
    offsets[i] = total_results;
    total_results += results[i].size();
  }
  if (total_results == 0)
  {
    ROS_ERROR_STREAM("VIS FOUND NO VALID TSDF VALUES");
    return;
  }

  std::vector<geometry_msgs::Point> points(total_results);
  std::vector<std_msgs::ColorRGBA> colors(total_results);

#pragma omp parallel num_threads(thread_count)
  {
    auto& result = results[omp_get_thread_num()];
    int offset = offsets[omp_get_thread_num()];
    for (int i = 0; i < result.size(); i++)
    {
      auto& p = result[i];
      points[i + offset] = p.first;
      colors[i + offset] = p.second;
    }
  }

  std_msgs::Header header;
  header.frame_id = "map";
  header.stamp = ros::Time::now();

  visualization_msgs::Marker marker;
  marker.header = header;
  marker.header.stamp = ros::Time();
  marker.type = visualization_msgs::Marker::POINTS;
  marker.action = visualization_msgs::Marker::ADD;
  marker.ns = "window";
  marker.id = 0;
  marker.scale.x = marker.scale.y = map_resolution * 0.6 / 1000;
  marker.points = points;
  marker.colors = colors;
  pub.publish(marker);
}

inline void draw_line(std::vector<geometry_msgs::Point>& points, const geometry_msgs::Point& from, const geometry_msgs::Point& to)
{
  points.push_back(from);
  points.push_back(to);
}

inline void draw_line(std::vector<geometry_msgs::Point>& line_strips, const int from[3], const int to[3])
{
  geometry_msgs::Point from_geo;
  from_geo.x = from[0];
  from_geo.y = from[1];
  from_geo.z = from[2];
  geometry_msgs::Point to_geo;
  to_geo.x = to[0];
  to_geo.y = to[1];
  to_geo.z = to[2];
  draw_line(line_strips, from_geo, to_geo);
}

/**
 * Draw rectangle, given bottom left corner
 * tl ---- tr
 *  |     |
 * bl --- br
 * @param points
 * @param bottom_right
 */
inline void draw_rectangle(std::vector<geometry_msgs::Point>& line_strips, const int bottom_right[3], const int dims[3], int z_offset = 0)
{
  const auto& width_x = dims[0];
  const auto& width_y = dims[1];
  const auto& width_z = dims[2];

  geometry_msgs::Point bbr;
  bbr.x = bottom_right[0];
  bbr.y = bottom_right[1];
  bbr.z = bottom_right[2] + z_offset;
  auto btr = bbr;
  btr.x += width_x;
  draw_line(line_strips, bbr, btr);

  auto btl = btr;
  btl.y += width_y;
  draw_line(line_strips, btr, btl);

  auto bbl = btl;
  bbl.x -= width_x;
  draw_line(line_strips, btl, bbl);

  draw_line(line_strips, bbl, bbr);
}

inline void publish_local_map_skeleton(ros::Publisher &pub, const HDF5LocalMap &values, int tau, int map_resolution)
{
  int top_left[3], bottom_right[3];
  auto size = values.get_size();
  auto pos = values.get_pos();
  for (int i = 0; i < 3; i++)
  {
    bottom_right[i] = pos[i] - size[i] / 2;
    bottom_right[i] *= (float)map_resolution / 1000.f;
    top_left[i] = pos[i] + size[i] / 2;
    top_left[i] *= (float)map_resolution / 1000.f;
  }

  int dims[3] = {top_left[0] - bottom_right[0], top_left[1] - bottom_right[1], top_left[2] - bottom_right[2] };
  std::vector<geometry_msgs::Point> line_strips;

  // draw bottom rectangle
  draw_rectangle(line_strips, bottom_right, dims);
  draw_rectangle(line_strips, bottom_right, dims, dims[2]);

  int top_left_low[3];
  top_left_low[0] = top_left[0];
  top_left_low[1] = top_left[1];
  top_left_low[2] = top_left[2] - dims[2];
  draw_line(line_strips, top_left, top_left_low);

  int bottom_right_high[3];
  bottom_right_high[0] = bottom_right[0];
  bottom_right_high[1] = bottom_right[1];
  bottom_right_high[2] = bottom_right[2] + dims[2];
  draw_line(line_strips, bottom_right, bottom_right_high);

  int bottom_left[3];
  bottom_left[0] = top_left[0] - dims[0];
  bottom_left[1] = top_left[1];
  bottom_left[2] = top_left[2];

  int bottom_left_low[3];
  bottom_left_low[0] = bottom_left[0];
  bottom_left_low[1] = bottom_left[1];
  bottom_left_low[2] = bottom_left[2] - dims[2];
  draw_line(line_strips, bottom_left, bottom_left_low);

  int top_right[3];
  top_right[0] = bottom_right[0] + dims[0];
  top_right[1] = bottom_right[1];
  top_right[2] = bottom_right[2];

  int top_right_high[3];
  top_right_high[0] = top_right[0];
  top_right_high[1] = top_right[1];
  top_right_high[2] = top_right[2] + dims[2];
  draw_line(line_strips, top_right, top_right_high);

  std_msgs::Header header;
  header.frame_id = "map";
  header.stamp = ros::Time::now();
  visualization_msgs::Marker marker;
  marker.header = header;
  marker.header.stamp = ros::Time();
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.ns = "skeleton";
  marker.id = 0;
  marker.scale.x = 0.2;
  marker.points = line_strips;
  marker.pose.orientation.w = 1.0;
  marker.color.g = 1;
  marker.color.r = 1;
  marker.color.a = 1;
  pub.publish(marker);
}


