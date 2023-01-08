
#include <map/hdf5_local_map.h>
#include <cpu/update_tsdf.h>
#include <set>
#include <unordered_map>
#include <omp.h>
#include <unordered_set>

#include <unordered_map>

using ARITH_TYPE = long;

Point to_point(const Vector3f &vec)
{
  return Point(static_cast<int>(std::floor(vec.x())),
               static_cast<int>(std::floor(vec.y())),
               static_cast<int>(std::floor(vec.z())));
}

long floor_long(long val)
{
  return val < 0 ? val - MATRIX_RESOLUTION/2 : val;
}

Pointl floor_long(const Pointl& point)
{
  return Pointl(floor_long(point.x()), floor_long(point.y()), floor_long(point.z()));
}

rmagine::Pointl floor_long(const rmagine::Pointl& point)
{
  return rmagine::Pointl(floor_long(point.x), floor_long(point.y), floor_long(point.z));
}

void update_tsdf(const std::vector<Point> &scan_points, pcl::PointCloud<pcl::Normal>::Ptr &normals,
                 const Eigen::Vector3i &scanner_pos, HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution)
{
  float angle = 45.f / 256.f; // TODO: Scanner FoV as Parameter
  int dz_per_distance = std::tan(angle / 180.f * M_PI) / 2.0 * MATRIX_RESOLUTION;

  int weight_epsilon = tau / 10;

  int thread_count = 1;
  int iteration = 0;

  std::vector<std::unordered_map<Point, TSDFEntry>> values(thread_count);

  int distance_zero_cnt = 0;
  int interpolation_zero_cnt = 0;
#pragma omp parallel num_threads(thread_count)
  {
    int current_thread = omp_get_thread_num();
    auto &local_values = values[current_thread];
    std::unordered_set<Point> visited;

#pragma omp for schedule(static)
    for (int i = 0; i < scan_points.size(); i++)
    {
      Pointl point = scan_points[i].cast<ARITH_TYPE>() * MATRIX_RESOLUTION / map_resolution;
      Pointl normal(normals->points[i].normal_x * MATRIX_RESOLUTION, normals->points[i].normal_y * MATRIX_RESOLUTION, normals->points[i].normal_z * MATRIX_RESOLUTION);
      Pointl direction_vector = point - scanner_pos.cast<ARITH_TYPE>() * MATRIX_RESOLUTION;
      ARITH_TYPE distance = direction_vector.norm();
      if (distance == 0)
      {
        distance_zero_cnt++;
        continue;
      }

      auto cell = scan_points[i] / map_resolution;
      if (!buffer.in_bounds(cell))
      {
        continue;
      }

      Pointl normed_direction_vector = direction_vector * MATRIX_RESOLUTION / distance;

      ARITH_TYPE accuracy = -normal.dot(normed_direction_vector) / MATRIX_RESOLUTION;

      // discard normals perpendicular to the scanner
      if (accuracy == 0)
      {
        continue;
      }
      // flip normals that point away from the scanner
      // TODO not necessary because of demeaning?
      if (accuracy < 0)
      {
        normal = -normal;
        accuracy = -accuracy;
      }

      Pointl axis_a = Pointl::Zero();
      Pointl axis_b = Pointl::Zero();
      axis_a = normal.cross(Pointl(0, 1, 0));
      if (axis_a == Pointl::Zero())
      {
        axis_a = normal.cross(Pointl(1, 0, 0));
      }
      axis_b = normal.cross(axis_a) / MATRIX_RESOLUTION;
      axis_a = normal.cross(axis_b) / MATRIX_RESOLUTION;
      int delta_ab = dz_per_distance * distance * 2 / MATRIX_RESOLUTION + 1;

      long tau_mat = tau * MATRIX_RESOLUTION / map_resolution;
      visited.clear();
      for (ARITH_TYPE len_a = -delta_ab; len_a <= delta_ab; len_a += MATRIX_RESOLUTION / 2)
      {
        for (ARITH_TYPE len_b = -delta_ab; len_b <= delta_ab; len_b += MATRIX_RESOLUTION / 2)
        {
          for (ARITH_TYPE len_n = -tau_mat; len_n <= tau_mat; len_n += MATRIX_RESOLUTION / 2)
          {
            Pointl proj_a = (axis_a * len_a) / MATRIX_RESOLUTION;
            Pointl proj_b = (axis_b * len_b) / MATRIX_RESOLUTION;
            Pointl proj_n = (normal * len_n) / MATRIX_RESOLUTION;
            Pointl proj = point + proj_a + proj_b + proj_n;
            Point index = (floor_long(proj) / MATRIX_RESOLUTION).cast<int>();

            if (visited.find(index) != visited.end())
            {
              continue;
            }

            visited.insert(index);
            if (!buffer.in_bounds(index.x(), index.y(), index.z()))
            {
              continue;
            }

            // use the distance to the center of the cell, since 'proj' can be anywhere in the cell
            Point target_center = index * map_resolution + Point::Constant(map_resolution / 2);
            Point delta = (point * map_resolution / MATRIX_RESOLUTION).cast<int>() - target_center;
            int value = delta.norm();
            if (value > tau)
            {
              break;
            }
            if (normal.dot(delta.cast<ARITH_TYPE>()) > 0)
            {
              value = -value;
            }


            // Calculate the corresponding weight for every TSDF value
            int weight = WEIGHT_RESOLUTION;
            if (value < -weight_epsilon)
            {
              weight = WEIGHT_RESOLUTION * (tau + value) / (tau - weight_epsilon);
            }
            weight = (weight * accuracy) / MATRIX_RESOLUTION;
            if (weight == 0)
            {
              continue;
            }

            auto object = TSDFEntry(value, weight);
            auto existing = local_values.try_emplace(index, object);
            if (!existing.second && fabsf(value) < fabsf(existing.first->second.value()))
            {
              existing.first->second = object;
            }
          }
        }
      }
    }

    // wait for all threads to fill their local_values
#pragma omp barrier
    for (auto &map_entry: local_values)
    {

      bool skip = false;
      for (int i = 0; i < thread_count; i++)
      {
        if (i == current_thread)
        {
          continue;
        }

        auto iter = values[i].find(map_entry.first);
        if (iter != values[i].end() && fabsf(iter->second.value()) < fabsf(map_entry.second.value()))
        {
          skip = true;
          break;
        }
      }
      if (skip)
      {
        continue;
      }

      auto &index = map_entry.first;
      auto value = map_entry.second.value();
      auto weight = map_entry.second.weight();

      auto &entry = buffer.value(index.x(), index.y(), index.z());

      // if both cell already exited, and new value calculated for it -> average da ting
      if (weight > 0 && entry.weight() > 0)
      {
        entry.value((entry.value() * entry.weight() + value * weight) / (entry.weight() + weight));
        entry.weight(std::min(max_weight, entry.weight() +
                                          weight)); // This variable (max_weight) ensures that later changes can still have an influence to the map
      }
        // if this is the first time writing to cell, overwrite with newest values
      else if (weight != 0 && entry.weight() <= 0)
      {
        entry.value(value);
        entry.weight(weight);
      }

    }
  }
}

void update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const std::vector<rmagine::Pointf> &normals,
                 const rmagine::Pointi &scanner_pos, HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution)
{
  float angle = 45.f / 256.f; // TODO: Scanner FoV as Parameter
  int dz_per_distance = std::tan(angle / 180.f * M_PI) / 2.0 * MATRIX_RESOLUTION;

  int weight_epsilon = tau / 10;

  int thread_count = 1;
  int iteration = 0;

  std::vector<std::unordered_map<rmagine::Pointi, TSDFEntry>> values(thread_count);

  int distance_zero_cnt = 0;
  int interpolation_zero_cnt = 0;
#pragma omp parallel num_threads(thread_count)
  {
    int current_thread = omp_get_thread_num();
    auto &local_values = values[current_thread];
    std::unordered_set<rmagine::Pointi> visited;

#pragma omp for schedule(static)
    for (int i = 0; i < scan_points.size(); i++)
    {
      rmagine::Pointl point = scan_points[i].cast<ARITH_TYPE>() * MATRIX_RESOLUTION / map_resolution;
      rmagine::Pointl normal(normals[i].x * MATRIX_RESOLUTION, normals[i].y * MATRIX_RESOLUTION, normals[i].z * MATRIX_RESOLUTION);
      rmagine::Pointl direction_vector = point - scanner_pos.cast<ARITH_TYPE>() * MATRIX_RESOLUTION;
      ARITH_TYPE distance = direction_vector.l2norm();
      if (distance == 0)
      {
        distance_zero_cnt++;
        continue;
      }

      auto cell = scan_points[i] / map_resolution;
      if (!buffer.in_bounds(cell.x, cell.y, cell.z))
      {
        continue;
      }

      rmagine::Pointl normed_direction_vector = direction_vector * MATRIX_RESOLUTION / distance;

      ARITH_TYPE accuracy = -normal.dot(normed_direction_vector) / MATRIX_RESOLUTION;

      // discard normals perpendicular to the scanner
      if (accuracy == 0)
      {
        continue;
      }
      // flip normals that point away from the scanner
      // TODO not necessary because of demeaning?
      if (accuracy < 0)
      {
        normal = -normal;
        accuracy = -accuracy;
      }

      rmagine::Pointl axis_a;
      axis_a.setZeros();
      rmagine::Pointl axis_b;
      axis_b.setZeros();
      axis_a = normal.cross(rmagine::Pointl(0, 1, 0));
      if (axis_a == rmagine::Pointl(0, 0, 0))
      {
        axis_a = normal.cross(rmagine::Pointl(1, 0, 0));
      }
      axis_b = normal.cross(axis_a) / MATRIX_RESOLUTION;
      axis_a = normal.cross(axis_b) / MATRIX_RESOLUTION;
      int delta_ab = dz_per_distance * distance * 2 / MATRIX_RESOLUTION + 1;

      long tau_mat = tau * MATRIX_RESOLUTION / map_resolution;
      visited.clear();
      for (ARITH_TYPE len_a = -delta_ab; len_a <= delta_ab; len_a += MATRIX_RESOLUTION / 2)
      {
        for (ARITH_TYPE len_b = -delta_ab; len_b <= delta_ab; len_b += MATRIX_RESOLUTION / 2)
        {
          for (ARITH_TYPE len_n = -tau_mat; len_n <= tau_mat; len_n += MATRIX_RESOLUTION / 2)
          {
            rmagine::Pointl proj_a = (axis_a * len_a) / MATRIX_RESOLUTION;
            rmagine::Pointl proj_b = (axis_b * len_b) / MATRIX_RESOLUTION;
            rmagine::Pointl proj_n = (normal * len_n) / MATRIX_RESOLUTION;
            rmagine::Pointl proj = point + proj_a + proj_b + proj_n;
            rmagine::Pointi index = (floor_long(proj) / MATRIX_RESOLUTION).cast<int>();

            if (visited.find(index) != visited.end())
            {
              continue;
            }

            visited.insert(index);
            if (!buffer.in_bounds(index.x, index.y, index.z))
            {
              continue;
            }

            // use the distance to the center of the cell, since 'proj' can be anywhere in the cell
            rmagine::Pointi target_center = index * map_resolution + rmagine::Pointi::Constant(map_resolution / 2);
            rmagine::Pointi delta = (point * map_resolution / MATRIX_RESOLUTION).cast<int>() - target_center;
            int value = delta.l2norm();
            if (value > tau)
            {
              break;
            }
            if (normal.dot(delta.cast<ARITH_TYPE>()) > 0)
            {
              value = -value;
            }


            // Calculate the corresponding weight for every TSDF value
            //int weight = 1.0 * accuracy;
            int weight = WEIGHT_RESOLUTION;
            if (value < -weight_epsilon)
            {
              weight = WEIGHT_RESOLUTION * (tau + value) / (tau - weight_epsilon);
            }
            weight = (weight * accuracy) / MATRIX_RESOLUTION;
            if (weight == 0)
            {
              continue;
            }

            auto object = TSDFEntry(value, weight);
            auto existing = local_values.try_emplace(index, object);
            if (!existing.second && fabsf(value) < fabsf(existing.first->second.value()))
            {
              existing.first->second = object;
            }
          }
        }
      }
    }

    // wait for all threads to fill their local_values
#pragma omp barrier
    for (auto &map_entry: local_values)
    {

      bool skip = false;
      for (int i = 0; i < thread_count; i++)
      {
        if (i == current_thread)
        {
          continue;
        }

        auto iter = values[i].find(map_entry.first);
        if (iter != values[i].end() && fabsf(iter->second.value()) < fabsf(map_entry.second.value()))
        {
          skip = true;
          break;
        }
      }
      if (skip)
      {
        continue;
      }

      auto &index = map_entry.first;
      auto value = map_entry.second.value();
      auto weight = map_entry.second.weight();

      auto &entry = buffer.value(index.x, index.y, index.z);

      // if both cell already exited, and new value calculated for it -> average da ting
      if (weight > 0 && entry.weight() > 0)
      {
        entry.value((entry.value() * entry.weight() + value * weight) / (entry.weight() + weight));
        entry.weight(std::min(max_weight, entry.weight() +
                                          weight)); // This variable (max_weight) ensures that later changes can still have an influence to the map
      }
        // if this is the first time writing to cell, overwrite with newest values
      else if (weight != 0 && entry.weight() <= 0)
      {
        entry.value(value);
        entry.weight(weight);
      }

    }
  }
}


void update_tsdf(const std::vector<Point> &scan_points, const Eigen::Vector3i &scanner_pos, const Eigen::Vector3i &up,
                 HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution)
{
  float angle = 45.f / 128.f; // TODO: Scanner FoV as Parameter
  int dz_per_distance = std::tan(angle / 180 * M_PI) / 2.0 * MATRIX_RESOLUTION;

  int weight_epsilon = tau / 10;

  int thread_count = 1;
  int iteration = 0;

  std::vector<std::unordered_map<Point, TSDFEntry>> values(thread_count);

  auto pos = scanner_pos * map_resolution;

  int distance_zero_cnt = 0;
  int interpolation_zero_cnt = 0;
#pragma omp parallel num_threads(thread_count)
  {
    int current_thread = omp_get_thread_num();
    auto &local_values = values[current_thread];

#pragma omp for schedule(static)
    for (const auto &point: scan_points)
    {
      Point direction_vector = point - pos;
      int distance = direction_vector.norm();
      if (distance == 0)
      {
        distance_zero_cnt++;
        continue;
      }

      auto cell = point / map_resolution;
      if (!buffer.in_bounds(cell))
      {
        continue;
      }

      iteration++;

      auto normed_direction_vector = (direction_vector.cast<ARITH_TYPE>() * MATRIX_RESOLUTION) / distance;
      auto interpolation_vector = (normed_direction_vector.cross(normed_direction_vector.cross(up.cast<ARITH_TYPE>()) / MATRIX_RESOLUTION));
      auto interpolation_norm = interpolation_vector.norm();
      if (interpolation_norm == 0)
      {
        interpolation_zero_cnt++;
        continue;
      }
      interpolation_vector = (interpolation_vector * MATRIX_RESOLUTION) / interpolation_norm;

      Point prev;

      for (int len = 1; len <= distance + tau; len += map_resolution / 2)
      {
        Point proj = pos + direction_vector * len / distance;
        Point index = proj / map_resolution;

        if (index.x() == prev.x() && index.y() == prev.y())
        {
          continue;
        }
        prev = index;
        if (!buffer.in_bounds(index.x(), index.y(), index.z()))
        {
          continue;
        }

        // use the distance to the center of the cell, since 'proj' can be anywhere in the cell
        Point target_center = index * map_resolution + Point::Constant(map_resolution / 2);
        long long value = (point - target_center).norm();
        value = std::min(value, (long long)tau);
        if (len > distance)
        {
          value = -value;
        }

        // Calculate the corresponding weight for every TSDF value
        int weight = WEIGHT_RESOLUTION;
        if (value < -weight_epsilon)
        {
          weight = WEIGHT_RESOLUTION * (tau + value) / (tau - weight_epsilon);
        }
        if (weight == 0)
        {
          continue;
        }
        auto object = TSDFEntry(value, weight);
        int delta_z = dz_per_distance * len / MATRIX_RESOLUTION;
        auto iter_steps = (delta_z * 2) / map_resolution + 1;
        auto mid = delta_z / map_resolution;
        auto lowest = (proj - ((delta_z * interpolation_vector) / MATRIX_RESOLUTION).cast<int>());
        auto mid_index = index;

        for (auto step = 0; step < iter_steps; ++step)
        {
          index = (lowest + ((step * map_resolution * interpolation_vector) / MATRIX_RESOLUTION).cast<int>()) / map_resolution;

          if (!buffer.in_bounds(index.x(), index.y(), index.z()))
          {
            continue;
          }

          auto tmp = object;

          //if (mid_index != index)
          if (step != mid)
          {
            tmp.weight(tmp.weight() * -1);
          }

          auto existing = local_values.try_emplace(index, tmp);
          if (!existing.second && (abs(value) < abs(existing.first->second.value()) || existing.first->second.weight() < 0))
          {
            existing.first->second = tmp;
          }
        }
      }
    }

    // wait for all threads to fill their local_values
#pragma omp barrier
    for (auto &map_entry: local_values)
    {

      bool skip = false;
      for (int i = 0; i < thread_count; i++)
      {
        if (i == current_thread)
        {
          continue;
        }

        auto iter = values[i].find(map_entry.first);
        if (iter != values[i].end() && fabsf(iter->second.value()) < fabsf(map_entry.second.value()))
        {
          skip = true;
          break;
        }
      }
      if (skip)
      {
        continue;
      }

      auto &index = map_entry.first;
      auto value = map_entry.second.value();
      auto weight = map_entry.second.weight();

      auto &entry = buffer.value(index.x(), index.y(), index.z());

      // if both cell already exited, and new value calculated for it -> average da ting
      if (weight > 0 && entry.weight() > 0)
      {
        entry.value((entry.value() * entry.weight() + value * weight) / (entry.weight() + weight));
        entry.weight(std::min(max_weight, entry.weight() +
                                          weight)); // This variable (max_weight) ensures that later changes can still have an influence to the map
      }
        // if this is the first time writing to cell, overwrite with newest values
      else if (weight != 0 && entry.weight() <= 0)
      {
        entry.value(value);
        entry.weight(weight);
      }

    }
  }
}

void update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const Eigen::Vector3i &scanner_pos, const Eigen::Vector3i &up,
                 HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution)
{
  float angle = 45.f / 128.f; // TODO: Scanner FoV as Parameter
  int dz_per_distance = std::tan(angle / 180 * M_PI) / 2.0 * MATRIX_RESOLUTION;

  int weight_epsilon = tau / 10;

  int thread_count = omp_get_max_threads();

  std::vector<std::unordered_map<Point, TSDFEntry>> values(thread_count);

  auto pos = scanner_pos * map_resolution;

  int distance_zero_cnt = 0;
  int interpolation_zero_cnt = 0;
#pragma omp parallel num_threads(thread_count)
  {
    int current_thread = omp_get_thread_num();
    auto &local_values = values[current_thread];

#pragma omp for schedule(static)
    for (const auto &point: scan_points)
    {
      auto epoint = *reinterpret_cast<const Point*>(&point);
      Point direction_vector = epoint - pos;
      int distance = direction_vector.norm();
      if (distance == 0)
      {
        distance_zero_cnt++;
        continue;
      }

      auto normed_direction_vector = (direction_vector.cast<ARITH_TYPE>() * MATRIX_RESOLUTION) / distance;
      auto interpolation_vector = (normed_direction_vector.cross(normed_direction_vector.cross(up.cast<ARITH_TYPE>()) / MATRIX_RESOLUTION));
      auto interpolation_norm = interpolation_vector.norm();
      if (interpolation_norm == 0)
      {
        interpolation_zero_cnt++;
        continue;
      }
      interpolation_vector = (interpolation_vector * MATRIX_RESOLUTION) / interpolation_norm;

      Point prev;

      for (int len = 1; len <= distance + tau; len += map_resolution / 2)
      {
        Point proj = pos + direction_vector * len / distance;
        Point index = proj / map_resolution;
        if (index.x() == prev.x() && index.y() == prev.y())
        {
          continue;
        }
        prev = index;
        if (!buffer.in_bounds(index.x(), index.y(), index.z()))
        {
          continue;
        }

        // use the distance to the center of the cell, since 'proj' can be anywhere in the cell
        Point target_center = index * map_resolution + Point::Constant(map_resolution / 2);
        int value = (epoint - target_center).norm();
        value = std::min(value, tau);
        if (len > distance)
        {
          value = -value;
        }

        // Calculate the corresponding weight for every TSDF value
        int weight = WEIGHT_RESOLUTION;
        if (value < -weight_epsilon)
        {
          weight = WEIGHT_RESOLUTION * (tau + value) / (tau - weight_epsilon);
        }
        if (weight == 0)
        {
          continue;
        }
        auto object = TSDFEntry(value, weight);
        int delta_z = dz_per_distance * len / MATRIX_RESOLUTION;
        auto iter_steps = (delta_z * 2) / map_resolution + 1;
        auto mid = delta_z / map_resolution;
        auto lowest = (proj - ((delta_z * interpolation_vector) / MATRIX_RESOLUTION).cast<int>());
        auto mid_index = index;

        for (auto step = 0; step < iter_steps; ++step)
        {
          index = (lowest + ((step * map_resolution * interpolation_vector) / MATRIX_RESOLUTION).cast<int>()) / map_resolution;

          if (!buffer.in_bounds(index.x(), index.y(), index.z()))
          {
            continue;
          }

          auto tmp = object;

          //if (mid_index != index)
          if (step != mid)
          {
            tmp.weight(tmp.weight() * -1);
          }

          auto existing = local_values.try_emplace(index, tmp);
          if (!existing.second && (abs(value) < abs(existing.first->second.value()) || existing.first->second.weight() < 0))
          {
            existing.first->second = tmp;
          }
        }
      }
    }

    // wait for all threads to fill their local_values
#pragma omp barrier
    for (auto &map_entry: local_values)
    {

      bool skip = false;
      for (int i = 0; i < thread_count; i++)
      {
        if (i == current_thread)
        {
          continue;
        }

        auto iter = values[i].find(map_entry.first);
        if (iter != values[i].end() && fabsf(iter->second.value()) < fabsf(map_entry.second.value()))
        {
          skip = true;
          break;
        }
      }
      if (skip)
      {
        continue;
      }

      auto &index = map_entry.first;
      auto value = map_entry.second.value();
      auto weight = map_entry.second.weight();

      auto &entry = buffer.value(index.x(), index.y(), index.z());

      // if both cell already exited, and new value calculated for it -> average da ting
      if (weight > 0 && entry.weight() > 0)
      {
        entry.value((entry.value() * entry.weight() + value * weight) / (entry.weight() + weight));
        entry.weight(std::min(max_weight, entry.weight() +
                                          weight)); // This variable (max_weight) ensures that later changes can still have an influence to the map
      }
        // if this is the first time writing to cell, overwrite with newest values
      else if (weight != 0 && entry.weight() <= 0)
      {
        entry.value(value);
        entry.weight(weight);
      }

    }
  }
}

