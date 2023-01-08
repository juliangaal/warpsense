#pragma once

#include "util/csv_wrapper.h"
#include <filesystem>

struct KDTreeMeasurements
{
  KDTreeMeasurements(const std::filesystem::path& dir, const std::string& description)
  : measurements(dir, description)
  {
    kdtree = measurements.get_instance("kdtree");
  }

  ~KDTreeMeasurements()
  {
    kdtree->add_row(kdtree_time_data);
    kdtree->add_row(kdtree_size_data);
    std::vector<std::string> labels(kdtree_time_data.size());
    std::generate(labels.begin(), labels.end(), [i=0]() mutable { return std::to_string(i++); });
    kdtree->set_header(labels);
  }

  void add_time(float time)
  {
    kdtree_time_data.push_back(time);
  }

  void add_size(float size)
  {
    kdtree_size_data.push_back(size);
  }

  CSVWrapper<float> measurements;
  CSVWrapper<float>::CSVObject* kdtree;
  CSVWrapper<float>::CSVRow kdtree_time_data;
  CSVWrapper<float>::CSVRow kdtree_size_data;
};
