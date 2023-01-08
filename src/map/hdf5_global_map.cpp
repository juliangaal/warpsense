#include <sstream>
#include "map/hdf5_global_map.h"

HDF5GlobalMap::HDF5GlobalMap(const MapParams &params)
    : file_{std::string(params.filename.c_str()), HighFive::File::OpenOrCreate | HighFive::File::Truncate}, // Truncate clears already existing file
      default_tsdf_entry_{static_cast<TSDFEntry::ValueType>(params.tau), static_cast<TSDFEntry::WeightType>(params.initial_weight)},
      active_chunks_{},
      num_poses_{0}
{
    if (!file_.exist(hdf5_constants::MAP_GROUP_NAME))
    {
        file_.createGroup(hdf5_constants::MAP_GROUP_NAME);
    }

    if (!file_.exist(hdf5_constants::POSES_GROUP_NAME))
    {
      file_.createGroup(hdf5_constants::POSES_GROUP_NAME);
    }

    write_meta(params);
}

HDF5GlobalMap::HDF5GlobalMap(const std::string& name, TSDFEntry::ValueType initial_value, TSDFEntry::WeightType initial_weight)
    : file_{name, HighFive::File::OpenOrCreate | HighFive::File::Truncate}, // Truncate clears already existing file
      default_tsdf_entry_{initial_value, initial_weight},
      active_chunks_{},
      num_poses_{0}
{
  if (!file_.exist(hdf5_constants::MAP_GROUP_NAME))
  {
    file_.createGroup(hdf5_constants::MAP_GROUP_NAME);
  }

  if (!file_.exist(hdf5_constants::POSES_GROUP_NAME))
  {
    file_.createGroup(hdf5_constants::POSES_GROUP_NAME);
  }
}

std::string HDF5GlobalMap::filename() const
{
    return file_.getName();
}


std::string HDF5GlobalMap::tag_from_chunk_pos(const Eigen::Vector3i& pos)
{
    std::stringstream ss;
    ss << pos.x() << "_" << pos.y() << "_" << pos.z();
    return ss.str();
}

int HDF5GlobalMap::index_from_pos(Eigen::Vector3i pos, const Eigen::Vector3i& chunkPos)
{
    pos -= chunkPos * CHUNK_SIZE;
    return (pos.x() * CHUNK_SIZE * CHUNK_SIZE + pos.y() * CHUNK_SIZE + pos.z());
}

std::vector<TSDFEntry::RawType>& HDF5GlobalMap::activate_chunk(const Eigen::Vector3i& chunkPos)
{
    int index = -1;
    int n = active_chunks_.size();
    for (int i = 0; i < n; i++)
    {
        if (active_chunks_[i].pos == chunkPos)
        {
            // chunk is already active
            index = i;
        }
    }
    if (index == -1)
    {
        // chunk is not already active
        ActiveChunk newChunk;
        newChunk.pos = chunkPos;
        newChunk.age = 0;

        HighFive::Group g = file_.getGroup("/map");
        auto tag = tag_from_chunk_pos(chunkPos);
        if (g.exist(tag))
        {
            // read chunk from file
            HighFive::DataSet d = g.getDataSet(tag);
            d.read(newChunk.data);
        }
        else
        {
            // create new chunk
            newChunk.data = std::vector<TSDFEntry::RawType>(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE, default_tsdf_entry_.raw());
        }
        // put new chunk into active_chunks_
        if (n < NUM_CHUNKS)
        {
            // there is still room for active chunks
            index = n;
            newChunk.age = n; // temporarily assign oldest age so that all other ages get incremented
            active_chunks_.push_back(newChunk);
        }
        else
        {
            // write oldest chunk into file
            int max = -1;
            for (int i = 0; i < n; i++)
            {
                if (active_chunks_[i].age > max)
                {
                    max = active_chunks_[i].age;
                    index = i;
                }
            }
            auto tag = tag_from_chunk_pos(active_chunks_[index].pos);

            if (g.exist(tag))
            {
                auto d = g.getDataSet(tag);
                d.write(active_chunks_[index].data);
            }
            else
            {
                g.createDataSet(tag, active_chunks_[index].data);
            }
            // overwrite with new chunk
            active_chunks_[index] = newChunk;
        }
    }
    // update ages
    int age = active_chunks_[index].age;
    for (auto& chunk : active_chunks_)
    {
        if (chunk.age < age)
        {
            chunk.age++;
        }
    }
    active_chunks_[index].age = 0;
    return active_chunks_[index].data;
}

TSDFEntry HDF5GlobalMap::get_value(const Eigen::Vector3i& pos)
{
    Eigen::Vector3i chunkPos = floor_divide(pos, CHUNK_SIZE);
    const auto& chunk = activate_chunk(chunkPos);
    int index = index_from_pos(pos, chunkPos);
    return TSDFEntry(chunk[index]);
}

void HDF5GlobalMap::set_value(const Eigen::Vector3i& pos, const TSDFEntry& value)
{
    Eigen::Vector3i chunkPos = floor_divide(pos, CHUNK_SIZE);
    auto& chunk = activate_chunk(chunkPos);
    int index = index_from_pos(pos, chunkPos);
    chunk[index] = value.raw();
}

void HDF5GlobalMap::write_back()
{
    HighFive::Group g = file_.getGroup("/map");
    for (auto& chunk : active_chunks_)
    {
        auto tag = tag_from_chunk_pos(chunk.pos);

        if (g.exist(tag))
        {
            auto d = g.getDataSet(tag);
            d.write(chunk.data);
        }
        else
        {
            g.createDataSet(tag, chunk.data);
        }
    }
    file_.flush();
}

void HDF5GlobalMap::write_pose(const Eigen::Isometry3d& pose, float scale)
{
  HighFive::Group g = file_.getGroup(hdf5_constants::POSES_GROUP_NAME);

  size_t identfier = g.listObjectNames().size();

  // create a new sub group for the new pose
  auto sub_g = g.createGroup(std::string(hdf5_constants::POSES_GROUP_NAME) + "/" + std::to_string(identfier));

  std::vector<float> values(hdf5_constants::POSE_DATASET_SIZE);

  // round to three decimal places here.
  values[0] = std::round((pose.translation().x() / scale) * 1000.0f) / 1000.0f;
  values[1] = std::round((pose.translation().y() / scale) * 1000.0f) / 1000.0f;
  values[2] = std::round((pose.translation().z() / scale) * 1000.0f) / 1000.0f;

  Eigen::Quaternionf quat(pose.cast<float>().rotation());
  values[3] = std::round(quat.x() * 1000.0f) / 1000.0f;
  values[4] = std::round(quat.y() * 1000.0f) / 1000.0f;
  values[5] = std::round(quat.z() * 1000.0f) / 1000.0f;
  values[6] = std::round(quat.w() * 1000.0f) / 1000.0f;

  sub_g.createDataSet(hdf5_constants::POSE_DATASET_NAME, values);

  file_.flush();
}

void HDF5GlobalMap::write_pose(const Eigen::Matrix4f &pose, float scale)
{
  Eigen::Isometry3d iso(pose.cast<double>());
  write_pose(iso, scale);
}

void HDF5GlobalMap::write_meta(const MapParams &params)
{
  HighFive::Group g = file_.getGroup(hdf5_constants::MAP_GROUP_NAME);

  g.createAttribute("tau", params.tau);
  g.createAttribute("map_size_x", params.size.x());
  g.createAttribute("map_size_y", params.size.y());
  g.createAttribute("map_size_z", params.size.z());
  g.createAttribute("max_distance", params.max_distance);
  g.createAttribute("map_resolution", params.resolution);
  g.createAttribute("max_weight", params.max_weight);

  file_.flush();
}

const TSDFEntry &HDF5GlobalMap::get_default_tsdf_entry() const
{
  return default_tsdf_entry_;
}
