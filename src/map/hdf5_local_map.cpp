
#include <map/hdf5_local_map.h>
#include <stdlib.h> // for abs
#include <stdexcept>

HDF5LocalMap::HDF5LocalMap(unsigned int sX, unsigned int sY, unsigned int sZ, const std::shared_ptr<HDF5GlobalMap>& map)
    : size_{static_cast<int>(sX % 2 == 1 ? sX : sX + 1),
            static_cast<int>(sY % 2 == 1 ? sY : sY + 1),
            static_cast<int>(sZ % 2 == 1 ? sZ : sZ + 1)},
      data_{new TSDFEntry[size_.prod()]},
      pos_{Eigen::Vector3i::Zero()},
      offset_{size_ / 2},
      map_{map}
{
    auto default_entry = map_->get_value(Eigen::Vector3i(0, 0, 0));
    for (int i = 0; i < size_.x() * size_.y() * size_.z(); i++)
    {
        data_[i] = default_entry;
    }
}

HDF5LocalMap::HDF5LocalMap(const HDF5LocalMap& rhs)
{
  int size = rhs.size_.prod();
  this->data_ = new TSDFEntry[size];
  std::copy(rhs.data_, rhs.data_ + size, this->data_);
  this->size_ = rhs.size_;
  this->pos_ = rhs.pos_;
  this->offset_ = rhs.offset_;
  this->map_ = rhs.map_;
}

void HDF5LocalMap::swap(HDF5LocalMap& rhs)
{
    std::swap(this->data_, rhs.data_);
    std::swap(this->size_, rhs.size_);
    std::swap(this->pos_, rhs.pos_);
    std::swap(this->offset_, rhs.offset_);
    std::swap(this->map_, rhs.map_);
}

void HDF5LocalMap::fill_from(const HDF5LocalMap& rhs)
{
    auto size = size_.x() * size_.y() * size_.z();
    std::copy(rhs.data_, rhs.data_ + size, this->data_);
    this->data_ = rhs.data_;
    this->size_ = rhs.size_;
    this->pos_ = rhs.pos_;
    this->offset_ = rhs.offset_;
    this->map_ = rhs.map_;
}

void HDF5LocalMap::shift(const Eigen::Vector3i& new_pos)
{
    // Explanation:
    // shifting consists of 3 Steps:
    // #1: Save the region that would be removed to the GlobalMap
    // #2: Perform the shift
    // #3: Load the newly visible regions from the GlobalMap

    Eigen::Vector3i diff = new_pos - pos_;

    assert(std::abs(diff.x()) <= size_.x() &&
           std::abs(diff.y()) <= size_.y() &&
           std::abs(diff.z()) <= size_.z());

    // each axis is treated independently
    for (int axis = 0; axis < 3; axis++)
    {
        if (diff[axis] == 0)
        {
            continue;
        }
        // Step #1:
        // The area that needs to be saved covers the entire Map across two axes:
        //   => [pos - size/2, pos + size/2]
        // and stretches from the edge of the Map up to diff on the current axis
        //   => either  [start, start + (diff - 1)]  or  [end - (diff - 1), end]
        // the -1 is a result of save_load_area() using inclusive ranges
        Eigen::Vector3i start = pos_ - size_ / 2;
        Eigen::Vector3i end = pos_ + size_ / 2;
        if (diff[axis] > 0)
        {
            end[axis] = start[axis] + diff[axis] - 1;
        }
        else
        {
            // would be end - (abs(diff) - 1),  but diff < 0
            start[axis] = end[axis] + diff[axis] + 1;
        }
        save_area(start, end);

        // Step #2:
        // The actual 'shift' is simply adjusting pos and offset to new_pos
        pos_[axis] += diff[axis];
        offset_[axis] = (offset_[axis] + diff[axis] + size_[axis]) % size_[axis];

        // Step #3:
        // The area that needs to be loaded is at the opposite end of the Map from the saved area
        // => below formulas are basically the opposite of above formulas
        // These calculations would usually require severe calculations since the loaded-in area
        // was outside of the bounds of the map and needs to be written into the cells that were
        // saved in #1, but since Step #2 adjusted pos and offset we can just use the regular
        // value() method within save_load_area()
        start = pos_ - size_ / 2;
        end = pos_ + size_ / 2;
        if (diff[axis] > 0)
        {
            start[axis] = end[axis] - (diff[axis] - 1);
        }
        else
        {
            // end = start + abs(diff) - 1,  but diff < 0
            end[axis] = start[axis] - diff[axis] - 1;
        }
        load_area(start, end);
    }
}

template<bool save>
void HDF5LocalMap::save_load_area(const Eigen::Vector3i& bottom_corner, const Eigen::Vector3i& top_corner)
{
    // Explanation: We only want to touch each Chunk once instead of every time that
    // GlobalMap::get/set_value is called.
    // => iterate over all affected Chunks and handle all values within each Chunk

    assert(in_bounds(bottom_corner) && in_bounds(top_corner));

    Eigen::Vector3i start = bottom_corner.cwiseMin(top_corner);
    Eigen::Vector3i end = bottom_corner.cwiseMax(top_corner);

    constexpr int CHUNK_SIZE = HDF5GlobalMap::CHUNK_SIZE;
    Eigen::Vector3i chunk_start = floor_divide(start, CHUNK_SIZE);
    Eigen::Vector3i chunk_end = floor_divide(end, CHUNK_SIZE);

    // The start and end point within chunk_start and chunk_end
    Eigen::Vector3i start_delta = start - chunk_start * CHUNK_SIZE;
    Eigen::Vector3i end_delta = end - chunk_end * CHUNK_SIZE;

    // Is there any good way to iterate over 3 dimensions without 3 for-loops?
    for (int chunk_x = chunk_start.x(); chunk_x <= chunk_end.x(); ++chunk_x)
    {
        for (int chunk_y = chunk_start.y(); chunk_y <= chunk_end.y(); ++chunk_y)
        {
            for (int chunk_z = chunk_start.z(); chunk_z <= chunk_end.z(); ++chunk_z)
            {
                auto& chunk = map_->activate_chunk(Eigen::Vector3i(chunk_x, chunk_y, chunk_z));

                // The usual scenario is to iterate over [0, CHUNK_SIZE) in x,y,z, unless the
                // current chunk is on the boundary of the area.

                // if we are on a lower boundary, start at an offset. Otherwise take the entire Chunk
                int dx_start = chunk_x == chunk_start.x() ? start_delta.x() : 0;
                int dy_start = chunk_y == chunk_start.y() ? start_delta.y() : 0;
                int dz_start = chunk_z == chunk_start.z() ? start_delta.z() : 0;
                // if we are in an upper boundary, end at an offset. Otherwise take the entire Chunk
                int dx_end = chunk_x == chunk_end.x() ? end_delta.x() : CHUNK_SIZE - 1;
                int dy_end = chunk_y == chunk_end.y() ? end_delta.y() : CHUNK_SIZE - 1;
                int dz_end = chunk_z == chunk_end.z() ? end_delta.z() : CHUNK_SIZE - 1;

                Eigen::Vector3i global_pos;

                // The index in a Chunk is calculated as
                // index = pos.x() * CHUNK_SIZE * CHUNK_SIZE + pos.y() * CHUNK_SIZE + pos.z()
                //         \__________ index_x ____________/   \_____ index_y ____/
                int index_x, index_y, index;

                for (int dx = dx_start; dx <= dx_end; ++dx)
                {
                    index_x = dx * CHUNK_SIZE * CHUNK_SIZE;
                    global_pos.x() = chunk_x * CHUNK_SIZE + dx;

                    for (int dy = dy_start; dy <= dy_end; ++dy)
                    {
                        index_y = dy * CHUNK_SIZE;
                        global_pos.y() = chunk_y * CHUNK_SIZE + dy;

                        for (int dz = dz_start; dz <= dz_end; ++dz)
                        {
                            index = index_x + index_y + dz;
                            global_pos.z() = chunk_z * CHUNK_SIZE + dz;

                            if constexpr(save)
                            {
                                // We did a bounds check at the start of the function => unchecked is fine
                                chunk[index] = value_unchecked(global_pos).raw();
                            }
                            else
                            {
                                value_unchecked(global_pos).raw(chunk[index]);
                            }
                        }
                    }
                }
            }
        }
    }
}

TSDFEntry* HDF5LocalMap::get_data()
{
    return data_;
}

const TSDFEntry* HDF5LocalMap::get_data() const
{
  return data_;
}

void HDF5LocalMap::write_back()
{
    Eigen::Vector3i start = pos_ - size_ / 2;
    Eigen::Vector3i end = pos_ + size_ / 2;
    save_area(start, end);

    map_->write_back();
}

HDF5LocalMap::~HDF5LocalMap()
{
  delete[] data_;
}
