#pragma once
#include "hdf5_global_map.h"

template<typename T>
T overflow(T val, T max)
{
  if (val >= 2 * max)
  {
    return val - 2 * max;
  }
  else if (val >= max)
  {
    return val - max;
  }
  else
  {
    return val;
  }
}


/**
 * Three dimensional array that can be shifted without needing to copy every entry.
 * This is done by implementing it as a ring in every dimension.
 * It is used to store truncated signed distance function (tsdf) values.
 */
class HDF5LocalMap
{

private:

  /// Pointer to the global map in which the values outside of the buffer are stored
  std::shared_ptr<HDF5GlobalMap> map_;

public:

  /**
   * Side lengths of the local map. They are always odd, so that there is a central cell.
   * The local map contains x * y * z values.
   */
  Eigen::Vector3i size_;

  /// Position (x, y, z) of the center of the cuboid in global coordinates.
  Eigen::Vector3i pos_;

  /**
   * Offset (x, y, z) of the data in the ring.
   * Each variable is the index of the center of the cuboid in the data array in its dimension.
   *
   * If size = 5, pos = 1 (-> indices range from -1 to 3) and offset = 3 (in one dimension),
   * the indices of the global map in the data array are for example:
   * 3 -1  0  1  2
   *          ^
   *       offset
   */
  Eigen::Vector3i offset_;

  /**
   * Constructor of the local map.
   * The position is initialized to (0, 0, 0).
   * The sizes are given as parameters. If they are even, they are initialized as s + 1.
   * The offsets are initialized to size / 2, so that the ring boundarys match the array bounds.
   * @param sX Side length of the local map in the x direction
   * @param sY Side length of the local map in the y direction
   * @param sZ Side length of the local map in the z direction
   * @param map Pointer to the global map
   */
  HDF5LocalMap(unsigned int sX, unsigned int sY, unsigned int sZ, const std::shared_ptr<HDF5GlobalMap> &map);

  /// Actual data of the local map.
  TSDFEntry *data_;

  /**
   * Destructor of the local map.
   * Deletes the array in particular.
   */
  ~HDF5LocalMap();

  /**
   * Copy constructor of the local map.
   * This constructor is needed in the asynchronous shift, update and visualization.
   * In the beginning of the thread the local map is copied
   * so that the the cloud callback and the map thread can work simultaneously.
   */
  HDF5LocalMap(const HDF5LocalMap &);

  /**
   * Deleted assignment operator of the local map.
   */
  HDF5LocalMap &operator=(const HDF5LocalMap &) = delete;

  /**
   * Deleted move constructor of the local map.
   */
  HDF5LocalMap(HDF5LocalMap &&) = delete;

  /**
   * Deleted move assignment operator of the local map.
   */
  HDF5LocalMap &operator=(HDF5LocalMap &&) = delete;

  /**
   * @brief Swaps this local map with another one
   *
   * @param rhs the other map
   */
  void swap(HDF5LocalMap &rhs);

  /**
   * @brief Swaps this local map with another one
   *
   * @param rhs the other map
   */
  void fill_from(const HDF5LocalMap &rhs);

  /**
   * Returns a value from the local map per reference.
   * Throws an exception if the index is out of bounds i.e. if it is more than size / 2 away from the position.
   * @param x x-coordinate of the index in global coordinates
   * @param y y-coordinate of the index in global coordinates
   * @param z z-coordinate of the index in global coordinates
   * @return Value of the local map
   */
  inline TSDFEntry &value(int x, int y, int z)
  {
    return value(Eigen::Vector3i(x, y, z));
  }

  inline const std::shared_ptr<HDF5GlobalMap> &get_global_map() const
  {
    return map_;
  }

  /**
  * @brief Calculate the Index of a Point
  *
  * @param point the Point
  * @return int the index in data_
  */
    inline int get_index(const Eigen::Vector3i& point) const
    {
      const auto& p = point;
      auto x = p.x();
      auto y = p.y();
      auto z = p.z();
      int x_offset = overflow(x - pos_.x() + offset_.x() + size_.x(), size_.x()) * size_.y() * size_.z();
      int y_offset = overflow(y - pos_.y() + offset_.y() + size_.y(), size_.y()) * size_.z();
      int z_offset = overflow(z - pos_.z() + offset_.z() + size_.z(), size_.z());
      int index = x_offset  + y_offset + z_offset;
      return index;
    }

  /**
   * Returns a value from the local map per reference.
   * Throws an exception if the index is out of bounds i.e. if it is more than size / 2 away from the position.
   * @param x x-coordinate of the index in global coordinates
   * @param y y-coordinate of the index in global coordinates
   * @param z z-coordinate of the index in global coordinates
   * @return Value of the local map
   */
  inline const TSDFEntry &value(int x, int y, int z) const
  {
    return value(Eigen::Vector3i(x, y, z));
  }

  /**
   * Returns a value from the local map per reference.
   * Throws an exception if the index is out of bounds i.e. if it is more than size / 2 away from the position.
   * @param p position of the index in global coordinates
   * @return value of the local map
   */
  inline TSDFEntry &value(const Eigen::Vector3i &p)
  {
    if (!in_bounds(p))
    {
      throw std::out_of_range(std::string(
          "Index out of bounds: " + std::to_string(p.x()) + "; " + std::to_string(p.y()) + "; " +
          std::to_string(p.z())));
    }
    return value_unchecked(p);
  }

  /**
   * Returns a value from the local map per reference.
   * Throws an exception if the index is out of bounds i.e. if it is more than size / 2 away from the position.
   * @param p position of the index in global coordinates
   * @return value of the local map
   */
  inline const TSDFEntry &value(const Eigen::Vector3i &p) const
  {
    if (!in_bounds(p))
    {
      throw std::out_of_range(std::string(
          "Index out of bounds: " + std::to_string(p.x()) + "; " + std::to_string(p.y()) + "; " +
          std::to_string(p.z())));
    }
    return value_unchecked(p);
  }

  /**
   * Returns the size of the local map
   * @return size of the local map
   */
  inline const Eigen::Vector3i &get_size() const
  {
    return size_;
  }

  inline Eigen::Vector3i &get_size()
  {
    return size_;
  }


  /**
     * Returns the pos of the local map
     * @return pos of the local map
     */
  inline const Eigen::Vector3i &get_pos() const
  {
    return pos_;
  }

  inline Eigen::Vector3i &get_pos()
  {
    return pos_;
  }

  /**
   * Returns the offset of the local map
   * @return offset of the local map
   */
  inline const Eigen::Vector3i &get_offset() const
  {
    return offset_;
  }

  inline Eigen::Vector3i &get_offset()
  {
    return offset_;
  }

  /**
   * Shifts the local map, so that a new position is the center of the cuboid.
   * Entries, that stay in the buffer, stay in place.
   * Values outside of the buffer are loaded from and stored in the global map.
   * @param new_pos the new position. Must not be more than get_size() units away from get_pos()
   */
  void shift(const Eigen::Vector3i &new_pos);

  /**
   * Checks if x, y and z are within the current range
   *
   * @param x x-coordinate to check
   * @param y y-coordinate to check
   * @param z z-coordinate to check
   * @return true if (x, y, z) is within the area of the buffer
   */
  inline bool in_bounds(int x, int y, int z) const
  {
    return in_bounds(Eigen::Vector3i(x, y, z));
  }

  inline bool in_bounds(int x, int y, int z, int xo, int yo, int zo) const
  {
    return in_bounds(Eigen::Vector3i(x, y, z), Eigen::Vector3i(xo, yo, zo));
  }

  /**
   * Checks if x, y and z are within the current range
   *
   * @param p position of the index in global coordinates
   * @return true if (x, y, z) is within the area of the buffer
   */
  inline bool in_bounds(Eigen::Vector3i p) const
  {
    p = (p - pos_).cwiseAbs();
    return p.x() <= size_.x() / 2 && p.y() <= size_.y() / 2 && p.z() <= size_.z() / 2;
  }

  inline bool in_bounds(Eigen::Vector3i p, Eigen::Vector3i offset) const
  {
    p = (p - pos_).cwiseAbs();
    return p.x() <= (size_.x() / 2 - offset.x()) &&
           p.y() <= (size_.y() / 2 - offset.y()) &&
           p.z() <= (size_.z() / 2 - offset.z());
  }

  /**
   * Returns the buffer in which the actual data of the local map is stored.
   * It can be used with hardware kernels.
   * @return data buffer
   */
  TSDFEntry *get_data();

  const TSDFEntry *get_data() const;

  /**
   * Writes all data into the global map.
   * Calls write_back of the global map to store the data in the file.
   */
  void write_back();

  using Ptr = std::shared_ptr<HDF5LocalMap>;

private:
  /**
   * @brief writes to the global map in an area
   *
   * @param bottom_corner the "bottom" corner of the area (smallest values along all axes); inclusive
   * @param top_corner the "top" corner of the area (biggest values along all axes); inclusive
   */
  inline void save_area(const Eigen::Vector3i &bottom_corner, const Eigen::Vector3i &top_corner)
  {
    save_load_area<true>(bottom_corner, top_corner);
  }

  /**
   * @brief reads from the global map in an area
   *
   * @param bottom_corner the "bottom" corner of the area (smallest values along all axes); inclusive
   * @param top_corner the "top" corner of the area (biggest values along all axes); inclusive
   */
  inline void load_area(const Eigen::Vector3i &bottom_corner, const Eigen::Vector3i &top_corner)
  {
    save_load_area<false>(bottom_corner, top_corner);
  }

  /**
   * @brief writes to or reads from the global map in an area
   *
   * @param bottom_corner the "bottom" corner of the area (smallest values along all axes); inclusive
   * @param top_corner the "top" corner of the area (biggest values along all axes); inclusive
   */
  template<bool save>
  void save_load_area(const Eigen::Vector3i &bottom_corner, const Eigen::Vector3i &top_corner);

  /**
   * @brief see #value but without bounds checking
   *
   * @param p position of the index in global coordinates
   * @return value of the local map
   */
  inline TSDFEntry &value_unchecked(const Eigen::Vector3i &p)
  {
    return data_[get_index(p)];
  }

  /**
   * @brief see #value but without bounds checking
   *
   * @param p position of the index in global coordinates
   * @return value of the local map
   */
  inline const TSDFEntry &value_unchecked(const Eigen::Vector3i &p) const
  {
    return data_[get_index(p)];
  }
};