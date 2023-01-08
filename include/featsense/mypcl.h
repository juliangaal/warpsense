#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

struct PCLMetaData
{
  ::pcl::PCLHeader header;
  std::uint32_t height;
  std::uint32_t width;
  std::vector<::pcl::PCLPointField> fields;
  static_assert(BOOST_ENDIAN_BIG_BYTE || BOOST_ENDIAN_LITTLE_BYTE, "unable to determine system endianness");
  std::uint8_t is_bigendian = BOOST_ENDIAN_BIG_BYTE;
  std::uint32_t point_step;
  std::uint32_t row_step;
  std::uint8_t is_dense;
  size_t size;
};

inline
void copyPointCloud2MetaData(const sensor_msgs::PointCloud2 &pc2, PCLMetaData &pcl_pc2)
{
  pcl_conversions::toPCL(pc2.header, pcl_pc2.header);
  pcl_pc2.height = pc2.height;
  pcl_pc2.width = pc2.width;
  pcl_conversions::toPCL(pc2.fields, pcl_pc2.fields);
  pcl_pc2.is_bigendian = pc2.is_bigendian;
  pcl_pc2.point_step = pc2.point_step;
  pcl_pc2.row_step = pc2.row_step;
  pcl_pc2.is_dense = pc2.is_dense;
  pcl_pc2.size = pc2.data.size();
}

namespace mypcl
{
template<typename PointT>
void
fromROSMsg(const sensor_msgs::PointCloud2 &msg, pcl::PointCloud<PointT> &cloud)
{
  PCLMetaData meta;
  copyPointCloud2MetaData(msg, meta);
  pcl::MsgFieldMap field_map;
  pcl::createMapping<PointT>(meta.fields, field_map);

  // Copy info fields
  cloud.header = meta.header;
  cloud.width = meta.width;
  cloud.height = meta.height;
  cloud.is_dense = meta.is_dense == 1;

  // Copy point data
  std::uint32_t num_points = meta.width * meta.height;
  cloud.points.resize(num_points);
  std::uint8_t *cloud_data = reinterpret_cast<std::uint8_t *>(&cloud.points[0]);

  // Check if we can copy adjacent points in a single memcpy.  We can do so if there
  // is exactly one field to copy and it is the same size as the source and destination
  // point types.
  if (field_map.size() == 1 &&
      field_map[0].serialized_offset == 0 &&
      field_map[0].struct_offset == 0 &&
      field_map[0].size == meta.point_step &&
      field_map[0].size == sizeof(PointT))
  {
    std::uint32_t cloud_row_step = static_cast<std::uint32_t> (sizeof(PointT) * cloud.width);
    const std::uint8_t *msg_data = &msg.data[0];
    // Should usually be able to copy all rows at once
    if (meta.row_step == cloud_row_step)
    {
      memcpy(cloud_data, msg_data, meta.size);
    }
    else
    {
      for (std::uint32_t i = 0; i < meta.height; ++i, cloud_data += cloud_row_step, msg_data += meta.row_step)
        memcpy(cloud_data, msg_data, cloud_row_step);
    }

  }
  else
  {
    // If not, memcpy each group of contiguous fields separately
    for (std::uint32_t row = 0; row < meta.height; ++row)
    {
      const std::uint8_t *row_data = &msg.data[row * meta.row_step];
      for (std::uint32_t col = 0; col < meta.width; ++col)
      {
        const std::uint8_t *msg_data = row_data + col * meta.point_step;
        for (const pcl::detail::FieldMapping &mapping: field_map)
        {
          memcpy(cloud_data + mapping.struct_offset, msg_data + mapping.serialized_offset, mapping.size);
        }
        cloud_data += sizeof(PointT);
      }
    }
  }
}
} // end namespace mypcl
