#pragma once
#include <highfive/H5File.hpp>
#include <cmath>
#include <string>
#include <utility>

#include "util.h"
#include "hdf5_constants.h"
#include "tsdf.h"
#include "params/map_params.h"

struct ActiveChunk
{
    std::vector<TSDFEntry::RawType> data;
    Eigen::Vector3i pos;
    int age;
};

/**
 * Global map containing containing truncated signed distance function (tsdf) values and weights.
 * The map is divided into chunks.
 * An HDF5 file is used to store the chunks.
 * Additionally poses can be saved.
 */
class HDF5GlobalMap
{

private:
    /**
     * HDF5 file in which the chunks are stored.
     * The file structure looks like this:
     *
     * file.h5
     * |
     * |-/map
     * | |
     * | |-0_0_0 \
     * | |-0_0_1  \
     * | |-0_1_0    chunk datasets named after their tag
     * | |-0_1_1  /
     * | |-1_0_0 /
     */
    HighFive::File file_;

    /// default tsdf value.
    TSDFEntry default_tsdf_entry_;

    /**
     * Vector of active chunks.
     */
    std::vector<ActiveChunk> active_chunks_;

    /// Number of poses that are saved in the HDF5 file
    int num_poses_;

    /** 
     * Given a position in a chunk the tag of the chunk gets returned.
     * @param pos the position
     * @return tag of the chunk
     */
    std::string tag_from_chunk_pos(const Eigen::Vector3i& pos);

    /**
     * Returns the index of a global position in a chunk.
     * The returned index is that of the tsdf value.
     * The index of the weight is one greater.
     * @param pos the position
     * @return index in the chunk
     */
    int index_from_pos(Eigen::Vector3i pos, const Eigen::Vector3i& chunkPos);

public:

    /// Side length of the cube-shaped chunks
    static constexpr int CHUNK_SIZE = 64;

    /// Maximum number of active chunks.
    static constexpr int NUM_CHUNKS = 64;

    /**
     * Constructor of the global map.
     * It is initialized without chunks.
     * The chunks are instead created dynamically depending on which are used.
     * @param name name with path and extension (.h5) of the HDF5 file in which the map is stored
     * @param params default tsdf value
     * @param initial_weight initial default weight
     */
    HDF5GlobalMap(const MapParams &params);

    HDF5GlobalMap(const std::string& name, TSDFEntry::ValueType initial_value, TSDFEntry::WeightType initial_weight);

    ~HDF5GlobalMap() = default;

    void write_pose(const Eigen::Isometry3d& pose, float scale);

    void write_pose(const Eigen::Matrix4f& pose, float scale);

    void write_meta(const MapParams &params);

    std::string filename() const;

    const TSDFEntry& get_default_tsdf_entry() const;

    /**
     * Returns a value pair consisting of a tsdf value and a weight from the map.
     * @param pos the position
     * @return value pair from the map
     */
    TSDFEntry get_value(const Eigen::Vector3i& pos);

    /**
     * Sets a value pair consisting of a tsdf value and a weight on the map.
     * @param pos the position
     * @param value value pair that is set
     */
    void set_value(const Eigen::Vector3i& pos, const TSDFEntry& value);

    /**
     * Activates a chunk and returns it by reference.
     * If the chunk was already active, it is simply returned.
     * Else the HDF5 file is checked for the chunk.
     * If it also doesn't exist there, a new empty chunk is created.
     * Chunks get replaced and written into the HDF5 file by a LRU strategy.
     * The age of the activated chunk is reset and all ages are updated.
     * @param chunk position of the chunk that gets activated
     * @return reference to the activated chunk
     */
    std::vector<TSDFEntry::RawType>& activate_chunk(const Eigen::Vector3i& chunk);

    /**
     * Writes all active chunks into the HDF5 file.
     */
    void write_back();

};