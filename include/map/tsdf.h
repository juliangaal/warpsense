#pragma once

#include <warpsense/math/common.h>
#include <cstdint>

/**
  * @file tsdf.h
  * @author julian 
  * @date 6/23/22
 */

/**
 * @brief Hardware representation of a TSDF entry in the map. Consists of a TSDF value and a current weight for the update procedure
 *
 */
struct TSDFEntryBase
{
  using ValueType = int16_t;
  using WeightType = int16_t;

  ValueType value;
  WeightType weight;
};



/**
 * @brief Interface for a TSDF entry in hardware.
 *        Entry can accessed via the complete datatype or the hardware representation,
 *        which constists of a value and a weight
 */
class TSDFEntry
{
public:
  using RawType = uint32_t;
  using ValueType = TSDFEntryBase::ValueType;
  using WeightType = TSDFEntryBase::WeightType;

private:

  /// Internally managed datatype
  union
  {
    RawType raw;
    TSDFEntryBase tsdf;
  } data;
  static_assert(sizeof(RawType) == sizeof(TSDFEntryBase));            // raw and TSDF types must be of the same size

public:

  /**
   * @brief Initialize the entry through a weight and a value
   */
  RMAGINE_INLINE_FUNCTION
  TSDFEntry(ValueType value, WeightType weight)
  {
    data.tsdf.value = value;
    data.tsdf.weight = weight;
  }

  /**
   * @brief Initialize the entry through the raw data
   */
  explicit TSDFEntry(RawType raw)
  {
    data.raw = raw;
  }

  // Default behaviour for copy and move

  TSDFEntry() = default;
  TSDFEntry(const TSDFEntry&) = default;
  TSDFEntry(TSDFEntry&&) = default;
  ~TSDFEntry() = default;
  TSDFEntry& operator=(const TSDFEntry&) = default;
  TSDFEntry& operator=(TSDFEntry&&) = default;

  /**
   * @brief Are the two TSDF entries equal?
   */
  bool operator==(const TSDFEntry& rhs) const
  {
    return raw() == rhs.raw();
  }

  /**
   * @brief Get the complete data as raw type
   */
  RMAGINE_INLINE_FUNCTION
  RawType raw() const
  {
    return data.raw;
  }

  /**
   * @brief Set the TSDF entry through the raw type
   */
  void raw(RawType val)
  {
    data.raw = val;
  }

  /**
   * @brief Get the value from the TSDF entry
   */
  RMAGINE_INLINE_FUNCTION
  ValueType value() const
  {
    return data.tsdf.value;
  }

  /**
   * @brief Set th value of the TSDF entry
   */
  RMAGINE_INLINE_FUNCTION
  void value(ValueType val)
  {
    data.tsdf.value = val;
  }

  /**
   * @brief Get the weight of the TSDF entry
   */
  RMAGINE_INLINE_FUNCTION
  WeightType weight() const
  {
    return data.tsdf.weight;
  }

  /**
   * @brief Set the weight of the TSDF entry
   */
  RMAGINE_INLINE_FUNCTION
  void weight(WeightType val)
  {
    data.tsdf.weight = val;
  }
};

static_assert(sizeof(TSDFEntry) == sizeof(TSDFEntryBase));

