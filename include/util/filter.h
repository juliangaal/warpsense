#pragma once
#include <queue>

/**
 * Filter Base class
 * @tparam T Type of data to be filtered
 */
template<typename T>
class Filter
{
public:
  /// delete move constructor
  Filter(Filter&&) noexcept = delete;

  /// delete copy assignment operator
  Filter& operator=(const Filter& other) = delete;

  /// delete move assignment operator
  Filter& operator=(Filter&&) noexcept = delete;

protected:
  /// Default Construktor
  Filter() = default;

  /// Default Destructor
  virtual ~Filter() = default;

  // default copy constructor
  Filter(const Filter&) = default;

  /**
   * Update performs one filter step
   * @param new_value new measurement
   * @return mean value
   */
  virtual const T &update(const T &new_value) = 0;
};

/**
 * Performs Average calculation across a dynamic window
 * https://de.wikipedia.org/wiki/Gleitender_Mittelwert#Gleitender_Durchschnitt_mit_dynamischem_Fenster
 *
 * @tparam T Datatype to be filtered
 */
template<typename T>
class SlidingWindowFilter : public Filter<T>
{
public:
  /**
   * Construct a Sliding Window Filter
   * @param window_size size of window
   */
  explicit SlidingWindowFilter(size_t window_size)
      : Filter<T>{}
      , window_size_{static_cast<double>(window_size)}
      , buffer_{}
      , mean_{}
  {}

  /**
   * Performs update of average with new data across dynamic window
   *
   * @param new_value type T
   * @return updated mean
   */
  const T &update(const T &new_value)
  {
    /// Window too small for sensible results
    if (window_size_ < 2)
    {
      return new_value;
    }

    buffer_.push(new_value);

    /// Fill buffer until window_size is reached (initialization)
    if (buffer_.size() <= window_size_)
    {
      mean_ += (new_value / window_size_);
      return new_value;
    }

    /// Calculate mean and throw away head of queue
    mean_ += (buffer_.back() - buffer_.front())/window_size_;
    buffer_.pop();
    return mean_;
  }

  /// Get const reference to underlying buffer
  const std::queue<T>& get_buffer() const
  {
    return buffer_;
  }

  /// Get const reference to current mean
  const T& get_mean() const
  {
    return mean_;
  }

private:
  /// Window size: to not risk
  double window_size_;

  /// Queue thats used to implement the dynamic window
  std::queue<T> buffer_;

  /// The current mean
  T mean_;
};