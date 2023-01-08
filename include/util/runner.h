#pragma once

/**
 * @brief Runner object: manages starting and stopping ProcessThread objects
 * 
 * @tparam T : object inheriting ProcessThread
 */
template<typename T>
class ThreadRunner
{
private:
  T& object;
public:
  /**
   * @brief Construct a new Runner object: start the thread
   *
   * @param obj
   */
  explicit ThreadRunner(T& obj) : object(obj)
  {
    object.start();
  }

  /**
   * @brief Destroy the Runner object: stop the thread
   */
  ~ThreadRunner()
  {
    object.stop();
  }
};