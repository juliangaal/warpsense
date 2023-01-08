#pragma once
#include <vector> // for the vector of measurement variables
#include <memory> // for singleton pointer
#include <exception> // for custom exception
#include <limits> // for maximum value
#include <chrono>
#include <boost/filesystem.hpp>
#include "filter.h"

/**
 * Define this token to enable time measurements in the whole project.
 * Every measurement must be enclosed in
 * #ifdef TIME_MEASUREMENT
 * ...
 * #endif
 */
#define TIME_MEASUREMENT

using measurement_unit = std::chrono::microseconds;

/**
 * Helper struct for managing the different measurement variables for every measured task.
 */
struct TimeEvaluationFormular
{
  /**
   * Constructor for the evaluation formular. All measurement variables are initialized.
   * @param name Name of the task
   */
  explicit TimeEvaluationFormular(std::string name) :
      name(std::move(name)), active(false), accumulate(0),
      count(0), last(0), sum(0), min(std::numeric_limits<unsigned long long>::max()), max(0),
      filter(100) {}
  /// Name of the task that is used as an identifier for the measurement variables
  std::string name;
  /// Flag that indicates whether the task is currently being measured (i.e. it is between start and stop)
  bool active;
  /// Accumulated runtime for the task. The runtime of the methods of the evaluator are not included, so that the evaluator is transparent.
  unsigned long long accumulate;

  /// Number of measurements
  unsigned int count;
  /// Last measured runtime
  unsigned long long last;
  /// Sum of all measured runtimes
  unsigned long long sum;
  /// Minimum of all measured runtimes
  unsigned long long min;
  /// Maximum of all measured runtimes
  unsigned long long max;
  /// Gives an Average of the last 100 measurements
  SlidingWindowFilter<double> filter;
};

template <typename T>
struct EvaluationFormular
{
  /**
   * Constructor for the evaluation formular. All measurement variables are initialized.
   * @param name Name of the task
   */
  explicit EvaluationFormular(std::string name) :
      name(std::move(name)),
      count(0), last(0), sum(0), min(std::numeric_limits<T>::max()), max(0),
      filter(5) {}
  /// Name of the task that is used as an identifier for the measurement variables
  std::string name;
  /// Number of measurements
  T count;
  /// Last measured runtime
  T last;
  /// Sum of all measured runtimes
  T sum;
  /// Minimum of all measured runtimes
  T min;
  /// Maximum of all measured runtimes
  T max;
  /// Gives an Average of the last 100 measurements
  SlidingWindowFilter<T> filter;
};

/**
 * Custom exception that is thrown when the protocol of calling start and stop is not followed.
 */
struct RuntimeEvaluationException : public std::exception
{
  /**
   * Returns the message, that gives information over what caused the exception to occur.
   * @return Exception message
   */
  [[nodiscard]] const char* what() const noexcept override
  {
    return "Runtime evaluation exception:\nStart was called for an already started measurement or stop was called before calling start!";
  }
};

/**
 * Encapsulates the runtime measurement for different tasks.
 * Different tasks can be measured at the same time and nested. They are distinguished by names.
 * The evaluation is transparent in the runtime measurement, i.e. the runtime of the evaluator itself is not measured.
 * This is achieved by measuring the time only in between every call of the functions and
 * updating the measurements of those tasks, whose measurements are currently active, accordingly.
 * The evaluator is implemented as a singleton (except for the public constructor for measurements in a different thread).
 */
class RuntimeEvaluator
{

public:

  /**
   * Returns the singleton instance. Creates a new one the first time.
   * @return Singleton instance
   */
  static RuntimeEvaluator& get_instance();

  /**
   * Default Destructor
   */
  ~RuntimeEvaluator() = default;

  /**
   * Don't allow copies of the instance by copy constructor to ensure singleton property.
   */
  RuntimeEvaluator(RuntimeEvaluator&) = delete;

  /// delete copy constructor
  RuntimeEvaluator(const RuntimeEvaluator&) = delete;

  /// delete move constructor
  RuntimeEvaluator(RuntimeEvaluator&&) = delete;

  /**
   * Don't allow copies of the instance by assignment operator to ensure singleton property.
   */
  RuntimeEvaluator& operator=(RuntimeEvaluator&) = delete;

  /// delete move assignment operator
  RuntimeEvaluator& operator=(RuntimeEvaluator&&) = delete;

  /**
   * @brief Clear all running time measurements
   */
  void clear();

  /**
   * Starts a new measurent for a task.
   * @param task_name Name of the task, which will be shown in the printed overview
   *                  and is used as an identifier to find the right measurement variables
   * @throws RuntimeEvaluationException if a measurement for that task was already started
   */
  void start(const std::string& task_name);

  void update(const std::string& task_name, int value);

  /**
   * Stops the measurement and updates the variables for a task.
   * This function has to be called after the start function was called for that task.
   * @param task_name Name of the task, which will be shown in the printed overview
   *                  and is used as an identifier to find the right measurement variables
   * @throws RuntimeEvaluationException if the measurement for that task has not been started yet
   */
  void stop(const std::string& task_name);

  /**
   * Getter for the vector of the different measurement variables for every measured task
   * @return Vector of the measurement variables
   */
  const std::vector<TimeEvaluationFormular> get_time_forms();

  const std::vector<EvaluationFormular<int>> get_int_forms();

  /**
   * Returns a string that contains a table with the measurements for every measured task.
   * @return String that contains a table with all measurements
   */
  std::string to_string();

  /**
   * Do not use this constructor except in a different thread!
   * This was previously private to ensure singleton property.
   */
  RuntimeEvaluator();

  bool export_results(const boost::filesystem::path& path);


private:
  bool export_results(const std::string &file);

  /**
   * Pauses the time measurements. The time that has past since the last resume is accumulated for every currently measured task.
   * This method is called at the beginning of every public method.
   */
  void pause();

  /**
   * Resumes the time measurements by simply storing the current time so that it can be used at the next pause.
   * This method is called at the end of every public method.
   */
  void resume();

  /**
   * Try to find the formular with the given task name
   *
   * @param task_name Task name of the wanted formular
   * @return int if found, the index of the formular
   *             else -1
   */
  int find_time_formular(const std::string& task_name);

  int find_int_formular(const std::string& task_name);

  /// Vector of the different int variables for every measured task
  std::vector<TimeEvaluationFormular> time_forms_;

  /// Vector of the different measurement variables for every measured task
  std::vector<EvaluationFormular<int>> int_forms_;

  /// Histogram of the total time
  std::vector<int> histogram_;
  constexpr static int HIST_BUCKET_SIZE = 10;

  constexpr static int FIELD_COUNT = 7;
  const std::string labels_[FIELD_COUNT];

  constexpr static char SEPERATOR = ',';

  /// Temporary variable for storing the start time of the current measurement
  std::chrono::_V2::system_clock::time_point start_;
};

/**
 * Puts a by a given evaluator (using its to_string method) into a given output stream.
 * @param os Output stream in which the evaluator is put
 * @param evaluator Runtime evaluator that is put into the stream
 * @return Output stream with the evaluator
 */
std::ostream& operator<<(std::ostream& os, RuntimeEvaluator& evaluator);
