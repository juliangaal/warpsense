/**
 * @file runtime_evaluator.cpp
 * @author Marc Eisoldt
 * @author Julian Gaal
 * @author Steffen Hinderink
 */

#include "util/runtime_evaluator.h"
#include "util/filter.h"

#include <sstream> // for output
#include <iostream>
#include <iomanip> // for formatting
#include <cmath>

using namespace std::chrono;
namespace fs = boost::filesystem;

RuntimeEvaluator& RuntimeEvaluator::get_instance()
{
  static RuntimeEvaluator instance;
  return instance;
}

RuntimeEvaluator::RuntimeEvaluator()
    : time_forms_()
    , int_forms_()
    , histogram_(10, 0)
    , labels_{"task", "count", "last", "min", "max", "avg", "run_avg" }
    , start_(high_resolution_clock::now())
{}

void RuntimeEvaluator::pause()
{
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<measurement_unit>(stop - start_);

  // add new interval to all active measurements
  for (auto& form: time_forms_)
  {
    if (form.active)
    {
      form.accumulate += duration.count();
    }
  }
}

void RuntimeEvaluator::resume()
{
  start_ = high_resolution_clock::now();
}

int RuntimeEvaluator::find_time_formular(const std::string& task_name)
{
  for (uint i = 0; i < time_forms_.size(); i++)
  {
    if (time_forms_[i].name == task_name)
    {
      return i;
    }
  }

  return -1;
}

int RuntimeEvaluator::find_int_formular(const std::string& task_name)
{
  for (uint i = 0; i < int_forms_.size(); i++)
  {
    if (int_forms_[i].name == task_name)
    {
      return i;
    }
  }

  return -1;
}

void RuntimeEvaluator::start(const std::string& task_name)
{
#ifdef TIME_MEASUREMENT
  pause();

  // get or create task that is started
  int index = find_time_formular(task_name);

  if (index == -1)
  {
    index = time_forms_.size();
    time_forms_.emplace_back(task_name);
  }
  else if (time_forms_[index].active)
  {
    throw RuntimeEvaluationException();
  }

  // start
  time_forms_[index].active = true;
  time_forms_[index].accumulate = 0;

  resume();
#endif
}

void RuntimeEvaluator::update(const std::string &task_name, const int value)
{
#ifdef TIME_MEASUREMENT
  pause();

  // get or create task that is started
  int index = find_int_formular(task_name);

  if (index == -1)
  {
    index = int_forms_.size();
    int_forms_.emplace_back(task_name);
  }

  if (value < int_forms_[index].min)
  {
    int_forms_[index].min = value;
  }
  if (value > int_forms_[index].max && int_forms_[index].count != 1) // ignore first
  {
    int_forms_[index].max = value;
  }

  // start
  int_forms_[index].count++;
  int_forms_[index].last = value;
  int_forms_[index].sum += value;
  int_forms_[index].filter.update(value);

  resume();
#endif
}

void RuntimeEvaluator::clear()
{
  time_forms_.clear();
  int_forms_.clear();
}

void RuntimeEvaluator::stop(const std::string& task_name)
{
#ifdef TIME_MEASUREMENT
  pause();

  // get task that is stopped
  auto index = find_time_formular(task_name);

  if (index == -1 || !time_forms_[index].active)
  {
    throw RuntimeEvaluationException();
  }

  // stop
  unsigned long long time = time_forms_[index].accumulate;
  time_forms_[index].active = false;
  time_forms_[index].count++;
  time_forms_[index].last = time;
  time_forms_[index].sum += time;
  time_forms_[index].filter.update(time);
  if (time < time_forms_[index].min)
  {
    time_forms_[index].min = time;
  }
  if (time > time_forms_[index].max && time_forms_[index].count != 1) // ignore first
  {
    time_forms_[index].max = time;
  }

  if (task_name == "total")
  {
    double time_ms = time / 1000.0;

    size_t bucket = time_ms / HIST_BUCKET_SIZE;
    bucket = std::min(bucket, histogram_.size() - 1);
    histogram_[bucket]++;
  }

  resume();
#endif
}

const std::vector<TimeEvaluationFormular> RuntimeEvaluator::get_time_forms()
{
  return time_forms_;
}

const std::vector<EvaluationFormular<int>> RuntimeEvaluator::get_int_forms()
{
  return int_forms_;
}

std::string RuntimeEvaluator::to_string()
{
  pause();

  size_t width = 0;
  for (const auto& ef : time_forms_)
  {
    width = std::max(width, ef.name.length());
  }
  for (const auto& ef : int_forms_)
  {
    width = std::max(width, ef.name.length());
  }
  for (int i = 0; i < FIELD_COUNT; i++)
  {
    width = std::max(width, labels_[i].length());
  }

  std::stringstream ss;
  ss << std::setfill(' ') << "\n";
  for (int i = 0; i < FIELD_COUNT; i++)
  {
    ss << std::setw(width) << labels_[i] << (i == FIELD_COUNT - 1 ? "\n" : " | ");
  }
  ss << std::setfill('-');
  for (int i = 0; i < FIELD_COUNT; i++)
  {
    ss << std::setw(width) << "" << (i == FIELD_COUNT - 1 ? "-\n" : "-+-");
  }
  ss << std::setfill(' ');

  for (const auto& ef : time_forms_)
  {
    if (ef.active)
    {
      continue;
    }
    unsigned long long avg = ef.sum / ef.count;
    unsigned long long run_avg = (int)ef.filter.get_mean();
    unsigned long long values[] = { ef.last, ef.min, ef.max, avg, run_avg };
    ss << std::setw(width) << ef.name << " | "
       << std::setw(width) << ef.count << " | ";
    for (int i = 0; i < FIELD_COUNT - 2; i++)
    {
      ss << std::setw(width) << values[i] / 1000 << (i == FIELD_COUNT - 3 ? "\n" : " | ");
    }
  }

  ss << std::setfill('-');
  for (int i = 0; i < FIELD_COUNT; i++)
  {
    ss << std::setw(width) << "" << (i == FIELD_COUNT - 1 ? "-\n" : "-+-");
  }
  ss << std::setfill(' ');

  for (const auto& ef : int_forms_)
  {
    int avg = ef.sum / ef.count;
    int run_avg = ef.filter.get_mean();
    int values[] = { ef.last, ef.min, ef.max, avg, run_avg };
    ss << std::setw(width) << ef.name << " | "
       << std::setw(width) << ef.count << " | ";
    for (int i = 0; i < FIELD_COUNT - 2; i++)
    {
      ss << std::setw(width) << values[i] << (i == FIELD_COUNT - 3 ? "\n" : " | ");
    }
  }

  if (!int_forms_.empty())
  {
    ss << std::setfill('-');
    for (int i = 0; i < FIELD_COUNT; i++)
    {
      ss << std::setw(width) << "" << (i == FIELD_COUNT - 1 ? "-\n" : "-+-");
    }
    ss << std::setfill(' ');
  }

  // Histogram of total time
  int total_index = find_time_formular("total");
  if (total_index != -1)
  {
    const auto& form = time_forms_[total_index];

    //                  columns with padding    +    separators     -       start of line
    int line_length = FIELD_COUNT * (width + 2) + (FIELD_COUNT - 1) - std::string("10-20: ").length();

    for (size_t i = 0; i < histogram_.size(); i++)
    {
      ss << std::setw(2) << (i * HIST_BUCKET_SIZE) << '-';
      if (i < histogram_.size() - 1)
      {
        ss << std::setw(2) << (i + 1) * HIST_BUCKET_SIZE;
      }
      else
      {
        ss << "  ";
      }
      int count = std::ceil((double)histogram_[i] * line_length / form.count);
      ss << ": " << std::string(count, '=') << "\n";
    }
  }

  resume();
  return ss.str();
}

bool RuntimeEvaluator::export_results(const boost::filesystem::path &path)
{
  return export_results(path.string());
}

bool RuntimeEvaluator::export_results(const std::string &file)
{
  if (file.empty())
  {
    std::cout << "WARNING: received empty filename\n";
    return false;
  }

  std::ofstream f;
  f.open(file);
  if (!f.is_open())
  {
    std::cout << "WARNING: couldn't open file for reading\n";
    return false;
  }

  // header
  for (const auto& label: labels_)
  {
    f << label << SEPERATOR;
  }
  f << "\n";

  // data
  for (const auto& form : time_forms_)
  {
    if (form.active)
    {
      continue;
    }

    unsigned long long avg = form.sum / form.count;
    unsigned long long run_avg = (int)form.filter.get_mean();

    std::array<unsigned long long, 5> time_values = {form.last, form.min, form.max, avg, run_avg };
    f << form.name << SEPERATOR;
    // treat count separately -> no need to divide by thousand
    f << form.count << SEPERATOR;

    for (const auto& val: time_values)
    {
      f << val / 1000 << SEPERATOR;
    }

    f << "\n";
  }

  for (const auto& form : int_forms_)
  {
    int avg = form.sum / form.count;
    int run_avg = (int)form.filter.get_mean();
    int values[] = {form.count, form.last, form.min, form.max, avg, run_avg };

    f << form.name << SEPERATOR;
    for (const auto& val: values)
    {
      f << val << SEPERATOR;
    }

    f << "\n";
  }

  f.close();
  return true;
}

std::ostream& operator<<(std::ostream& os, RuntimeEvaluator& evaluator)
{
#ifdef TIME_MEASUREMENT
  os << evaluator.to_string();
#endif
  return os;
}
