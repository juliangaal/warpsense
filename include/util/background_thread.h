#pragma once
#include <thread>
#include <memory>

class BackgroundThread
{
public:
  using UPtr = std::unique_ptr<BackgroundThread>;

  BackgroundThread() : worker{}, running{false} {}

  virtual ~BackgroundThread()
  {
    stop();
  }

  BackgroundThread(BackgroundThread&) = delete;
  BackgroundThread(BackgroundThread&&) = delete;
  BackgroundThread& operator=(const BackgroundThread&) = delete;
  BackgroundThread& operator=(BackgroundThread&&) = delete;

  virtual void start()
  {
    if (!running)
    {
      running = true;
      worker = std::thread([&]()
                           {
                             this->thread_run();
                           });
    }
  }

  virtual void stop()
  {
    if (running && worker.joinable())
    {
      running = false;
      worker.join();
    }
  }

protected:

  virtual void thread_run() = 0;

  /// Worker thread
  std::thread worker;
  /// Flag if the thread is running
  bool running;
};