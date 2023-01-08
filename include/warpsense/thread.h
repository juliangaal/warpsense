#pragma once
#include <thread>

class BackGroundProcess
{
public:
    using UPtr = std::unique_ptr<BackGroundProcess>;

    BackGroundProcess() : worker{}, running{false} {}

    virtual ~BackGroundProcess()
    {
        stop();
    }
    
    BackGroundProcess(BackGroundProcess&) = delete;
    BackGroundProcess(BackGroundProcess&&) = delete;
    BackGroundProcess& operator=(const BackGroundProcess&) = delete;
    BackGroundProcess& operator=(BackGroundProcess&&) = delete;

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
