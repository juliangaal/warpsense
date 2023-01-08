#include "warpsense/cuda/map_shift.h"

using namespace cuda;

MapShift::MapShift(cuda::TSDFCuda& tsdf, std::shared_lock& mutex, ConcurrentRingBuffer<rmagine::Pointi>& buffer)
: tsdf_(tsdf)
, mutex_(mutex)
, buffer_(buffer)
{
}

MapShift::~MapShift()
{
    stop();
}

MapShift::thread_run()
{
    rmagine::Pointi pos;


    while (running && ros::ok())
    {
        if(!buffer_->pop_nb(&pos, DEFAULT_POP_TIMEOUT))
        {
            continue;
        }
        
        std::shared_lock lock(mutex_);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}