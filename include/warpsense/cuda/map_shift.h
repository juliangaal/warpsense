#include "warpsense/thread.h"
#include "warpsense/cuda/update_tsdf.h"
#include "util/concurrent_ring_buffer.h"
#include "warpsense/tsdf_mapping.h"

namespace cuda
{

class MapShift : public BackGroundProcess
{
public:
    MapShift(TSDFCuda& tsdf, std::shared_lock& mutex_, ConcurrentRingBuffer<rmagine::Pointi>::Ptr& buffer_);
    ~MapShift();

private:
    void thread_run();
    TSDFCuda &tsdf_;
    std::shared_lock& mutex_;
    ConcurrentRingBuffer<rmagine::Pointi>::Ptr buffer_;
};

} // end namespace cuda