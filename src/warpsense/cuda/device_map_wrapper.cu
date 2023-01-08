#include "warpsense/cuda/device_map_wrapper.h"

namespace cuda
{

DeviceMapMemWrapper::DeviceMapMemWrapper(size_t n_voxels)
: n_voxels_(n_voxels)
{
  // allocate inner map
  CHECK(cudaMalloc(&inner_.size_, sizeof(rmagine::Pointi)));
  CHECK(cudaMalloc(&inner_.offset_, sizeof(rmagine::Pointi)));

  CHECK(cudaMalloc(&inner_.data_, n_voxels * sizeof(TSDFEntry)));
  CHECK(cudaMalloc(&inner_.pos_, sizeof(rmagine::Pointi)));

  // allocate device pointer
  CHECK(cudaMalloc((void**) &dev_, sizeof(cuda::DeviceMap)));
}

DeviceMapMemWrapper::DeviceMapMemWrapper(const DeviceMap& existing_map)
: DeviceMapMemWrapper(existing_map.size_->prod())
{
  to_device(existing_map);
}

DeviceMapMemWrapper::~DeviceMapMemWrapper()
{
  CHECK(cudaFree(inner_.size_));
  CHECK(cudaFree(inner_.offset_));
  CHECK(cudaFree(inner_.data_));
  CHECK(cudaFree(inner_.pos_));
  CHECK(cudaFree(dev_));
}

void DeviceMapMemWrapper::to_device(const cuda::DeviceMap &existing_map)
{
  // copy inner map
  CHECK(cudaMemcpy(inner_.size_, existing_map.size_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner_.offset_, existing_map.offset_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner_.data_, existing_map.data_, n_voxels_ * sizeof(TSDFEntry), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner_.pos_, existing_map.pos_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));

  // copy device pointer
  CHECK(cudaMemcpy(dev_, &inner_, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));
}

void DeviceMapMemWrapper::update_params(const cuda::DeviceMap &existing_map)
{
  // copy inner map, params **only**
  CHECK(cudaMemcpy(inner_.size_, existing_map.size_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner_.offset_, existing_map.offset_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner_.pos_, existing_map.pos_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));

  // copy device pointer 
  CHECK(cudaMemcpy(dev_, &inner_, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));
}


void DeviceMapMemWrapper::to_host(cuda::DeviceMap &output, cuda::DeviceMap &inner, const cuda::DeviceMap *dev, size_t n_voxels)
{
  CHECK(cudaMemcpy(&inner, dev, sizeof(cuda::DeviceMap), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(output.size_, inner.size_, sizeof(rmagine::Pointi), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(output.offset_, inner.offset_, sizeof(rmagine::Pointi), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(output.data_, inner.data_, n_voxels * sizeof(TSDFEntry), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(output.pos_, inner.pos_, sizeof(rmagine::Pointi), cudaMemcpyDeviceToHost));
}

cuda::DeviceMap* DeviceMapMemWrapper::to_device(const cuda::DeviceMap &input, cuda::DeviceMap &inner, size_t n_voxels)
{
  cuda::DeviceMap* existing_map_dev;
  CHECK(cudaMalloc(&inner.size_, sizeof(rmagine::Pointi)));
  CHECK(cudaMalloc(&inner.offset_, sizeof(rmagine::Pointi)));
  CHECK(cudaMalloc(&inner.data_, n_voxels * sizeof(TSDFEntry)));
  CHECK(cudaMalloc(&inner.pos_, sizeof(rmagine::Pointi)));
  CHECK(cudaMemcpy(inner.size_, input.size_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner.offset_, input.offset_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner.data_, input.data_, n_voxels * sizeof(TSDFEntry), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(inner.pos_, input.pos_, sizeof(rmagine::Pointi), cudaMemcpyHostToDevice));

  CHECK(cudaMalloc((void **) &existing_map_dev, sizeof(cuda::DeviceMap)));
  CHECK(cudaMemcpy(existing_map_dev, &inner, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));
  return existing_map_dev;
}

void DeviceMapMemWrapper::to_host(const cuda::DeviceMap &existing_map)
{
  CHECK(cudaMemcpy(&inner_, dev_, sizeof(cuda::DeviceMap), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(existing_map.size_, inner_.size_, sizeof(rmagine::Pointi), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(existing_map.offset_, inner_.offset_, sizeof(rmagine::Pointi), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(existing_map.data_, inner_.data_, n_voxels_ * sizeof(TSDFEntry), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(existing_map.pos_, inner_.pos_, sizeof(rmagine::Pointi), cudaMemcpyDeviceToHost));
}

cuda::DeviceMap* DeviceMapMemWrapper::dev() const
{
  return dev_;
}

} // end namespace cuda
