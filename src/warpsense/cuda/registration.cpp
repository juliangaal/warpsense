#include <Eigen/Dense>
#include <warpsense/math/math.h>
#include <warpsense/map/hdf5_local_map.h>
#include <warpsense/registration/util.h>
#include <warpsense/registration/registration.h>
#include <warpsense/cuda/device_map.h>
#include <warpsense/cuda/registration.h>
#include <util/runtime_evaluator.h>

namespace rm = rmagine;

#define TIME_MEASUREMENT

namespace cuda
{
Eigen::Matrix4f
register_cloud(cuda::RegistrationCuda &cuda_reg, std::vector<rmagine::Pointi> &cloud,
               const Eigen::Matrix4f &pretransform, int max_iterations, float it_weight_gradient,
               float epsilon)
{
  Eigen::Matrix4f total_transform = pretransform;
  const rm::Matrix4x4f* total_transform_cu = reinterpret_cast<rm::Matrix4x4f*>(&total_transform);
  Point center = total_transform.block<3, 1>(0, 3).cast<int>();
  float alpha = 0;
  float previous_errors[4] = {0, 0, 0, 0};
  bool finished = false;
  Vector6d xi;

  std::vector<rm::Point6l> gpu_jacobis;
  std::vector<TSDFEntry::ValueType> gpu_values;
  std::vector<bool> gpu_mask;
  rm::Matrix6x6l gpu_h;
  rm::Point6l gpu_g;
  int gpu_e;
  int gpu_c;

  cuda_reg.prepare_registration(cloud);

#ifdef TIME_MEASUREMENT
  auto &runtime_eval_ = RuntimeEvaluator::get_instance();
  runtime_eval_.start("cu_registration");
#endif
  for (int i = 0; i < max_iterations && !finished; i++)
  {
// Uncomment to measure kernels individualls
//#ifdef TIME_MEASUREMENT
//    runtime_eval_.start("cu_jacobi");
//#endif
//     cuda_reg.jacobi_kernel(total_transform_cu);
//#ifdef TIME_MEASUREMENT
//    runtime_eval_.stop("cu_jacobi");
//    runtime_eval_.start("cu_reduction");
//#endif
//     cuda_reg.reduction_kernel(gpu_h,
//                               gpu_g,
//                               gpu_e,
//                               gpu_c);
    cuda_reg.perform_registration(total_transform_cu,
                                  gpu_h,
                                  gpu_g,
                                  gpu_e,
                                  gpu_c, 0);

    Matrix6d hf = (*reinterpret_cast<Matrix6l *>(&gpu_h)).cast<double>();
    Vector6d gf = (*reinterpret_cast<Point6l *>(&gpu_g)).cast<double>();

    //printf("h (gpu): \n");
    // for (int j = 0; j < 6; ++j)
    //{
    //  for (int k = 0; k < 6; ++k)
    //  {
    //    printf("%ld ", gpu_h.at(j, k));
    //    if (k % 5 == 0 && k != 0)
    //    {
    //      printf("\n");
    //    }
    //  }
    //}

    //printf("g (gpu): %ld %ld %ld %ld %ld %ld\n", gpu_g[0], gpu_g[1], gpu_g[2], gpu_g[3], gpu_g[4], gpu_g[5]);
    //printf("e (gpu): %d\n", gpu_e);
    //printf("c (gpu): %d\n", gpu_c);

    // W Matrix
    hf += alpha * gpu_c * Matrix6d::Identity();

    xi = -hf.inverse() * gf; //-h.completeOrthogonalDecomposition().pseudoInverse() * g;

    //convert xi into transform matrix T
    Eigen::Matrix4f transform = xi_to_transform(xi, center);

    alpha += it_weight_gradient;

    // update transform
    total_transform = transform * total_transform;
    total_transform_cu = reinterpret_cast<rm::Matrix4x4f *>(&total_transform);

    float err = (float) gpu_e / gpu_c;
    if (fabs(err - previous_errors[2]) < epsilon && fabs(err - previous_errors[0]) < epsilon)
    {
      //std::cout << "(GPU) Stopped after " << i << " / " << max_iterations << " Iterations" << std::endl;
      finished = true;
    }

    for (int e = 1; e < 4; e++)
    {
      previous_errors[e - 1] = previous_errors[e];
    }

    previous_errors[3] = err;
  }

#ifdef TIME_MEASUREMENT
  runtime_eval_.stop("cu_registration");
#endif

  return total_transform;
}
}