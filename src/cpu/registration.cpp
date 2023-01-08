/**
 * @author Malte Hillmann
 */

#include <util/util.h>
#include <warpsense/math/math.h>
#include <warpsense/registration/util.h>
#include <warpsense/registration/registration.h>
#include <mutex>

using namespace Eigen;
namespace rm = rmagine;

Eigen::Matrix4f
register_cloud(const HDF5LocalMap &map, std::vector<Point> &cloud, const Eigen::Matrix4f &pretransform,
               int max_iterations, float it_weight_gradient, float epsilon, int map_resolution)
{
  Matrix4f total_transform = pretransform;

  float alpha = 0;

  float previous_errors[4] = {0, 0, 0, 0};
  int error = 0;
  int count = 0;

  bool finished = false;

  Matrix6l h = Matrix6l::Zero();
  Point6l g = Point6l::Zero();
  Vector6d xi;

  // instead of splitting and joining threads twice per iteration, stay in a
  // multithreaded environment and guard the appropriate places with #pragma omp single
#pragma omp parallel
  {
    Matrix6l local_h;
    Point6l local_g;
    int local_error;
    int local_count;
    Matrix4i next_transform;

    Point gradient;
    Point6l jacobi;

    for (int i = 0; i < max_iterations && !finished; i++)
    {
      local_h = Matrix6l::Zero();
      local_g = Point6l::Zero();
      local_error = 0;
      local_count = 0;

      Point center = total_transform.block<3, 1>(0, 3).cast<int>();

      next_transform = to_int_mat(total_transform);
      if (i < 2)
      {
        //std::cout << "next\n" << total_transform << "\n" << next_transform << "\n";
      }

#pragma omp for schedule(static) nowait
      for (size_t j = 0; j < cloud.size(); j++)
      {
        Point point = transform_point(cloud[j], next_transform);

        Point buf = point / map_resolution;
        point -= center;

        try
        {
          const auto& current = map.value(buf.x(), buf.y(), buf.z());
          if (current.weight() == 0)
          {
            continue;
          }

          const auto& x_next = map.value(buf.x() + 1, buf.y(), buf.z());
          const auto& x_last = map.value(buf.x() - 1, buf.y(), buf.z());
          const auto& y_next = map.value(buf.x(), buf.y() + 1, buf.z());
          const auto& y_last = map.value(buf.x(), buf.y() - 1, buf.z());
          const auto& z_next = map.value(buf.x(), buf.y(), buf.z() + 1);
          const auto& z_last = map.value(buf.x(), buf.y(), buf.z() - 1);

          gradient = Point::Zero();

          if (x_next.weight() != 0 && x_last.weight() != 0 && !((x_next.value() > 0 && x_last.value() < 0) || (x_next.value() < 0 && x_last.value() > 0)))
          {
            gradient.x() = (x_next.value() - x_last.value()) / 2;
          }
          if (y_next.weight() != 0 && y_last.weight() != 0 && !((y_next.value() > 0 && y_last.value() < 0) || (y_next.value() < 0 && y_last.value() > 0)))
          {
            gradient.y() = (y_next.value() - y_last.value()) / 2;
          }
          if (z_next.weight() != 0 && z_last.weight() != 0 && !((z_next.value() > 0 && z_last.value() < 0) || (z_next.value() < 0 && z_last.value() > 0)))
          {
            gradient.z() = (z_next.value() - z_last.value()) / 2;
          }

          jacobi << point.cross(gradient).cast<long>(), gradient.cast<long>();
          
          if (j < 50 && !(jacobi[0] == 0 && jacobi[1] == 0 && jacobi[2] == 0 && jacobi[3] == 0 && jacobi[4] == 0 && jacobi[5] == 0))
          {
            //printf("jacobi (cpu): %ld %ld %ld %ld %ld %ld @ %d %d %d\n", jacobi[0], jacobi[1], jacobi[2], jacobi[3], jacobi[4], jacobi[5], buf.x(), buf.y(), buf.z());
          }
          local_h += jacobi * jacobi.transpose();
          local_g += jacobi * current.value();
          local_error += abs(current.value());
          local_count++;
        }
        catch (std::out_of_range)
        {
        }
      }
      // write local results back into shared variables
#pragma omp critical
      {
        h += local_h;
        g += local_g;
        error += local_error;
        count += local_count;
      }

      //std::cout << "h: \n" << h << "\n";
      //printf("g: %ld %ld %ld %ld %ld %ld\n", g[0], g[1], g[2], g[3], g[4], g[5]);
      //std::cout << "e: " << error << "\n";
      //std::cout << "c: " << count << "\n";

      // wait for all threads to finish
#pragma omp barrier
      // only execute on a single thread, all others wait
#pragma omp single
      {
        Matrix6d hf = h.cast<double>();
        Vector6d gf = g.cast<double>();

        // W Matrix
        hf += alpha * count * Matrix6d::Identity();

        xi = -hf.inverse() * gf; //-h.completeOrthogonalDecomposition().pseudoInverse() * g;

        //convert xi into transform matrix T
        Matrix4f transform = xi_to_transform(xi, center);

        alpha += it_weight_gradient;

        total_transform = transform * total_transform; //update transform

        float err = (float)error / count;
        if (fabs(err - previous_errors[2]) < epsilon && fabs(err - previous_errors[0]) < epsilon)
        {
          //std::cout << "(CPU) Stopped after " << i << " / " << max_iterations << " Iterations" << std::endl;
          finished = true;
        }
        for (int e = 1; e < 4; e++)
        {
          previous_errors[e - 1] = previous_errors[e];
        }
        previous_errors[3] = err;

        h = Matrix6l::Zero();
        g = Point6l::Zero();
        error = 0;
        count = 0;
      }
    }
  }

  // apply final transformation
  Matrix4i next_transform = to_int_mat(total_transform);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < cloud.size(); i++)
  {
    cloud[i] = transform_point(cloud[i], next_transform);
  }

  return total_transform;
}