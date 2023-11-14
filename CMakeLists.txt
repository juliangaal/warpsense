cmake_minimum_required(VERSION 3.0.2)
project(warpsense LANGUAGES CUDA CXX C)

## Compile as C++17, necessary for PCL on Noetic
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_STANDARD 11)
set(CATKIN_ENABLE_TEST OFF)
set(CMAKE_BUILD_TYPE Release)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  tf_conversions
  visualization_msgs
  fast_gicp
  cv_bridge
  image_transport
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE) # 3.3 includes CUDA support
find_package(Open3D)
find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED PATHS)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem) # compatibility between CUDA 14 and C++17
find_package(Ceres 2.1.0 EXACT REQUIRED)

# OpenMP for CMake < 3.9
if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(Threads REQUIRED)
  find_package(MPI REQUIRED) # for agx xavier compatibility
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads mpi_cxx tbb)
else()
  find_package(MPI REQUIRED) # for agx xavier compatibility
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} mpi_cxx tbb)
endif()

message(STATUS "Cuda found: ${CUDA_VERSION}")

## Highfive
set(HIGHFIVE_EXAMPLES FALSE CACHE INTERNAL "")
set(HIGHFIVE_UNIT_TESTS FALSE CACHE INTERNAL "")
add_subdirectory(ext/HighFive)
file(REMOVE_RECURSE ext/HighFive/tests)
find_package(HDF5 REQUIRED COMPONENTS CXX C HL)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES warpsense
 CATKIN_DEPENDS nav_msgs pcl_conversions pcl_ros roscpp sensor_msgs tf_conversions visualization_msgs fast_gicp
 DEPENDS PCL
)

# globally available data path
add_compile_definitions(DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")

# map library
add_library(map src/map/hdf5_global_map.cpp src/map/hdf5_local_map.cpp)
target_include_directories(map PUBLIC ${HDF5_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} "ext/HighFive/include" include)
target_link_libraries(map PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES} Eigen3::Eigen)

# runtime evaluator library
add_library(runtime_evaluator src/util/runtime_evaluator.cpp)
target_include_directories(runtime_evaluator PUBLIC ${Boost_INCLUDE_DIRS} include)
target_link_libraries(runtime_evaluator PUBLIC ${Boost_LIBRARIES})

# registration library (cpu)
add_library(cpu_registration src/cpu/registration.cpp src/util/imu_accumulator.cpp)
target_include_directories(cpu_registration PRIVATE include ${catkin_INCLUDE_DIRS})
target_link_libraries(cpu_registration PUBLIC map ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

# open3d 
if (Open3d_FOUND)
  # ROS api
  add_library(open3d_conversions src/visualization/open3d_conversions.cpp)
  target_include_directories(open3d_conversions PRIVATE include ${catkin_INCLUDE_DIRS})
  target_link_libraries(open3d_conversions PUBLIC ${catkin_LIBRARIES} Open3D::Open3D Eigen3::Eigen)

  # large scale voxelgrid
  add_executable(large_scale_voxelgrid src/visualization/large_scale_voxelgrid.cpp)
  target_include_directories(large_scale_voxelgrid PRIVATE include ${catkin_INCLUDE_DIRS})
  target_link_libraries(large_scale_voxelgrid ${catkin_LIBRARIES} open3d_conversions)
endif()

# simulation
# add_executable(simulation src/registration/simulation.cpp src/update_tsdf.cpp)
# target_include_directories(simulation PRIVATE include ${catkin_INCLUDE_DIRS})
# target_link_libraries(simulation PUBLIC map ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

add_library(cuda_cleanup src/warpsense/cuda/cleanup.cu)
target_include_directories(cuda_cleanup PRIVATE include)
set_target_properties(cuda_cleanup PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# compile deviceQuery from cuda samples, force CUDA compilation because file is .cpp, not .cu
set(DEVICE_QUERY_FILE "ext/cuda-samples/Samples/1_Utilities/deviceQuery/deviceQuery.cpp")
set_source_files_properties(${DEVICE_QUERY_FILE} PROPERTIES LANGUAGE CUDA)
add_executable(cuda_device_query ${DEVICE_QUERY_FILE})
target_include_directories(cuda_device_query PRIVATE ext/cuda-samples/Common)
set_target_properties(cuda_device_query PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(cuda_registration src/warpsense/cuda/registration.cu)
target_include_directories(cuda_registration PRIVATE include ${Eigen3_INCLUDE_DIRS})
set_target_properties(cuda_registration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(cuda_update_tsdf src/warpsense/cuda/update_tsdf.cu src/warpsense/cuda/device_map_wrapper.cu)
target_include_directories(cuda_update_tsdf PRIVATE include ${Eigen3_INCLUDE_DIRS})
set_target_properties(cuda_update_tsdf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(cuda_playground_lib src/warpsense/cuda/playground.cu)
target_include_directories(cuda_playground_lib PRIVATE include ${Eigen3_INCLUDE_DIRS})
set_target_properties(cuda_playground_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_LIBS cuda_registration cuda_update_tsdf cuda_cleanup)

# featsense
file(GLOB FEATSENSE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/featsense/*.cpp")
add_executable(featsense ${FEATSENSE_SOURCES} src/warpsense/tsdf_mapping.cpp)
target_include_directories(featsense PRIVATE include ${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ext/nanoflann/include)
target_link_libraries(featsense PUBLIC map ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} Ceres::ceres OpenMP::OpenMP_CXX ${CUDA_LIBS})
target_compile_definitions(featsense PUBLIC USE_CUDA_VGICP)

# warpsense
file(GLOB WARPSENSE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/warpsense/*.cpp")
add_executable(${PROJECT_NAME} ${WARPSENSE_SOURCES} src/util/imu_accumulator.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC map runtime_evaluator ${CUDA_LIBS} ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

# fastsense
add_executable(fastsense src/cpu/fastsense.cpp src/cpu/update_tsdf.cpp)
target_include_directories(fastsense PRIVATE include ${catkin_INCLUDE_DIRS})
target_link_libraries(fastsense PUBLIC map runtime_evaluator cpu_registration ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

# ply_writer
add_executable(pcl_writer src/visualization/pcl_writer.cpp)
target_include_directories(pcl_writer PRIVATE ${catkin_INCLUDE_DIRS} include)
target_link_libraries(pcl_writer PUBLIC ${catkin_LIBRARIES})

# testing
add_subdirectory(test)

# docs
add_subdirectory(docs)
