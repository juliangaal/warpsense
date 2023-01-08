if (CATKIN_ENABLE_TESTING)
    message("Compling ${PROJECT_NAME} test(s)")
    find_package(rostest REQUIRED)

    add_rostest_gtest(test_map map.test map.cpp ../src/cpu/update_tsdf.cpp)
    target_include_directories(test_map PRIVATE ../include ${PCL_INCLUDE_DIRS})
    target_link_libraries(test_map map ${PCL_LIBRARY_DIRS} ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

    add_rostest_gtest(test_params_a params_a.test params_a.cpp)
    target_include_directories(test_params_a PRIVATE ../include)
    target_link_libraries(test_params_a map ${catkin_LIBRARIES})

    add_rostest_gtest(test_params_b params_b.test params_b.cpp)
    target_include_directories(test_params_b PRIVATE ../include)
    target_link_libraries(test_params_b map ${catkin_LIBRARIES})

    add_rostest_gtest(test_params_c params_c.test params_c.cpp)
    target_include_directories(test_params_c PRIVATE ../include)
    target_link_libraries(test_params_c map ${catkin_LIBRARIES})

    add_rostest_gtest(test_concurrent_ring_buffer concurrent_ring_buffer.test concurrent_ring_buffer.cpp)
    target_include_directories(test_concurrent_ring_buffer PRIVATE ../include)
    target_link_libraries(test_concurrent_ring_buffer ${catkin_LIBRARIES})

    add_rostest_gtest(test_imu_accumulator imu_accumulator.test imu_accumulator.cpp ../src/util/imu_accumulator.cpp)
    target_include_directories(test_imu_accumulator PRIVATE ../include ${catkin_INCLUDE_DIRS})
    target_link_libraries(test_imu_accumulator ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

    add_rostest_gtest(test_util util.test util.cpp)
    target_include_directories(test_util PRIVATE ../include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
    target_link_libraries(test_util ${catkin_LIBRARIES} ${catkin_LIBRARIES})

    # add_executable(visualization_node visualization_node.cpp ../src/update_tsdf.cpp)
    # target_include_directories(visualization_node PRIVATE ../include ${catkin_INCLUDE_DIRS})
    # target_link_libraries(visualization_node PUBLIC map ${catkin_LIBRARIES} OpenMP::OpenMP_CXX ${PCL_LIBRARY_DIRS})

    add_executable(pcd2tsdf_node pcd2tsdf.cpp ../src/cpu/update_tsdf.cpp)
    target_include_directories(pcd2tsdf_node PRIVATE ../include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    target_link_libraries(pcd2tsdf_node PUBLIC map cuda_update_tsdf ${catkin_LIBRARIES} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX ${PCL_LIBRARY_DIRS})

    add_executable(kdtree kdtree.cpp)
    target_include_directories(kdtree PRIVATE ../include ../ext/nanoflann/include)
    target_link_libraries(kdtree PUBLIC ikdtree ${PCL_LIBRARIES})

    add_executable(pcd_registration_node pcd_registration.cpp ../src/cpu/update_tsdf.cpp ../src/warpsense/tsdf_mapping.cpp ../src/warpsense/tsdf_registration.cpp)
    target_include_directories(pcd_registration_node PRIVATE ../include ${Boost_INCLUDE_DIRS})
    target_link_libraries(pcd_registration_node PUBLIC map cuda_registration cuda_update_tsdf cuda_cleanup cpu_registration runtime_evaluator ${Boost_LIBRARIES} OpenMP::OpenMP_CXX mpi_cxx)

    add_executable(inc_pose_updater inc_pose_updater.cpp)
    target_include_directories(inc_pose_updater PRIVATE ../include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
    target_link_libraries(inc_pose_updater map ${PCL_LIBRARY_DIRS} ${catkin_LIBRARIES})

    add_executable(feature_compare_node feature_compare.cpp)
    target_include_directories(feature_compare_node PRIVATE ../include ${catkin_INCLUDE_DIRS})
    target_link_libraries(feature_compare_node PUBLIC ${catkin_LIBRARIES} ${PCL_LIBRARY_DIRS} runtime_evaluator)

    add_executable(edge_detection edge_detection.cpp)
    target_include_directories(edge_detection PRIVATE ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)
    target_link_libraries(edge_detection PUBLIC ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

    add_library(cuda_test_lib test.cu ../src/warpsense/cuda/device_map_wrapper.cu)
    target_include_directories(cuda_test_lib PRIVATE ../include)
    set_target_properties(cuda_test_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(test_cuda cuda.test cuda.cpp ../src/cpu/update_tsdf.cpp ../src/cpu/registration.cpp)
    target_include_directories(test_cuda PRIVATE ../include)
    target_link_libraries(test_cuda cuda_playground_lib cuda_test_lib map Eigen3::Eigen OpenMP::OpenMP_CXX)
endif ()
