cmake_minimum_required(VERSION 2.6.3)

project(WDLSPriorVelSolver)

set (ROS_ROOT $ENV{ROS_ROOT} )
if (ROS_ROOT)
  include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
  set(ROS_BUILD_TYPE Debug) #set type to Debug to get debug info compiled
  rosbuild_init()
  rosbuild_find_ros_package( rtt )
  set( RTT_HINTS HINTS ${rtt_PACKAGE_PATH}/../install )
endif()

find_package(OROCOS-RTT REQUIRED ${RTT_HINTS})

include(${OROCOS-RTT_USE_FILE_PATH}/UseOROCOS-RTT.cmake)

#add_definitions(-DEIGEN2_SUPPORT)
orocos_component(WDLSPriorVelSolver src/WDLSPriorVelSolver.hpp src/WDLSPriorVelSolver.cpp)
orocos_generate_package()
