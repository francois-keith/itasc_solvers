cmake_minimum_required(VERSION 2.6.3)

project(WDLSPriorVelSolver)

include(cmake/pkg-config.cmake)

set (ROS_ROOT $ENV{ROS_ROOT} )
if (ROS_ROOT)
  include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
  set(ROS_BUILD_TYPE Release) #set type to Debug to get debug info compiled
  rosbuild_init()
  rosbuild_find_ros_package( rtt )
  set( RTT_HINTS HINTS ${rtt_PACKAGE_PATH}/../install )
endif()

find_package(OROCOS-RTT REQUIRED ${RTT_HINTS})

include(${OROCOS-RTT_USE_FILE_PATH}/UseOROCOS-RTT.cmake)

#add_definitions(-DEIGEN2_SUPPORT)
orocos_component(WDLSPriorVelSolver src/WDLSPriorVelSolver.hpp src/WDLSPriorVelSolver.cpp)
orocos_component(HQPVelSolver src/HQPVelSolver.hpp src/HQPVelSolver.cpp)

# Search for dependencies.
ADD_REQUIRED_DEPENDENCY("soth")
PKG_CONFIG_USE_DEPENDENCY(HQPVelSolver soth)

orocos_generate_package()
