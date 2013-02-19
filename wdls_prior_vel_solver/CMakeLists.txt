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

# # Additional solver.
# Search for dependencies.
ADD_OPTIONAL_DEPENDENCY("soth")

IF(SOTH_FOUND)
  orocos_component(HQPSolver src/HQPSolver.hpp src/HQPSolver.cpp)
  PKG_CONFIG_USE_DEPENDENCY(HQPSolver soth)
ENDIF(SOTH_FOUND)

orocos_generate_package()
