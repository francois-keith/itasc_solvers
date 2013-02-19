include(CTest)

INCLUDE_DIRECTORIES(../src)

orocos_executable(testSolver testSolver.cpp)
TARGET_LINK_LIBRARIES(testSolver WDLSPriorVelSolver HQPVelSolver)
add_test(testSolver testSolver)



