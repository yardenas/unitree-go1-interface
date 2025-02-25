cmake_minimum_required(VERSION 3.11)

# set name of the project
project(unitree_go1_interface)

# set cxx standard and compiler flag
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# ros dependencies
# -----------------------------------------------------------------------------

find_package(crl_unitree_go1 0.8.0 EXACT REQUIRED)
find_package(onnxruntime 1.15.0 EXACT REQUIRED)
find_package(crl_unitree_commons 0.8.0 EXACT REQUIRED)
find_package(crl_unitree_simulator 0.8.0 EXACT REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# -----------------------------------------------------------------------------
# resource files
# -----------------------------------------------------------------------------

# launch 
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

# config 
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

# data
install(DIRECTORY data DESTINATION share/${PROJECT_NAME}/)

# -----------------------------------------------------------------------------
# code
# -----------------------------------------------------------------------------
# executable
add_executable(robot
  src/robot_main.cpp
  src/Node.cpp
  src/NNPolicy.cpp
)

target_include_directories(
  robot
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" #
  "$<INSTALL_INTERFACE:include>"
)

ament_target_dependencies(robot
  crl_unitree_go1 onnxruntime crl_unitree_commons crl_unitree_simulator)

target_link_libraries(robot
  Boost::program_options
)

install(
  TARGETS robot
  DESTINATION lib/${PROJECT_NAME}
)

# -----------------------------------------------------------------------------
ament_export_dependencies(
  crl_unitree_go1 onnxruntime crl_unitree_commons crl_unitree_simulator)
ament_package()
