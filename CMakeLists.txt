cmake_minimum_required(VERSION 3.11)

# set name of the project
project(parkour_unitree_go1)

# set cxx standard and compiler flag
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# ros dependencies
# -----------------------------------------------------------------------------

find_package(crl_unitree_go1 0.7.0 EXACT REQUIRED)
find_package(parkour_unitree_commons 0.1.0 EXACT REQUIRED)

# -----------------------------------------------------------------------------
# resource files
# -----------------------------------------------------------------------------

# launch 
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

# config 
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

# -----------------------------------------------------------------------------
# code
# -----------------------------------------------------------------------------

# executable
add_executable(robot src/main.cpp)

ament_target_dependencies(robot 
  crl_unitree_go1 parkour_unitree_commons)

install(
        TARGETS robot
        DESTINATION lib/${PROJECT_NAME}
)

# -----------------------------------------------------------------------------
ament_export_dependencies(
  crl_unitree_go1)
ament_package()