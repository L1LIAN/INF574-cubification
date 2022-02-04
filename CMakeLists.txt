cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)

file(GLOB SRCFILES *.cpp)
file(GLOB HFILES *.h)
add_executable(ex_bin ${SRCFILES} ${HFILES})

target_link_libraries(ex_bin igl::core igl::opengl_glfw igl::opengl_glfw_imgui imgui::imgui)
target_precompile_headers(ex_bin PUBLIC pch.h)
