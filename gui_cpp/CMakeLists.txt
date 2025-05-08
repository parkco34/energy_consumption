cmake_minimum_required(VERSION 3.10)

project(FilePickerGUI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(FilePicker src/main.cpp)

target_link_libraries(FilePicker Qt6::Widgets)

