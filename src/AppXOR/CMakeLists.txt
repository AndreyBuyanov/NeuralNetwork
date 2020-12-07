cmake_minimum_required (VERSION 3.0)

project(AppXOR)

file(GLOB HEADERS *.hpp)
file(GLOB SOURSES *.cpp)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURSES})

target_link_libraries(${PROJECT_NAME} PRIVATE LibNN)
