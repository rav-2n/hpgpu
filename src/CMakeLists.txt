cmake_minimum_required(VERSION 3.8)

project(HPGPU LANGUAGES CUDA CXX)


add_executable(${PROJECT_NAME} main.cc)

add_subdirectory(Aggregator)

target_include_directories(${PROJECT_NAME} PRIVATE Aggregator)

target_link_directories(${PROJECT_NAME} PRIVATE Aggregator)

target_link_libraries(${PROJECT_NAME} Aggregator)

add_subdirectory(DataGenerator)

target_include_directories(${PROJECT_NAME} PRIVATE DataGenerator)

target_link_directories(${PROJECT_NAME} PRIVATE DataGenerator)

target_link_libraries(${PROJECT_NAME} DataGenerator)

add_subdirectory(Logger)

target_include_directories(${PROJECT_NAME} PRIVATE Logger)

target_link_directories(${PROJECT_NAME} PRIVATE Logger)

target_link_libraries(${PROJECT_NAME} Logger)

add_subdirectory(DataGeneratorWithCuda)

target_include_directories(${PROJECT_NAME} PRIVATE DataGeneratorWithCuda)

target_link_directories(${PROJECT_NAME} PRIVATE DataGeneratorWithCuda)

target_link_libraries(${PROJECT_NAME} DataGeneratorWithCuda)

add_subdirectory(CudaAggregator)

target_include_directories(${PROJECT_NAME} PRIVATE CudaAggregator)

target_link_directories(${PROJECT_NAME} PRIVATE CudaAggregator)

target_link_libraries(${PROJECT_NAME} CudaAggregator)

add_subdirectory(Utilities)

target_include_directories(${PROJECT_NAME} PRIVATE Utilities)

target_link_directories(${PROJECT_NAME} PRIVATE Utilities)

target_link_libraries(${PROJECT_NAME} Utilities)

add_subdirectory(Validator)

target_include_directories(${PROJECT_NAME} PRIVATE Validator)

target_link_directories(${PROJECT_NAME} PRIVATE Validator)

target_link_libraries(${PROJECT_NAME} Validator)
