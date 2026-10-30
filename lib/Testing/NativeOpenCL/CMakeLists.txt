add_library(VRCNativeOpenCLRuntime STATIC Runtime.cpp)
target_compile_features(VRCNativeOpenCLRuntime PUBLIC cxx_std_17)
set_target_properties(VRCNativeOpenCLRuntime PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(VRCNativeOpenCLRuntime PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>")
target_link_libraries(VRCNativeOpenCLRuntime PUBLIC VRCOpenCLRuntime VRCNativeFragmentRuntime)
