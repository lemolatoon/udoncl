add_library(VRCOpenCLRuntime STATIC OpenCL.c DeviceMemory.c)
target_compile_features(VRCOpenCLRuntime PRIVATE c_std_11)
set_target_properties(VRCOpenCLRuntime PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
target_include_directories(VRCOpenCLRuntime PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/OpenCL-Headers>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>")
target_compile_definitions(VRCOpenCLRuntime PUBLIC CL_TARGET_OPENCL_VERSION=120)
