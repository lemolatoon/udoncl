add_llvm_executable(vrc-translate vrc-translate.cpp)
llvm_update_compile_flags(vrc-translate)
target_link_libraries(vrc-translate PRIVATE VRCLLVMToHighLevelMLIR MLIRIR LLVMIRReader)
vrc_configure_target(vrc-translate)
