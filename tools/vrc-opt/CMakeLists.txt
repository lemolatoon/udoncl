add_llvm_executable(vrc-opt vrc-opt.cpp)
llvm_update_compile_flags(vrc-opt)
target_link_libraries(vrc-opt PRIVATE MLIROptLib MLIRDLTIDialect MLIRArithDialect MLIRFuncDialect
  MLIRControlFlowDialect MLIRSCFDialect MLIRMemRefDialect MLIRLLVMDialect
  VRCDialectUdonHL VRCDialectUdonAsm VRCDialectVRC VRCDialectVRCGPU)
vrc_configure_target(vrc-opt)
