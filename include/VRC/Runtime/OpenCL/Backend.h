#ifndef VRC_RUNTIME_OPENCL_BACKEND_H
#define VRC_RUNTIME_OPENCL_BACKEND_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Native adapter for the initially supported AOT kernel profile. API/handle
 * logic stays in C; this boundary owns only binding and actual GPU work. */
size_t udoncl_cl_live_objects(void);
int udoncl_cl_source_matches(const char *source, size_t length);
int udoncl_cl_kernel_matches(const char *name);
int udoncl_cl_draw(uint32_t buffer, uint32_t generation,
                   const unsigned char scalar[4], unsigned char result[4]);
#ifdef __cplusplus
}
#endif
#endif
