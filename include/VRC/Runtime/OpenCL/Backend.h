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
/* Zero means the legacy one-word profile; otherwise id is an AOT gather
 * kernel with two buffers and scalar_count four-byte scalar arguments. */
struct udoncl_cl_gather_profile {
  uint32_t id, words, scalar_count;
};
size_t udoncl_cl_buffer_limit(void);
int udoncl_cl_gather_profile(const char *name,
                             struct udoncl_cl_gather_profile *profile);
int udoncl_cl_draw_gather(uint32_t kernel, const unsigned char *input,
                          size_t bytes, const unsigned char *scalars,
                          size_t scalar_bytes, unsigned char *output);
int udoncl_cl_source_matches(const char *source, size_t length);
int udoncl_cl_kernel_matches(const char *name);
int udoncl_cl_draw(uint32_t buffer, uint32_t generation,
                   const unsigned char scalar[4], unsigned char result[4]);
#ifdef __cplusplus
}
#endif
#endif
