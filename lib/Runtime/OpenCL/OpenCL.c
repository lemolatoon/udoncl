/* OpenCL 1.2 AOT subset. Synchronous native execution, one device, in-order
 * queues, selected AOT kernels and bounded buffers. Not an ICD implementation.
 */
#include "VRC/Runtime/OpenCL/Backend.h"
#include <CL/cl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct _cl_platform_id {
  int tag;
};
struct _cl_device_id {
  int tag;
};
static struct _cl_platform_id platform = {1};
static struct _cl_device_id device = {1};
struct _cl_context {
  unsigned refs;
  struct _cl_context *next;
};
struct _cl_command_queue {
  unsigned refs;
  cl_context context;
  struct _cl_command_queue *next;
};
struct _cl_program {
  unsigned refs;
  cl_context context;
  char *source;
  size_t length;
  cl_build_status status;
  int cl12_options;
  struct _cl_program *next;
};
struct _cl_kernel {
  unsigned refs;
  cl_program program;
  cl_mem buffer, inputs[15];
  uint32_t buffer_generation, input_generations[15];
  struct udoncl_cl_gather_profile gather;
  unsigned char scalar[56];
  unsigned args;
  struct _cl_kernel *next;
};
struct _cl_mem {
  unsigned refs;
  cl_context context;
  cl_mem_flags flags;
  uint32_t id, generation;
  unsigned char *data, *initialized;
  size_t size, origin;
  cl_mem parent;
  struct _cl_mem *next;
};
struct _cl_event {
  unsigned refs;
  cl_command_queue queue;
  cl_command_type type;
  struct _cl_event *next;
};
static cl_context contexts;
static cl_command_queue queues;
static cl_program programs;
static cl_kernel kernels;
static cl_mem buffers;
static cl_event events;
static uint32_t next_buffer = 1;
#define FIND(type, name, list)                                                 \
  static type name(type value) {                                               \
    type p;                                                                    \
    for (p = list; p; p = p->next)                                             \
      if (p == value)                                                          \
        return p;                                                              \
    return NULL;                                                               \
  }
FIND(cl_context, context_find, contexts)
FIND(cl_command_queue, queue_find, queues)
FIND(cl_program, program_find, programs)
FIND(cl_kernel, kernel_find, kernels)
FIND(cl_mem, buffer_find, buffers)
FIND(cl_event, event_find, events)
#define UNLINK(type, value, list)                                              \
  do {                                                                         \
    type *udoncl_cursor = &list;                                               \
    while (*udoncl_cursor && *udoncl_cursor != value)                          \
      udoncl_cursor = &(*udoncl_cursor)->next;                                 \
    if (*udoncl_cursor)                                                        \
      *udoncl_cursor = value->next;                                            \
  } while (0)
#define ERROR(code)                                                            \
  do {                                                                         \
    if (errcode_ret)                                                           \
      *errcode_ret = (code);                                                   \
    return NULL;                                                               \
  } while (0)
static cl_int info(const void *source, size_t bytes, size_t size, void *value,
                   size_t *required) {
  if (value && size < bytes)
    return CL_INVALID_VALUE;
  if (required)
    *required = bytes;
  if (value)
    memcpy(value, source, bytes);
  return CL_SUCCESS;
}
static cl_int string_info(const char *s, size_t n, void *p, size_t *r) {
  return info(s, strlen(s) + 1, n, p, r);
}
static cl_int waits(cl_context context, cl_uint count, const cl_event *list) {
  cl_uint i;
  if ((!count && list) || (count && !list))
    return CL_INVALID_EVENT_WAIT_LIST;
  for (i = 0; i < count; i++) {
    cl_event e = event_find(list[i]);
    if (!e)
      return CL_INVALID_EVENT_WAIT_LIST;
    if (e->queue->context != context)
      return CL_INVALID_CONTEXT;
  }
  return CL_SUCCESS;
}
static cl_event new_event(cl_command_queue queue, cl_command_type type) {
  cl_event event = (cl_event)calloc(1, sizeof(*event));
  if (!event)
    return NULL;
  event->refs = 1;
  event->queue = queue;
  queue->refs++;
  event->type = type;
  event->next = events;
  events = event;
  return event;
}
CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint count,
                                                 cl_platform_id *values,
                                                 cl_uint *total) {
  if ((!values && !total) || (values && !count))
    return CL_INVALID_VALUE;
  if (total)
    *total = 1;
  if (values)
    values[0] = &platform;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id p,
                                                  cl_platform_info param,
                                                  size_t size, void *value,
                                                  size_t *required) {
  if (p != &platform)
    return CL_INVALID_PLATFORM;
  switch (param) {
  case CL_PLATFORM_PROFILE:
    return string_info("EMBEDDED_PROFILE", size, value, required);
  case CL_PLATFORM_VERSION:
    return string_info("OpenCL 1.2 UdonCL AOT subset (not conformant)", size,
                       value, required);
  case CL_PLATFORM_NAME:
    return string_info("UdonCL native fragment test", size, value, required);
  case CL_PLATFORM_VENDOR:
    return string_info("udoncl", size, value, required);
  case CL_PLATFORM_EXTENSIONS:
    return string_info("", size, value, required);
  default:
    return CL_INVALID_VALUE;
  }
}
CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id p,
                                               cl_device_type type,
                                               cl_uint count,
                                               cl_device_id *values,
                                               cl_uint *total) {
  if (p != &platform)
    return CL_INVALID_PLATFORM;
  if (!type || (type != CL_DEVICE_TYPE_ALL &&
                (type & ~(CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CPU |
                          CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR |
                          CL_DEVICE_TYPE_CUSTOM))))
    return CL_INVALID_DEVICE_TYPE;
  if (!(type & (CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_GPU)))
    return CL_DEVICE_NOT_FOUND;
  if ((!values && !total) || (values && !count))
    return CL_INVALID_VALUE;
  if (total)
    *total = 1;
  if (values)
    values[0] = &device;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id d,
                                                cl_device_info param,
                                                size_t size, void *value,
                                                size_t *required) {
  cl_uint u;
  cl_ulong l;
  size_t s;
  if (d != &device)
    return CL_INVALID_DEVICE;
  switch (param) {
  case CL_DEVICE_NAME:
    return string_info("AOT fragment kernel", size, value, required);
  case CL_DEVICE_VENDOR:
    return string_info("udoncl", size, value, required);
  case CL_DEVICE_VERSION:
    return string_info("OpenCL 1.2 AOT subset (not conformant)", size, value,
                       required);
  case CL_DEVICE_OPENCL_C_VERSION:
    return string_info("OpenCL C 1.2 AOT only", size, value, required);
  case CL_DRIVER_VERSION:
    return string_info("0.1", size, value, required);
  case CL_DEVICE_PROFILE:
    return string_info("EMBEDDED_PROFILE", size, value, required);
  case CL_DEVICE_EXTENSIONS:
    return string_info("", size, value, required);
  case CL_DEVICE_TYPE:
    l = CL_DEVICE_TYPE_GPU;
    return info(&l, sizeof(l), size, value, required);
  case CL_DEVICE_PLATFORM: {
    cl_platform_id p = &platform;
    return info(&p, sizeof(p), size, value, required);
  }
  case CL_DEVICE_ADDRESS_BITS:
    u = 32;
    return info(&u, sizeof(u), size, value, required);
  case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
    u = 3;
    return info(&u, sizeof(u), size, value, required);
  case CL_DEVICE_MAX_WORK_ITEM_SIZES: {
    size_t dims[3] = {1, 1, 1};
    return info(dims, sizeof(dims), size, value, required);
  }
  case CL_DEVICE_MAX_WORK_GROUP_SIZE:
    s = 1;
    return info(&s, sizeof(s), size, value, required);
  case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
    l = udoncl_cl_buffer_limit();
    return info(&l, sizeof(l), size, value, required);
  case CL_DEVICE_AVAILABLE:
    u = CL_TRUE;
    return info(&u, sizeof(u), size, value, required);
  case CL_DEVICE_COMPILER_AVAILABLE:
    u = CL_FALSE;
    return info(&u, sizeof(u), size, value, required);
  default:
    return CL_INVALID_VALUE;
  }
}
CL_API_ENTRY cl_context CL_API_CALL clCreateContext(
    const cl_context_properties *properties, cl_uint count,
    const cl_device_id *devices,
    void(CL_CALLBACK *notify)(const char *, const void *, size_t, void *),
    void *user, cl_int *errcode_ret) {
  cl_context c;
  size_t i;
  if (!notify && user)
    ERROR(CL_INVALID_VALUE);
  if (count != 1 || !devices)
    ERROR(CL_INVALID_VALUE);
  if (devices[0] != &device)
    ERROR(CL_INVALID_DEVICE);
  if (properties)
    for (i = 0; properties[i]; i += 2) {
      if (properties[i] != CL_CONTEXT_PLATFORM)
        ERROR(CL_INVALID_PROPERTY);
      if ((cl_platform_id)properties[i + 1] != &platform)
        ERROR(CL_INVALID_PLATFORM);
      if (i)
        ERROR(CL_INVALID_PROPERTY);
    }
  c = (cl_context)calloc(1, sizeof(*c));
  if (!c)
    ERROR(CL_OUT_OF_HOST_MEMORY);
  c->refs = 1;
  c->next = contexts;
  contexts = c;
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return c;
}
CL_API_ENTRY cl_int CL_API_CALL clRetainContext(cl_context c) {
  if (!context_find(c))
    return CL_INVALID_CONTEXT;
  c->refs++;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context c) {
  if (!context_find(c))
    return CL_INVALID_CONTEXT;
  if (!--c->refs) {
    UNLINK(cl_context, c, contexts);
    free(c);
  }
  return CL_SUCCESS;
}
CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(
    cl_context c, cl_device_id d, cl_command_queue_properties properties,
    cl_int *errcode_ret) {
  cl_command_queue q;
  if (!context_find(c))
    ERROR(CL_INVALID_CONTEXT);
  if (d != &device)
    ERROR(CL_INVALID_DEVICE);
  if (properties)
    ERROR(CL_INVALID_QUEUE_PROPERTIES);
  q = (cl_command_queue)calloc(1, sizeof(*q));
  if (!q)
    ERROR(CL_OUT_OF_HOST_MEMORY);
  q->refs = 1;
  q->context = c;
  c->refs++;
  q->next = queues;
  queues = q;
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return q;
}
CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue q) {
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  q->refs++;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue q, cl_command_queue_info param,
                      size_t size, void *value, size_t *required) {
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  if (param == CL_QUEUE_CONTEXT)
    return info(&q->context, sizeof(q->context), size, value, required);
  if (param == CL_QUEUE_REFERENCE_COUNT)
    return info(&q->refs, sizeof(q->refs), size, value, required);
  return CL_INVALID_VALUE;
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue q) {
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  if (!--q->refs) {
    cl_context c = q->context;
    UNLINK(cl_command_queue, q, queues);
    free(q);
    clReleaseContext(c);
  }
  return CL_SUCCESS;
}
CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context c, cl_mem_flags flags,
                                               size_t size, void *host,
                                               cl_int *errcode_ret) {
  cl_mem m;
  cl_mem_flags access =
      flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
  if (!context_find(c))
    ERROR(CL_INVALID_CONTEXT);
  if (!size || size > udoncl_cl_buffer_limit())
    ERROR(CL_INVALID_BUFFER_SIZE);
  if (flags & ~(CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY))
    ERROR(CL_INVALID_VALUE);
  if (access && (access & (access - 1)))
    ERROR(CL_INVALID_VALUE);
  if (host)
    ERROR(CL_INVALID_HOST_PTR);
  if (!next_buffer)
    ERROR(CL_OUT_OF_RESOURCES);
  m = (cl_mem)calloc(1, sizeof(*m));
  if (!m)
    ERROR(CL_OUT_OF_HOST_MEMORY);
  m->data = (unsigned char *)calloc(size, 1);
  m->initialized = (unsigned char *)calloc(size, 1);
  if (!m->data || !m->initialized) {
    free(m->data);
    free(m->initialized);
    free(m);
    ERROR(CL_OUT_OF_HOST_MEMORY);
  }
  m->size = size;
  m->refs = 1;
  m->context = c;
  c->refs++;
  m->flags = flags;
  m->id = next_buffer++;
  m->generation = m->id;
  m->next = buffers;
  buffers = m;
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return m;
}
CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(cl_mem m, cl_mem_flags flags,
                                                  cl_buffer_create_type type,
                                                  const void *info,
                                                  cl_int *errcode_ret) {
  const cl_buffer_region *region = (const cl_buffer_region *)info;
  cl_mem_flags access =
      flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
  cl_mem_flags parent_access;
  cl_mem child;
  if (!buffer_find(m))
    ERROR(CL_INVALID_MEM_OBJECT);
  if (m->parent)
    ERROR(CL_INVALID_MEM_OBJECT);
  if (type != CL_BUFFER_CREATE_TYPE_REGION || !region)
    ERROR(CL_INVALID_VALUE);
  if (flags & ~(CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY))
    ERROR(CL_INVALID_VALUE);
  if (access && (access & (access - 1)))
    ERROR(CL_INVALID_VALUE);
  parent_access =
      m->flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
  if ((parent_access == CL_MEM_WRITE_ONLY && access != 0 &&
       access != CL_MEM_WRITE_ONLY) ||
      (parent_access == CL_MEM_READ_ONLY && access != 0 &&
       access != CL_MEM_READ_ONLY))
    ERROR(CL_INVALID_VALUE);
  if (region->origin & 3u)
    ERROR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
  if (!region->size || region->origin > m->size ||
      region->size > m->size - region->origin)
    ERROR(CL_INVALID_VALUE);
  child = (cl_mem)calloc(1, sizeof(*child));
  if (!child)
    ERROR(CL_OUT_OF_HOST_MEMORY);
  child->refs = 1;
  child->context = m->context;
  child->flags = access ? access : m->flags;
  child->id = m->id;
  child->generation = m->generation;
  child->data = m->data + region->origin;
  child->initialized = m->initialized + region->origin;
  child->size = region->size;
  child->origin = region->origin;
  child->parent = m;
  ++m->refs;
  child->next = buffers;
  buffers = child;
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return child;
}
CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject(cl_mem m) {
  if (!buffer_find(m))
    return CL_INVALID_MEM_OBJECT;
  m->refs++;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem m) {
  if (!buffer_find(m))
    return CL_INVALID_MEM_OBJECT;
  if (!--m->refs) {
    cl_context c = m->context;
    cl_mem parent = m->parent;
    UNLINK(cl_mem, m, buffers);
    if (!parent) {
      free(m->data);
      free(m->initialized);
    }
    free(m);
    if (parent)
      clReleaseMemObject(parent);
    else
      clReleaseContext(c);
  }
  return CL_SUCCESS;
}
CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context c, cl_uint count, const char **strings,
                          const size_t *lengths, cl_int *errcode_ret) {
  cl_program p;
  size_t total = 0, pos = 0, n;
  cl_uint i;
  if (!context_find(c))
    ERROR(CL_INVALID_CONTEXT);
  if (!count || !strings)
    ERROR(CL_INVALID_VALUE);
  for (i = 0; i < count; i++) {
    if (!strings[i])
      ERROR(CL_INVALID_VALUE);
    n = lengths && lengths[i] ? lengths[i] : strlen(strings[i]);
    if (n > SIZE_MAX - total - 1)
      ERROR(CL_OUT_OF_HOST_MEMORY);
    total += n;
  }
  p = (cl_program)calloc(1, sizeof(*p));
  if (!p)
    ERROR(CL_OUT_OF_HOST_MEMORY);
  p->source = (char *)malloc(total + 1);
  if (!p->source) {
    free(p);
    ERROR(CL_OUT_OF_HOST_MEMORY);
  }
  for (i = 0; i < count; i++) {
    n = lengths && lengths[i] ? lengths[i] : strlen(strings[i]);
    memcpy(p->source + pos, strings[i], n);
    pos += n;
  }
  p->source[total] = 0;
  p->length = total;
  p->status = CL_BUILD_NONE;
  p->refs = 1;
  p->context = c;
  c->refs++;
  p->next = programs;
  programs = p;
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return p;
}
CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program p, cl_uint count, const cl_device_id *devices,
               const char *options,
               void(CL_CALLBACK *notify)(cl_program, void *), void *user) {
  if (!program_find(p))
    return CL_INVALID_PROGRAM;
  if ((!count && devices) || (count && !devices))
    return CL_INVALID_VALUE;
  if (count && (count != 1 || devices[0] != &device))
    return CL_INVALID_DEVICE;
  if (!notify && user)
    return CL_INVALID_VALUE;
  if (options && options[0] && strcmp(options, "-cl-std=CL1.2"))
    return CL_INVALID_BUILD_OPTIONS;
  {
    cl_kernel k;
    for (k = kernels; k; k = k->next)
      if (k->program == p)
        return CL_INVALID_OPERATION;
  }
  p->cl12_options = options && options[0];
  p->status = udoncl_cl_source_matches(p->source, p->length) ? CL_BUILD_SUCCESS
                                                             : CL_BUILD_ERROR;
  {
    cl_int result =
        p->status == CL_BUILD_SUCCESS ? CL_SUCCESS : CL_BUILD_PROGRAM_FAILURE;
    if (notify) {
      clRetainProgram(p);
      notify(p, user);
      clReleaseProgram(p);
    }
    return result;
  }
}
CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program p, cl_device_id d, cl_program_build_info param,
                      size_t size, void *value, size_t *required) {
  if (!program_find(p))
    return CL_INVALID_PROGRAM;
  if (d != &device)
    return CL_INVALID_DEVICE;
  if (param == CL_PROGRAM_BUILD_STATUS)
    return info(&p->status, sizeof(p->status), size, value, required);
  if (param == CL_PROGRAM_BUILD_LOG)
    return string_info(p->status == CL_BUILD_ERROR
                           ? "source is not the configured AOT Device program"
                           : "",
                       size, value, required);
  if (param == CL_PROGRAM_BUILD_OPTIONS)
    return string_info(p->cl12_options ? "-cl-std=CL1.2" : "", size, value,
                       required);
  return CL_INVALID_VALUE;
}
CL_API_ENTRY cl_int CL_API_CALL clRetainProgram(cl_program p) {
  if (!program_find(p))
    return CL_INVALID_PROGRAM;
  p->refs++;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program p) {
  if (!program_find(p))
    return CL_INVALID_PROGRAM;
  if (!--p->refs) {
    cl_context c = p->context;
    UNLINK(cl_program, p, programs);
    free(p->source);
    free(p);
    clReleaseContext(c);
  }
  return CL_SUCCESS;
}
CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program p,
                                                  const char *name,
                                                  cl_int *errcode_ret) {
  cl_kernel k;
  if (!program_find(p))
    ERROR(CL_INVALID_PROGRAM);
  if (p->status != CL_BUILD_SUCCESS)
    ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
  if (!name || !udoncl_cl_kernel_matches(name))
    ERROR(CL_INVALID_KERNEL_NAME);
  k = (cl_kernel)calloc(1, sizeof(*k));
  if (!k)
    ERROR(CL_OUT_OF_HOST_MEMORY);
  udoncl_cl_gather_profile(name, &k->gather);
  if (k->gather.id &&
      (k->gather.words < 1 || k->gather.words > 16384 ||
       !k->gather.input_count || k->gather.input_count > 13 ||
       k->gather.scalar_count > 15 - k->gather.input_count ||
       !k->gather.texture_width || !k->gather.texture_height ||
       (size_t)k->gather.texture_width * k->gather.texture_height <
           k->gather.words ||
       !k->gather.global_width || !k->gather.global_height ||
       (size_t)k->gather.global_width * k->gather.global_height !=
           k->gather.words)) {
    free(k);
    ERROR(CL_INVALID_KERNEL_DEFINITION);
  }
  k->refs = 1;
  k->program = p;
  p->refs++;
  k->next = kernels;
  kernels = k;
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return k;
}
CL_API_ENTRY cl_int CL_API_CALL clRetainKernel(cl_kernel k) {
  if (!kernel_find(k))
    return CL_INVALID_KERNEL;
  k->refs++;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel k) {
  if (!kernel_find(k))
    return CL_INVALID_KERNEL;
  if (!--k->refs) {
    cl_program p = k->program;
    UNLINK(cl_kernel, k, kernels);
    free(k);
    clReleaseProgram(p);
  }
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel k, cl_uint index,
                                               size_t size, const void *value) {
  cl_uint buffer_count = k->gather.id ? k->gather.input_count + 1 : 1;
  if (!kernel_find(k))
    return CL_INVALID_KERNEL;
  if (index >= (k->gather.id ? buffer_count + k->gather.scalar_count : 2))
    return CL_INVALID_ARG_INDEX;
  if (size !=
      ((k->gather.id ? index < buffer_count : index == 0) ? sizeof(cl_mem) : 4))
    return CL_INVALID_ARG_SIZE;
  if (!value)
    return CL_INVALID_ARG_VALUE;
  if (k->gather.id ? index < buffer_count : index == 0) {
    cl_mem m;
    memcpy(&m, value, sizeof(m));
    if (m && !buffer_find(m))
      return CL_INVALID_MEM_OBJECT;
    if (k->gather.id && index < k->gather.input_count) {
      k->inputs[index] = m;
      k->input_generations[index] = m ? m->generation : 0;
    } else {
      k->buffer = m;
      k->buffer_generation = m ? m->generation : 0;
    }
  } else
    memcpy(k->scalar + (k->gather.id ? (index - buffer_count) * 4 : 0), value,
           4);
  k->args |= 1u << index;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue q, cl_kernel k, cl_uint dims, const size_t *offset,
    const size_t *global, const size_t *local, cl_uint count,
    const cl_event *list, cl_event *result) {
  cl_uint i;
  cl_int status;
  cl_event event = NULL;
  cl_mem m;
  const unsigned char *input_data[15];
  cl_uint argument_count =
      k->gather.id ? k->gather.input_count + 1 + k->gather.scalar_count : 2;
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  if (!kernel_find(k))
    return CL_INVALID_KERNEL;
  if (k->program->context != q->context)
    return CL_INVALID_CONTEXT;
  if ((status = waits(q->context, count, list)))
    return status;
  if (!dims || dims > 3 ||
      (k->gather.id && dims != (k->gather.global_height > 1 ? 2u : 1u)))
    return CL_INVALID_WORK_DIMENSION;
  if (!global)
    return CL_INVALID_GLOBAL_WORK_SIZE;
  for (i = 0; i < dims; i++) {
    size_t expected = 1;
    if (k->gather.id)
      expected = i == 0 ? k->gather.global_width : k->gather.global_height;
    if (global[i] != expected)
      return CL_INVALID_GLOBAL_WORK_SIZE;
    if (offset && offset[i])
      return CL_INVALID_GLOBAL_OFFSET;
    if (local && local[i] != 1)
      return CL_INVALID_WORK_GROUP_SIZE;
  }
  if (k->args != ((1u << argument_count) - 1u))
    return CL_INVALID_KERNEL_ARGS;
  m = buffer_find(k->buffer);
  if (!m || m->generation != k->buffer_generation)
    return CL_INVALID_MEM_OBJECT;
  if (m->context != q->context)
    return CL_INVALID_CONTEXT;
  if (m->flags & CL_MEM_READ_ONLY)
    return CL_INVALID_OPERATION;
  if (m->size != (k->gather.id ? k->gather.words * 4 : 4))
    return CL_INVALID_MEM_OBJECT;
  if (k->gather.id) {
    size_t byte;
    for (i = 0; i < k->gather.input_count; ++i) {
      cl_mem input = buffer_find(k->inputs[i]);
      if (!input || input->generation != k->input_generations[i])
        return CL_INVALID_MEM_OBJECT;
      if (input->context != q->context)
        return CL_INVALID_CONTEXT;
      if (input->generation == m->generation ||
          (input->flags & CL_MEM_WRITE_ONLY))
        return CL_INVALID_OPERATION;
      if (input->size != m->size)
        return CL_INVALID_MEM_OBJECT;
      for (byte = 0; byte < input->size; ++byte)
        if (!input->initialized[byte])
          return CL_INVALID_OPERATION;
      input_data[i] = input->data;
    }
  }
  if (result) {
    event = new_event(q, CL_COMMAND_NDRANGE_KERNEL);
    if (!event)
      return CL_OUT_OF_HOST_MEMORY;
  }
  if (k->gather.id
          ? !udoncl_cl_draw_gather(k->gather.id, input_data,
                                   k->gather.input_count, m->size, k->scalar,
                                   k->gather.scalar_count * 4, m->data)
          : !udoncl_cl_draw(m->id, m->generation, k->scalar, m->data)) {
    if (event)
      clReleaseEvent(event);
    return CL_OUT_OF_RESOURCES;
  }
  memset(m->initialized, 1, m->size);
  if (result)
    *result = event;
  return CL_SUCCESS;
}
static cl_int transfer(cl_command_queue q, cl_mem m, cl_bool blocking,
                       size_t offset, size_t bytes, void *ptr, cl_uint count,
                       const cl_event *list, cl_event *result, int write) {
  cl_int status;
  cl_event event = NULL;
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  if (!buffer_find(m))
    return CL_INVALID_MEM_OBJECT;
  if (m->context != q->context)
    return CL_INVALID_CONTEXT;
  if (!ptr || (blocking != CL_FALSE && blocking != CL_TRUE) || !bytes ||
      offset > m->size || bytes > m->size - offset)
    return CL_INVALID_VALUE;
  if ((status = waits(q->context, count, list)))
    return status;
  if (!write) {
    size_t byte;
    for (byte = offset; byte < offset + bytes; ++byte)
      if (!m->initialized[byte])
        return CL_INVALID_OPERATION;
  }
  if (result) {
    event =
        new_event(q, write ? CL_COMMAND_WRITE_BUFFER : CL_COMMAND_READ_BUFFER);
    if (!event)
      return CL_OUT_OF_HOST_MEMORY;
  }
  if (write) {
    memcpy(m->data + offset, ptr, bytes);
    memset(m->initialized + offset, 1, bytes);
  } else
    memcpy(ptr, m->data + offset, bytes);
  if (result)
    *result = event;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(
    cl_command_queue q, cl_mem m, cl_bool blocking, size_t offset, size_t bytes,
    void *ptr, cl_uint n, const cl_event *list, cl_event *result) {
  return transfer(q, m, blocking, offset, bytes, ptr, n, list, result, 0);
}
CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(
    cl_command_queue q, cl_mem m, cl_bool blocking, size_t offset, size_t bytes,
    const void *ptr, cl_uint n, const cl_event *list, cl_event *result) {
  return transfer(q, m, blocking, offset, bytes, (void *)ptr, n, list, result,
                  1);
}
CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(
    cl_command_queue q, cl_mem source, cl_mem destination, size_t source_offset,
    size_t destination_offset, size_t bytes, cl_uint count,
    const cl_event *list, cl_event *result) {
  cl_int status;
  cl_event event = NULL;
  size_t byte;
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  if (!buffer_find(source) || !buffer_find(destination))
    return CL_INVALID_MEM_OBJECT;
  if (source->context != q->context || destination->context != q->context)
    return CL_INVALID_CONTEXT;
  if (!bytes || source_offset > source->size ||
      bytes > source->size - source_offset ||
      destination_offset > destination->size ||
      bytes > destination->size - destination_offset)
    return CL_INVALID_VALUE;
  if ((source == destination ||
       (source->parent ? source->parent : source) ==
           (destination->parent ? destination->parent : destination)) &&
      source->origin + source_offset <
          destination->origin + destination_offset + bytes &&
      destination->origin + destination_offset <
          source->origin + source_offset + bytes)
    return CL_MEM_COPY_OVERLAP;
  if ((status = waits(q->context, count, list)))
    return status;
  for (byte = source_offset; byte < source_offset + bytes; ++byte)
    if (!source->initialized[byte])
      return CL_INVALID_OPERATION;
  if (result) {
    event = new_event(q, CL_COMMAND_COPY_BUFFER);
    if (!event)
      return CL_OUT_OF_HOST_MEMORY;
  }
  memmove(destination->data + destination_offset, source->data + source_offset,
          bytes);
  memset(destination->initialized + destination_offset, 1, bytes);
  if (result)
    *result = event;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferRect(
    cl_command_queue q, cl_mem source, cl_mem destination,
    const size_t *source_origin, const size_t *destination_origin,
    const size_t *region, size_t source_row_pitch, size_t source_slice_pitch,
    size_t destination_row_pitch, size_t destination_slice_pitch, cl_uint count,
    const cl_event *list, cl_event *result) {
  (void)source_origin;
  (void)destination_origin;
  (void)region;
  (void)source_row_pitch;
  (void)source_slice_pitch;
  (void)destination_row_pitch;
  (void)destination_slice_pitch;
  (void)count;
  (void)list;
  (void)result;
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  if (!buffer_find(source) || !buffer_find(destination))
    return CL_INVALID_MEM_OBJECT;
  return CL_INVALID_OPERATION;
}
CL_API_ENTRY void *CL_API_CALL clEnqueueMapBuffer(
    cl_command_queue q, cl_mem m, cl_bool blocking, cl_map_flags flags,
    size_t offset, size_t bytes, cl_uint count, const cl_event *list,
    cl_event *result, cl_int *errcode_ret) {
  (void)blocking;
  (void)flags;
  (void)offset;
  (void)bytes;
  (void)count;
  (void)list;
  (void)result;
  if (!queue_find(q))
    ERROR(CL_INVALID_COMMAND_QUEUE);
  if (!buffer_find(m))
    ERROR(CL_INVALID_MEM_OBJECT);
  ERROR(CL_INVALID_OPERATION);
}
CL_API_ENTRY cl_int CL_API_CALL clEnqueueUnmapMemObject(cl_command_queue q,
                                                        cl_mem m, void *mapped,
                                                        cl_uint count,
                                                        const cl_event *list,
                                                        cl_event *result) {
  (void)mapped;
  (void)count;
  (void)list;
  (void)result;
  if (!queue_find(q))
    return CL_INVALID_COMMAND_QUEUE;
  if (!buffer_find(m))
    return CL_INVALID_MEM_OBJECT;
  return CL_INVALID_OPERATION;
}
CL_API_ENTRY cl_int CL_API_CALL clFlush(cl_command_queue q) {
  return queue_find(q) ? CL_SUCCESS : CL_INVALID_COMMAND_QUEUE;
}
CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue q) {
  return queue_find(q) ? CL_SUCCESS : CL_INVALID_COMMAND_QUEUE;
}
CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(cl_uint count,
                                                const cl_event *list) {
  cl_uint i;
  cl_context c;
  if (!count || !list)
    return CL_INVALID_VALUE;
  if (!event_find(list[0]))
    return CL_INVALID_EVENT;
  c = list[0]->queue->context;
  for (i = 0; i < count; i++) {
    cl_event e = event_find(list[i]);
    if (!e)
      return CL_INVALID_EVENT;
    if (e->queue->context != c)
      return CL_INVALID_CONTEXT;
  }
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clRetainEvent(cl_event e) {
  if (!event_find(e))
    return CL_INVALID_EVENT;
  e->refs++;
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(cl_event e) {
  if (!event_find(e))
    return CL_INVALID_EVENT;
  if (!--e->refs) {
    cl_command_queue q = e->queue;
    UNLINK(cl_event, e, events);
    free(e);
    clReleaseCommandQueue(q);
  }
  return CL_SUCCESS;
}
CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo(cl_event e, cl_event_info param,
                                               size_t size, void *value,
                                               size_t *required) {
  cl_int status = CL_COMPLETE;
  if (!event_find(e))
    return CL_INVALID_EVENT;
  if (param == CL_EVENT_COMMAND_EXECUTION_STATUS)
    return info(&status, sizeof(status), size, value, required);
  if (param == CL_EVENT_COMMAND_QUEUE)
    return info(&e->queue, sizeof(e->queue), size, value, required);
  if (param == CL_EVENT_CONTEXT)
    return info(&e->queue->context, sizeof(cl_context), size, value, required);
  if (param == CL_EVENT_COMMAND_TYPE)
    return info(&e->type, sizeof(e->type), size, value, required);
  if (param == CL_EVENT_REFERENCE_COUNT)
    return info(&e->refs, sizeof(e->refs), size, value, required);
  return CL_INVALID_VALUE;
}

/* Internal test instrumentation; not an OpenCL extension API. */
size_t udoncl_cl_live_objects(void) {
  size_t count = 0;
  cl_context c;
  cl_command_queue q;
  cl_program p;
  cl_kernel k;
  cl_mem m;
  cl_event e;
  for (c = contexts; c; c = c->next)
    count++;
  for (q = queues; q; q = q->next)
    count++;
  for (p = programs; p; p = p->next)
    count++;
  for (k = kernels; k; k = k->next)
    count++;
  for (m = buffers; m; m = m->next)
    count++;
  for (e = events; e; e = e->next)
    count++;
  return count;
}
