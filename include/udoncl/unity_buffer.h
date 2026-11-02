#ifndef UDONCL_UNITY_BUFFER_H
#define UDONCL_UNITY_BUFFER_H

#include <stdint.h>

typedef uint32_t udoncl_buffer;

uint32_t __udoncl_buffer_alloc(uint32_t byte_size);
int32_t __udoncl_buffer_copy(uint32_t source, uint32_t destination,
                             uint32_t source_offset,
                             uint32_t destination_offset, uint32_t byte_size);

#define udonclBufferAlloc(byte_size) __udoncl_buffer_alloc((byte_size))
#define udonclCopyBuffer(source, destination, source_offset,                   \
                         destination_offset, byte_size)                        \
  __udoncl_buffer_copy((source), (destination), (source_offset),               \
                       (destination_offset), (byte_size))

#endif
