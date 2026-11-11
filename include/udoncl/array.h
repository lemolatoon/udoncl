#ifndef UDONCL_ARRAY_H
#define UDONCL_ARRAY_H

#include <stdint.h>

/*
 * A caller-owned Udon/UdonSharp int[] reference. This is not C byte memory:
 * Host C must use the accessors below, and the reference remains owned by the
 * caller for the duration of the export call.
 */
#define UDONCL_INT32_ARRAY_ADDRESS_SPACE 201

#if defined(__clang__) && defined(__SPIR__)
typedef void __attribute__((address_space(UDONCL_INT32_ARRAY_ADDRESS_SPACE))) *
    udon_int32_array_t;

int32_t udonInt32ArrayLength(udon_int32_array_t value);
int32_t udonInt32ArrayGet(udon_int32_array_t value, int32_t index);
void udonInt32ArraySet(udon_int32_array_t value, int32_t index,
                       int32_t element);
#else
typedef struct udon_int32_array_native {
  int32_t *data;
  int32_t length;
} udon_int32_array_native_t;
typedef udon_int32_array_native_t *udon_int32_array_t;

static inline int32_t udonInt32ArrayLength(udon_int32_array_t value) {
  return value ? value->length : 0;
}
static inline int32_t udonInt32ArrayGet(udon_int32_array_t value,
                                        int32_t index) {
  return value->data[index];
}
static inline void udonInt32ArraySet(udon_int32_array_t value, int32_t index,
                                     int32_t element) {
  value->data[index] = element;
}
#endif

#endif
