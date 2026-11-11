#ifndef UDONCL_STRING_H
#define UDONCL_STRING_H

#include <stdint.h>

/*
 * An immutable host string reference supplied by Udon/UdonSharp.
 *
 * This is deliberately distinct from char *: it refers to a managed UTF-16
 * System.String and cannot be dereferenced through the C byte-memory model.
 * UdonCL recognizes the reserved LLVM address space after Host C import and
 * preserves the reference as a SystemString value. Native builds use an
 * equivalent opaque object pointer so host-only tests can pass references
 * through the same API.
 */
#define UDONCL_STRING_ADDRESS_SPACE 200

#if defined(__clang__) && defined(__SPIR__)
typedef const void __attribute__((address_space(UDONCL_STRING_ADDRESS_SPACE))) *
    udon_string_t;

int32_t udonStringLength(udon_string_t value);
int32_t udonStringCodeUnitAt(udon_string_t value, int32_t index);
#else
typedef struct udon_string_native {
  const uint16_t *data;
  int32_t length;
} udon_string_native_t;
typedef const udon_string_native_t *udon_string_t;

static inline int32_t udonStringLength(udon_string_t value) {
  return value ? value->length : 0;
}
static inline int32_t udonStringCodeUnitAt(udon_string_t value, int32_t index) {
  return value->data[index];
}
#endif

#endif
