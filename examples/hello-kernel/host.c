// Host-visible scalar export. It mirrors the Device calculation so the
// recovered Host IR can be inspected independently.
#include "hello_host.h"
#include <udoncl/export.h>
#include <udoncl/string.h>

UDONCL_EXPORT
float host_square_plus_one(float x) {
  float square = 1.0f;
  for (int factor = 0; factor < 2; ++factor)
    square *= x;
  return hello_host_identity(square) + 1.0f;
}

UDONCL_EXPORT
udon_string_t host_echo_string(udon_string_t value) { return value; }

// The current AOT boundary represents runtime-owned objects as uint32 handles;
// they are not C pointers. The request must forward them without arithmetic.
typedef unsigned int udoncl_buffer;
typedef unsigned int udoncl_dispatch_desc;
typedef unsigned int udoncl_event;

extern udoncl_event __udoncl_dispatch(udoncl_dispatch_desc desc,
                                      udoncl_buffer output);

udoncl_event request_square_plus_one(udoncl_buffer output,
                                     udoncl_dispatch_desc desc) {
  return __udoncl_dispatch(desc, output);
}
