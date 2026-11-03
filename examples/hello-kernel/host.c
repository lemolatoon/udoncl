// Host-visible scalar export. It mirrors the Device calculation so the
// recovered Host IR can be inspected independently.
float host_square_plus_one(float x) {
  float square = 1.0f;
  for (int factor = 0; factor < 2; ++factor)
    square *= x;
  return square + 1.0f;
}

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
