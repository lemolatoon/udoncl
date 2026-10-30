// Reduced LJ units: mass = sigma = epsilon = kB = 1.
// Eight aligned words per particle: x, y, z, reserved, vx, vy, vz, reserved.
#ifndef MD_PARTICLES
#define MD_PARTICLES 32
#endif
#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t md_cpu_gid;
static inline size_t get_global_id(unsigned dim) {
  (void)dim;
  return md_cpu_gid;
}
#define __kernel
#define __global
#define floor floorf
#define sqrt sqrtf
#endif
#define INLINE static inline __attribute__((always_inline))
INLINE float image(float x, float box) {
  return x - box * floor(x / box + 0.5f);
}
INLINE float noise(float x) {
  // A float-only deterministic mixer keeps the Unity Material ABI uniform;
  // this is an external heater, not a cryptographic random source.
  x = x * 12.9898f + 78.233f;
  x = x * x * 0.000137f + x * 0.6180339f;
  return 2.0f * (x - floor(x)) - 1.0f;
}
INLINE float heater(float x, float y, float z, float hx, float hy, float hz,
                    float radius) {
  float dx = x - hx, dy = y - hy, dz = z - hz;
  float w = 1.0f - (dx * dx + dy * dy + dz * dz) / (radius * radius);
  return w > 0.0f ? w : 0.0f;
}
INLINE void md_update(__global const float *state, __global float *out,
                      float dt, float box_x, float box_y, float box_z,
                      float hand0_x, float hand0_y, float hand0_z,
                      float hand1_x, float hand1_y, float hand1_z, float radius,
                      float amplitude, float cooling_rate, float step_value,
                      uint do_drift) {
  uint word = (uint)get_global_id(0), i = word >> 3, lane = word & 7u,
       base = i * 8u;
  float x = state[base], y = state[base + 1u], z = state[base + 2u];
  float vx = state[base + 4u], vy = state[base + 5u], vz = state[base + 6u];
  float fx = 0.0f, fy = 0.0f, fz = 0.0f, speed2_sum = 0.0f;
  // The 2x2x2 periodic FCC cell has L/2 = 1.587; keeping rc below that
  // avoids the ambiguous minimum-image plane while retaining 12 neighbours.
  const float rc = 1.5f, ic2 = 1.0f / (rc * rc), ic6 = ic2 * ic2 * ic2;
  const float fc = 24.0f * (2.0f * ic6 * ic6 - ic6) / rc;
  for (uint j = 0; j < MD_PARTICLES; ++j) {
    uint other = j * 8u;
    float ovx = state[other + 4u], ovy = state[other + 5u],
          ovz = state[other + 6u];
    speed2_sum += ovx * ovx + ovy * ovy + ovz * ovz;
    if (j != i) {
      float dx = image(x - state[other], box_x),
            dy = image(y - state[other + 1u], box_y);
      float dz = image(z - state[other + 2u], box_z),
            r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < rc * rc) {
        float inv2 = 1.0f / r2, inv6 = inv2 * inv2 * inv2;
        float c = 24.0f * (2.0f * inv6 * inv6 - inv6) * inv2 - fc / sqrt(r2);
        fx += c * dx;
        fy += c * dy;
        fz += c * dz;
      }
    }
  }
  float temperature = speed2_sum / (3.0f * (float)MD_PARTICLES - 3.0f);
  float liquid = (temperature - 0.12f) / 0.23f;
  liquid = liquid < 0.0f ? 0.0f : liquid > 1.0f ? 1.0f : liquid;
  // A downward driving field turns on continuously with the liquid fraction.
  // It is absent in the crystalline NVE regime.
  const float gravity = 0.8f;
  fy -= gravity * liquid;
  vx += 0.5f * dt * fx;
  vy += 0.5f * dt * fy;
  vz += 0.5f * dt * fz;
  if (do_drift) {
    x += dt * vx;
    y += dt * vy;
    z += dt * vz;
    x -= box_x * floor(x / box_x);
    z -= box_z * floor(z / box_z);
    if (liquid > 0.0f) {
      // A slightly inelastic floor/ceiling makes the cohesive LJ liquid pool
      // and run along the container instead of wrapping vertically.
      const float wall = 0.12f, restitution = 0.35f;
      if (y < wall) {
        y = 2.0f * wall - y;
        vy = -vy * restitution;
        vx *= 0.98f;
        vz *= 0.98f;
      } else if (y > box_y - wall) {
        y = 2.0f * (box_y - wall) - y;
        vy = -vy * restitution;
        vx *= 0.98f;
        vz *= 0.98f;
      }
    } else {
      y -= box_y * floor(y / box_y);
    }
  } else {
    uint partner = (i ^ 1u) * 8u;
    float w0 = heater(x, y, z, hand0_x, hand0_y, hand0_z, radius),
          w1 = heater(x, y, z, hand1_x, hand1_y, hand1_z, radius);
    float p0 = heater(state[partner], state[partner + 1u], state[partner + 2u],
                      hand0_x, hand0_y, hand0_z, radius);
    float p1 = heater(state[partner], state[partner + 1u], state[partner + 2u],
                      hand1_x, hand1_y, hand1_z, radius);
    float w = w0 > w1 ? w0 : w1, p = p0 > p1 ? p0 : p1, weight = w < p ? w : p;
    float sign = (i & 1u) ? -1.0f : 1.0f;
    float seed = (x + state[partner]) * 12.3f +
                 (y + state[partner + 1u]) * 45.6f +
                 (z + state[partner + 2u]) * 78.9f + step_value * 0.6180339f;
    vx += sign * amplitude * weight * noise(seed);
    vy += sign * amplitude * weight * noise(seed + 17.17f);
    vz += sign * amplitude * weight * noise(seed + 43.43f);
    // Contact heating and passive cooling are mutually exclusive thermal
    // modes.  The Host decays amplitude after release; once it falls below
    // the dead-band, the zero-temperature bath takes over and recrystallizes.
    float effective_cooling = amplitude > 0.001f ? 0.0f : cooling_rate;
    float damping = 1.0f - effective_cooling * dt;
    damping = damping > 0.0f ? damping : 0.0f;
    vx *= damping;
    vy *= damping;
    vz *= damping;
  }
  out[word] = lane == 0u   ? x
              : lane == 1u ? y
              : lane == 2u ? z
              : lane == 4u ? vx
              : lane == 5u ? vy
              : lane == 6u ? vz
                           : state[word];
}
#ifndef MD_KICK
__kernel void md_drift(__global const float *state, __global float *out,
                       float dt, float bx, float by, float bz, float h0x,
                       float h0y, float h0z, float h1x, float h1y, float h1z,
                       float radius, float amplitude, float cooling,
                       float step) {
  md_update(state, out, dt, bx, by, bz, h0x, h0y, h0z, h1x, h1y, h1z, radius,
            amplitude, cooling, step, 1u);
}
#endif
#ifndef MD_DRIFT
__kernel void md_kick(__global const float *state, __global float *out,
                      float dt, float bx, float by, float bz, float h0x,
                      float h0y, float h0z, float h1x, float h1y, float h1z,
                      float radius, float amplitude, float cooling,
                      float step) {
  md_update(state, out, dt, bx, by, bz, h0x, h0y, h0z, h1x, h1y, h1z, radius,
            amplitude, cooling, step, 0u);
}
#endif
