// Reduced LJ units: mass = sigma = epsilon = kB = 1. N must be even.
// Four adjacent scalar words per particle: x, y, vx, vy. Each invocation owns
// exactly out[get_global_id(0)], so no scatter, atomic, or work-group barrier.
#ifndef MD_PARTICLES
#define MD_PARTICLES 64
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
INLINE uint hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
INLINE float noise(uint x) {
  return (float)(hash(x) & 65535u) * (2.0f / 65535.0f) - 1.0f;
}
INLINE float heater(float x, float y, float hx, float hy, float radius) {
  float dx = x - hx, dy = y - hy;
  float w = 1.0f - (dx * dx + dy * dy) / (radius * radius);
  return w > 0.0f ? w : 0.0f;
}
INLINE void md_update(__global const float *state, __global float *out,
                      float dt, float box_x, float box_y, float hand0_x,
                      float hand0_y, float hand1_x, float hand1_y, float radius,
                      float amplitude, float cooling_rate, uint step,
                      uint do_drift) {
  uint word = (uint)get_global_id(0);
  uint i = word >> 2, lane = word & 3u, base = i * 4u;
  float x = state[base], y = state[base + 1u];
  float vx = state[base + 2u], vy = state[base + 3u];
  float fx = 0.0f, fy = 0.0f;
  // Shift the radial force to zero at rc=2.5; potential is shifted consistently
  // in the independent CPU energy calculation used by the verification runner.
  const float rc = 2.5f;
  const float ic2 = 1.0f / (rc * rc), ic6 = ic2 * ic2 * ic2;
  const float fc = 24.0f * (2.0f * ic6 * ic6 - ic6) / rc;
  for (uint j = 0; j < MD_PARTICLES; ++j) {
    if (j != i) {
      float dx = image(x - state[j * 4u], box_x);
      float dy = image(y - state[j * 4u + 1u], box_y);
      float r2 = dx * dx + dy * dy;
      if (r2 < rc * rc) {
        float inv2 = 1.0f / r2, inv6 = inv2 * inv2 * inv2;
        float coefficient =
            24.0f * (2.0f * inv6 * inv6 - inv6) * inv2 - fc / sqrt(r2);
        fx += coefficient * dx;
        fy += coefficient * dy;
      }
    }
  }
  vx += 0.5f * dt * fx;
  vy += 0.5f * dt * fy;
  if (do_drift) {
    x += dt * vx;
    y += dt * vy;
    x -= box_x * floor(x / box_x);
    y -= box_y * floor(y / box_y);
  } else {
    // Paired, zero-net-momentum random impulses provide a localized external
    // heater. Both members of each pair must be inside the heat footprint.
    uint partner = (i ^ 1u) * 4u;
    float px = state[partner], py = state[partner + 1u];
    float w0 = heater(x, y, hand0_x, hand0_y, radius),
          w1 = heater(x, y, hand1_x, hand1_y, radius);
    float p0 = heater(px, py, hand0_x, hand0_y, radius),
          p1 = heater(px, py, hand1_x, hand1_y, radius);
    float w = w0 > w1 ? w0 : w1, p = p0 > p1 ? p0 : p1;
    float weight = w < p ? w : p;
    float sign = (i & 1u) ? -1.0f : 1.0f;
    uint seed = (i >> 1) * 747796405u + step * 2891336453u;
    vx += sign * amplitude * weight * noise(seed);
    vy += sign * amplitude * weight * noise(seed + 277803737u);
    // A zero-temperature Langevin bath: F_f = -m*v/tau with no stochastic
    // term. It models passive heat loss to the world while preserving the
    // localized, paired heater as the only energy source.
    float damping = 1.0f - cooling_rate * dt;
    damping = damping > 0.0f ? damping : 0.0f;
    vx *= damping;
    vy *= damping;
  }
  float value = lane == 0u ? x : lane == 1u ? y : lane == 2u ? vx : vy;
  out[word] = value;
}

// Without build-time selection this is one valid OpenCL program containing
// both entry points. AOT compilation selects each independently.
#ifndef MD_KICK
__kernel void md_drift(__global const float *state, __global float *out,
                       float dt, float box_x, float box_y, float hand0_x,
                       float hand0_y, float hand1_x, float hand1_y,
                       float radius, float amplitude, float cooling_rate,
                       uint step) {
  md_update(state, out, dt, box_x, box_y, hand0_x, hand0_y, hand1_x, hand1_y,
            radius, amplitude, cooling_rate, step, 1u);
}
#endif
#ifndef MD_DRIFT
__kernel void md_kick(__global const float *state, __global float *out,
                      float dt, float box_x, float box_y, float hand0_x,
                      float hand0_y, float hand1_x, float hand1_y, float radius,
                      float amplitude, float cooling_rate, uint step) {
  md_update(state, out, dt, box_x, box_y, hand0_x, hand0_y, hand1_x, hand1_y,
            radius, amplitude, cooling_rate, step, 0u);
}
#endif
