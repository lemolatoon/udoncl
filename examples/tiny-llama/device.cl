#ifndef __OPENCL_C_VERSION__
#include <math.h>
#include <stddef.h>
typedef unsigned int uint;
extern size_t tiny_llama_cpu_gid;
static inline size_t get_global_id(unsigned dimension) {
  (void)dimension;
  return tiny_llama_cpu_gid;
}
#define __kernel
#define __global
#define cos cosf
#define exp expf
#define sin sinf
#define sqrt sqrtf
#endif

#define TINY_LLAMA_WORDS 2048u
#define TINY_LLAMA_ONE_HOT_OFFSET 0u
#define TINY_LLAMA_STATE_OFFSET 64u
#define TINY_LLAMA_QUERY_OFFSET 96u
#define TINY_LLAMA_KEY_OFFSET 128u
#define TINY_LLAMA_VALUE_OFFSET 160u
#define TINY_LLAMA_ATTENTION_OFFSET 192u
#define TINY_LLAMA_POST_ATTENTION_OFFSET 224u
#define TINY_LLAMA_ACTIVATED_OFFSET 256u
#define TINY_LLAMA_FINAL_STATE_OFFSET 320u
#define TINY_LLAMA_LOGITS_OFFSET 352u

#define TINY_LLAMA_EMBEDDING_OFFSET 512u
#define TINY_LLAMA_ATTENTION_NORM_OFFSET 640u
#define TINY_LLAMA_WQ_OFFSET 648u
#define TINY_LLAMA_WK_OFFSET 712u
#define TINY_LLAMA_WV_OFFSET 776u
#define TINY_LLAMA_WO_OFFSET 840u
#define TINY_LLAMA_FFN_NORM_OFFSET 904u
#define TINY_LLAMA_W1_OFFSET 912u
#define TINY_LLAMA_W2_OFFSET 1040u
#define TINY_LLAMA_W3_OFFSET 1168u
#define TINY_LLAMA_FINAL_NORM_OFFSET 1296u
#define TINY_LLAMA_OUTPUT_OFFSET 1304u

#if !defined(TINY_LLAMA_EMBED) && !defined(TINY_LLAMA_QKV) &&                  \
    !defined(TINY_LLAMA_ATTENTION) && !defined(TINY_LLAMA_PROJECT) &&          \
    !defined(TINY_LLAMA_FFN_UP) && !defined(TINY_LLAMA_FFN_DOWN) &&            \
    !defined(TINY_LLAMA_LOGITS)
#define TINY_LLAMA_EMBED 1
#define TINY_LLAMA_QKV 1
#define TINY_LLAMA_ATTENTION 1
#define TINY_LLAMA_PROJECT 1
#define TINY_LLAMA_FFN_UP 1
#define TINY_LLAMA_FFN_DOWN 1
#define TINY_LLAMA_LOGITS 1
#endif

// One fragment owns one (position, channel) embedding output. One-hot token
// inputs keep token selection entirely on Device without float-to-int casts.
#if TINY_LLAMA_EMBED
__kernel void tiny_llama_embed(__global const float *input,
                               __global float *output) {
  uint gid = (uint)get_global_id(0);
  float result = input[gid];
  if (gid >= TINY_LLAMA_STATE_OFFSET && gid < TINY_LLAMA_QUERY_OFFSET) {
    uint position = (gid / 8u) % 4u;
    uint channel = gid % 8u;
    result = 0.0f;
    for (uint token = 0; token < 16u; ++token)
      result += input[TINY_LLAMA_ONE_HOT_OFFSET + position * 16u + token] *
                input[TINY_LLAMA_EMBEDDING_OFFSET + token * 8u + channel];
  }
  output[gid] = result;
}
#endif

// Ninety-six fragments independently compute Q, K, and V channels. Q and K
// apply RoPE after the matrix-vector product.
#if TINY_LLAMA_QKV
__kernel void tiny_llama_qkv(__global const float *input,
                             __global float *output) {
  uint gid = (uint)get_global_id(0);
  float result = input[gid];
  if (gid >= TINY_LLAMA_QUERY_OFFSET && gid < TINY_LLAMA_ATTENTION_OFFSET) {
    uint projection = (gid / 32u) % 3u;
    uint position = (gid / 8u) % 4u;
    uint channel = gid % 8u;
    float square_sum = 0.0f;
    for (uint index = 0; index < 8u; ++index) {
      float state = input[TINY_LLAMA_STATE_OFFSET + position * 8u + index];
      square_sum += state * state;
    }
    float scale = 1.0f / sqrt(square_sum * 0.125f + 1.0e-5f);
    result = 0.0f;
    if (projection == 0u)
      for (uint index = 0; index < 8u; ++index) {
        float normalized =
            input[TINY_LLAMA_STATE_OFFSET + position * 8u + index] * scale *
            input[TINY_LLAMA_ATTENTION_NORM_OFFSET + index];
        result +=
            input[TINY_LLAMA_WQ_OFFSET + channel * 8u + index] * normalized;
      }
    else if (projection == 1u)
      for (uint index = 0; index < 8u; ++index) {
        float normalized =
            input[TINY_LLAMA_STATE_OFFSET + position * 8u + index] * scale *
            input[TINY_LLAMA_ATTENTION_NORM_OFFSET + index];
        result +=
            input[TINY_LLAMA_WK_OFFSET + channel * 8u + index] * normalized;
      }
    else
      for (uint index = 0; index < 8u; ++index) {
        float normalized =
            input[TINY_LLAMA_STATE_OFFSET + position * 8u + index] * scale *
            input[TINY_LLAMA_ATTENTION_NORM_OFFSET + index];
        result +=
            input[TINY_LLAMA_WV_OFFSET + channel * 8u + index] * normalized;
      }
    if (projection < 2u) {
      uint pair_channel = channel ^ 1u;
      float paired = 0.0f;
      for (uint index = 0; index < 8u; ++index) {
        float normalized =
            input[TINY_LLAMA_STATE_OFFSET + position * 8u + index] * scale *
            input[TINY_LLAMA_ATTENTION_NORM_OFFSET + index];
        if (projection == 0u)
          paired += input[TINY_LLAMA_WQ_OFFSET + pair_channel * 8u + index] *
                    normalized;
        else
          paired += input[TINY_LLAMA_WK_OFFSET + pair_channel * 8u + index] *
                    normalized;
      }
      float frequency = ((channel % 4u) / 2u) == 0u ? 1.0f : 0.01f;
      float angle = (float)position * frequency;
      float cosine = cos(angle);
      float sine = sin(angle);
      result = (channel % 2u) == 0u ? result * cosine - paired * sine
                                    : paired * sine + result * cosine;
    }
  }
  output[gid] = result;
}
#endif

// Thirty-two fragments compute causal two-head attention outputs. Scores are
// recomputed for each reduction so no thread-local array is required.
#if TINY_LLAMA_ATTENTION
__kernel void tiny_llama_attention(__global const float *input,
                                   __global float *output) {
  uint gid = (uint)get_global_id(0);
  float result = input[gid];
  if (gid >= TINY_LLAMA_ATTENTION_OFFSET &&
      gid < TINY_LLAMA_POST_ATTENTION_OFFSET) {
    uint position = (gid / 8u) % 4u;
    uint channel = gid % 8u;
    uint head_start = (channel / 4u) * 4u;
    float maximum = -1.0e30f;
    for (uint previous = 0; previous < 4u; ++previous) {
      if (previous <= position) {
        float score = 0.0f;
        for (uint index = 0; index < 4u; ++index) {
          uint lane = head_start + index;
          score += input[TINY_LLAMA_QUERY_OFFSET + position * 8u + lane] *
                   input[TINY_LLAMA_KEY_OFFSET + previous * 8u + lane];
        }
        score *= 0.5f;
        maximum = score > maximum ? score : maximum;
      }
    }
    float denominator = 0.0f;
    result = 0.0f;
    for (uint previous = 0; previous < 4u; ++previous) {
      if (previous <= position) {
        float score = 0.0f;
        for (uint index = 0; index < 4u; ++index) {
          uint lane = head_start + index;
          score += input[TINY_LLAMA_QUERY_OFFSET + position * 8u + lane] *
                   input[TINY_LLAMA_KEY_OFFSET + previous * 8u + lane];
        }
        float probability = exp(score * 0.5f - maximum);
        denominator += probability;
        result += probability *
                  input[TINY_LLAMA_VALUE_OFFSET + previous * 8u + channel];
      }
    }
    result /= denominator;
  }
  output[gid] = result;
}
#endif

// Thirty-two fragments project attention and add the first residual.
#if TINY_LLAMA_PROJECT
__kernel void tiny_llama_project(__global const float *input,
                                 __global float *output) {
  uint gid = (uint)get_global_id(0);
  float result = input[gid];
  if (gid >= TINY_LLAMA_POST_ATTENTION_OFFSET &&
      gid < TINY_LLAMA_ACTIVATED_OFFSET) {
    uint position = (gid / 8u) % 4u;
    uint channel = gid % 8u;
    result = input[TINY_LLAMA_STATE_OFFSET + position * 8u + channel];
    for (uint index = 0; index < 8u; ++index)
      result += input[TINY_LLAMA_WO_OFFSET + channel * 8u + index] *
                input[TINY_LLAMA_ATTENTION_OFFSET + position * 8u + index];
  }
  output[gid] = result;
}
#endif

// Sixty-four fragments compute the two FFN projections and fused SwiGLU.
#if TINY_LLAMA_FFN_UP
__kernel void tiny_llama_ffn_up(__global const float *input,
                                __global float *output) {
  uint gid = (uint)get_global_id(0);
  float result = input[gid];
  if (gid >= TINY_LLAMA_ACTIVATED_OFFSET &&
      gid < TINY_LLAMA_FINAL_STATE_OFFSET) {
    uint position = (gid / 16u) % 4u;
    uint channel = gid % 16u;
    float square_sum = 0.0f;
    for (uint index = 0; index < 8u; ++index) {
      float state =
          input[TINY_LLAMA_POST_ATTENTION_OFFSET + position * 8u + index];
      square_sum += state * state;
    }
    float scale = 1.0f / sqrt(square_sum * 0.125f + 1.0e-5f);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint index = 0; index < 8u; ++index) {
      float normalized =
          input[TINY_LLAMA_POST_ATTENTION_OFFSET + position * 8u + index] *
          scale * input[TINY_LLAMA_FFN_NORM_OFFSET + index];
      gate += input[TINY_LLAMA_W1_OFFSET + channel * 8u + index] * normalized;
      up += input[TINY_LLAMA_W3_OFFSET + channel * 8u + index] * normalized;
    }
    result = (gate / (1.0f + exp(-gate))) * up;
  }
  output[gid] = result;
}
#endif

// Thirty-two fragments apply the FFN down projection and second residual.
#if TINY_LLAMA_FFN_DOWN
__kernel void tiny_llama_ffn_down(__global const float *input,
                                  __global float *output) {
  uint gid = (uint)get_global_id(0);
  float result = input[gid];
  if (gid >= TINY_LLAMA_FINAL_STATE_OFFSET && gid < TINY_LLAMA_LOGITS_OFFSET) {
    uint position = (gid / 8u) % 4u;
    uint channel = gid % 8u;
    result = input[TINY_LLAMA_POST_ATTENTION_OFFSET + position * 8u + channel];
    for (uint index = 0; index < 16u; ++index)
      result += input[TINY_LLAMA_W2_OFFSET + channel * 16u + index] *
                input[TINY_LLAMA_ACTIVATED_OFFSET + position * 16u + index];
  }
  output[gid] = result;
}
#endif

// Sixteen fragments independently compute vocabulary logits for position 3.
#if TINY_LLAMA_LOGITS
__kernel void tiny_llama_logits(__global const float *input,
                                __global float *output) {
  uint gid = (uint)get_global_id(0);
  float result = input[gid];
  if (gid >= TINY_LLAMA_LOGITS_OFFSET && gid < TINY_LLAMA_LOGITS_OFFSET + 16u) {
    uint token = gid % 16u;
    float square_sum = 0.0f;
    for (uint index = 0; index < 8u; ++index) {
      float state = input[TINY_LLAMA_FINAL_STATE_OFFSET + 24u + index];
      square_sum += state * state;
    }
    float scale = 1.0f / sqrt(square_sum * 0.125f + 1.0e-5f);
    result = 0.0f;
    for (uint index = 0; index < 8u; ++index) {
      float normalized = input[TINY_LLAMA_FINAL_STATE_OFFSET + 24u + index] *
                         scale * input[TINY_LLAMA_FINAL_NORM_OFFSET + index];
      result +=
          input[TINY_LLAMA_OUTPUT_OFFSET + token * 8u + index] * normalized;
    }
  }
  output[gid] = result;
}
#endif
