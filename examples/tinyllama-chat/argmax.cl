/* Greedy decoding: the index of the largest of the 32000 logits of the last
 * run, ties to the lowest index. One work item reads every logit. */
#define VOCABULARY 32000u

__kernel void tinyllama_argmax(__global const float *logits,
                               __global uint *token) {
  uint best = 0u;
  float value = 0.0f;
  for (uint index = 0u; index < VOCABULARY; ++index) {
    float candidate = logits[index];
    if (index == 0u || candidate > value) {
      value = candidate;
      best = index;
    }
  }
  token[get_global_id(0)] = best;
}
