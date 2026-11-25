/* The class of the largest of the ten MNIST logits; ties go to the lowest
 * class. One work item reads every logit. */
#define MNIST_CLASSES 10u

__kernel void mnist_argmax(__global const float *logits, __global uint *digit) {
  uint best = 0u;
  float value = 0.0f;
  for (uint index = 0u; index < MNIST_CLASSES; ++index) {
    float candidate = logits[index];
    if (index == 0u || candidate > value) {
      value = candidate;
      best = index;
    }
  }
  digit[get_global_id(0)] = best;
}
