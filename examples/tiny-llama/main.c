#include "tiny_llama.h"
#include <stdio.h>

int main(void) {
  TinyLlamaWeights weights;
  const int prompt[] = {1, 5, 9, 3};
  float logits[TINY_LLAMA_VOCAB];
  int generated[8];
  tiny_llama_init(&weights, 0x4c4c414du);
  tiny_llama_forward(&weights, prompt, 4, logits);
  printf("logits=");
  for (int token = 0; token < TINY_LLAMA_VOCAB; ++token)
    printf(token ? ",%.9g" : "%.9g", logits[token]);
  printf("\nnext=%d\n", tiny_llama_argmax(logits));
  tiny_llama_generate(&weights, prompt, 4, generated, 8);
  printf("generated=");
  for (int index = 0; index < 8; ++index)
    printf(index ? ",%d" : "%d", generated[index]);
  printf("\narchitecture=llama-rmsnorm-rope-mha-swiglu\n");
  return 0;
}
