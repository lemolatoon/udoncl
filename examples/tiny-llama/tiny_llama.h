#pragma once

#define TINY_LLAMA_VOCAB 16
#define TINY_LLAMA_DIM 8
#define TINY_LLAMA_HIDDEN 16
#define TINY_LLAMA_HEADS 2
#define TINY_LLAMA_CONTEXT 4

typedef struct TinyLlamaWeights {
  float embedding[TINY_LLAMA_VOCAB * TINY_LLAMA_DIM];
  float attention_norm[TINY_LLAMA_DIM];
  float wq[TINY_LLAMA_DIM * TINY_LLAMA_DIM];
  float wk[TINY_LLAMA_DIM * TINY_LLAMA_DIM];
  float wv[TINY_LLAMA_DIM * TINY_LLAMA_DIM];
  float wo[TINY_LLAMA_DIM * TINY_LLAMA_DIM];
  float ffn_norm[TINY_LLAMA_DIM];
  float w1[TINY_LLAMA_HIDDEN * TINY_LLAMA_DIM];
  float w2[TINY_LLAMA_DIM * TINY_LLAMA_HIDDEN];
  float w3[TINY_LLAMA_HIDDEN * TINY_LLAMA_DIM];
  float final_norm[TINY_LLAMA_DIM];
  float output[TINY_LLAMA_VOCAB * TINY_LLAMA_DIM];
} TinyLlamaWeights;

#ifdef __cplusplus
extern "C" {
#endif
void tiny_llama_init(TinyLlamaWeights *weights, unsigned seed);
void tiny_llama_forward(const TinyLlamaWeights *weights, const int *tokens,
                        int token_count, float *logits);
int tiny_llama_argmax(const float *logits);
void tiny_llama_generate(const TinyLlamaWeights *weights, const int *prompt,
                         int prompt_count, int *output, int output_count);
#ifdef __cplusplus
}
#endif
