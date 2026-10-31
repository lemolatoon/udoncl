#include "tiny_llama.h"
#include <math.h>

static unsigned next_random(unsigned *state) {
  unsigned value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

static float next_weight(unsigned *state) {
  int sample = (int)((next_random(state) >> 16) & 65535u) - 32768;
  return (float)sample * (0.125f / 32768.0f);
}

static void fill(float *values, int count, unsigned *state) {
  for (int index = 0; index < count; ++index)
    values[index] = next_weight(state);
}

void tiny_llama_init(TinyLlamaWeights *weights, unsigned seed) {
  unsigned state = seed;
  fill(weights->embedding, TINY_LLAMA_VOCAB * TINY_LLAMA_DIM, &state);
  fill(weights->wq, TINY_LLAMA_DIM * TINY_LLAMA_DIM, &state);
  fill(weights->wk, TINY_LLAMA_DIM * TINY_LLAMA_DIM, &state);
  fill(weights->wv, TINY_LLAMA_DIM * TINY_LLAMA_DIM, &state);
  fill(weights->wo, TINY_LLAMA_DIM * TINY_LLAMA_DIM, &state);
  fill(weights->w1, TINY_LLAMA_HIDDEN * TINY_LLAMA_DIM, &state);
  fill(weights->w2, TINY_LLAMA_DIM * TINY_LLAMA_HIDDEN, &state);
  fill(weights->w3, TINY_LLAMA_HIDDEN * TINY_LLAMA_DIM, &state);
  fill(weights->output, TINY_LLAMA_VOCAB * TINY_LLAMA_DIM, &state);
  for (int index = 0; index < TINY_LLAMA_DIM; ++index) {
    weights->attention_norm[index] = 0.875f + next_weight(&state);
    weights->ffn_norm[index] = 0.875f + next_weight(&state);
    weights->final_norm[index] = 0.875f + next_weight(&state);
  }
}

static void rms_norm(float *output, const float *input, const float *weight) {
  float square_sum = 0.0f;
  for (int index = 0; index < TINY_LLAMA_DIM; ++index)
    square_sum += input[index] * input[index];
  float scale = 1.0f / sqrtf(square_sum / (float)TINY_LLAMA_DIM + 1.0e-5f);
  for (int index = 0; index < TINY_LLAMA_DIM; ++index)
    output[index] = input[index] * scale * weight[index];
}

static void matrix_vector(float *output, const float *matrix,
                          const float *input, int rows, int columns) {
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int column = 0; column < columns; ++column)
      sum += matrix[row * columns + column] * input[column];
    output[row] = sum;
  }
}

static void apply_rope(float *vector, int position) {
  const int head_dimension = TINY_LLAMA_DIM / TINY_LLAMA_HEADS;
  for (int head = 0; head < TINY_LLAMA_HEADS; ++head) {
    for (int pair = 0; pair < head_dimension / 2; ++pair) {
      float frequency = pair == 0 ? 1.0f : 0.01f;
      float angle = (float)position * frequency;
      float cosine = cosf(angle), sine = sinf(angle);
      int index = head * head_dimension + pair * 2;
      float first = vector[index], second = vector[index + 1];
      vector[index] = first * cosine - second * sine;
      vector[index + 1] = first * sine + second * cosine;
    }
  }
}

void tiny_llama_forward(const TinyLlamaWeights *weights, const int *tokens,
                        int token_count, float *logits) {
  float state[TINY_LLAMA_CONTEXT][TINY_LLAMA_DIM];
  float normalized[TINY_LLAMA_CONTEXT][TINY_LLAMA_DIM];
  float query[TINY_LLAMA_CONTEXT][TINY_LLAMA_DIM];
  float key[TINY_LLAMA_CONTEXT][TINY_LLAMA_DIM];
  float value[TINY_LLAMA_CONTEXT][TINY_LLAMA_DIM];
  float attention[TINY_LLAMA_CONTEXT][TINY_LLAMA_DIM];
  if (token_count < 1)
    token_count = 1;
  if (token_count > TINY_LLAMA_CONTEXT)
    token_count = TINY_LLAMA_CONTEXT;
  for (int position = 0; position < token_count; ++position) {
    int token = tokens[position] & (TINY_LLAMA_VOCAB - 1);
    for (int index = 0; index < TINY_LLAMA_DIM; ++index)
      state[position][index] =
          weights->embedding[token * TINY_LLAMA_DIM + index];
    rms_norm(normalized[position], state[position], weights->attention_norm);
    matrix_vector(query[position], weights->wq, normalized[position],
                  TINY_LLAMA_DIM, TINY_LLAMA_DIM);
    matrix_vector(key[position], weights->wk, normalized[position],
                  TINY_LLAMA_DIM, TINY_LLAMA_DIM);
    matrix_vector(value[position], weights->wv, normalized[position],
                  TINY_LLAMA_DIM, TINY_LLAMA_DIM);
    apply_rope(query[position], position);
    apply_rope(key[position], position);
  }

  const int head_dimension = TINY_LLAMA_DIM / TINY_LLAMA_HEADS;
  for (int position = 0; position < token_count; ++position) {
    for (int head = 0; head < TINY_LLAMA_HEADS; ++head) {
      float scores[TINY_LLAMA_CONTEXT];
      float maximum = -1.0e30f, denominator = 0.0f;
      for (int previous = 0; previous <= position; ++previous) {
        float score = 0.0f;
        for (int index = 0; index < head_dimension; ++index) {
          int lane = head * head_dimension + index;
          score += query[position][lane] * key[previous][lane];
        }
        score *= 0.5f;
        scores[previous] = score;
        if (score > maximum)
          maximum = score;
      }
      for (int previous = 0; previous <= position; ++previous) {
        scores[previous] = expf(scores[previous] - maximum);
        denominator += scores[previous];
      }
      for (int index = 0; index < head_dimension; ++index) {
        float sum = 0.0f;
        int lane = head * head_dimension + index;
        for (int previous = 0; previous <= position; ++previous)
          sum += scores[previous] * value[previous][lane];
        attention[position][lane] = sum / denominator;
      }
    }
  }

  for (int position = 0; position < token_count; ++position) {
    float projected[TINY_LLAMA_DIM];
    float gate[TINY_LLAMA_HIDDEN], up[TINY_LLAMA_HIDDEN];
    float activated[TINY_LLAMA_HIDDEN], down[TINY_LLAMA_DIM];
    matrix_vector(projected, weights->wo, attention[position], TINY_LLAMA_DIM,
                  TINY_LLAMA_DIM);
    for (int index = 0; index < TINY_LLAMA_DIM; ++index)
      state[position][index] += projected[index];
    rms_norm(normalized[position], state[position], weights->ffn_norm);
    matrix_vector(gate, weights->w1, normalized[position], TINY_LLAMA_HIDDEN,
                  TINY_LLAMA_DIM);
    matrix_vector(up, weights->w3, normalized[position], TINY_LLAMA_HIDDEN,
                  TINY_LLAMA_DIM);
    for (int index = 0; index < TINY_LLAMA_HIDDEN; ++index) {
      float silu = gate[index] / (1.0f + expf(-gate[index]));
      activated[index] = silu * up[index];
    }
    matrix_vector(down, weights->w2, activated, TINY_LLAMA_DIM,
                  TINY_LLAMA_HIDDEN);
    for (int index = 0; index < TINY_LLAMA_DIM; ++index)
      state[position][index] += down[index];
  }

  float final_state[TINY_LLAMA_DIM];
  rms_norm(final_state, state[token_count - 1], weights->final_norm);
  matrix_vector(logits, weights->output, final_state, TINY_LLAMA_VOCAB,
                TINY_LLAMA_DIM);
}

int tiny_llama_argmax(const float *logits) {
  int result = 0;
  for (int token = 1; token < TINY_LLAMA_VOCAB; ++token)
    if (logits[token] > logits[result])
      result = token;
  return result;
}

void tiny_llama_generate(const TinyLlamaWeights *weights, const int *prompt,
                         int prompt_count, int *output, int output_count) {
  int context[TINY_LLAMA_CONTEXT] = {0};
  int context_count =
      prompt_count < TINY_LLAMA_CONTEXT ? prompt_count : TINY_LLAMA_CONTEXT;
  int prompt_start = prompt_count - context_count;
  for (int index = 0; index < context_count; ++index)
    context[index] = prompt[prompt_start + index];
  for (int generated = 0; generated < output_count; ++generated) {
    float logits[TINY_LLAMA_VOCAB];
    tiny_llama_forward(weights, context, context_count, logits);
    int token = tiny_llama_argmax(logits);
    output[generated] = token;
    if (context_count < TINY_LLAMA_CONTEXT)
      context[context_count++] = token;
    else {
      for (int index = 1; index < TINY_LLAMA_CONTEXT; ++index)
        context[index - 1] = context[index];
      context[TINY_LLAMA_CONTEXT - 1] = token;
    }
  }
}
