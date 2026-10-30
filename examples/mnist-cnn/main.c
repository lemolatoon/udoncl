#include "mnist_cnn.h"
#include "samples.h"

#include <stdio.h>

int main(void) {
  float input[MNIST_CNN_INPUT_WORDS];
  int correct = 0;
  for (int expected = 0; expected < MNIST_CNN_CLASSES; ++expected) {
    for (int index = 0; index < MNIST_CNN_INPUT_WORDS; ++index)
      input[index] = (float)kMnistSamples[expected][index] * (1.0f / 255.0f);
    int prediction = mnist_cnn_predict(input);
    printf("sample=%d prediction=%d\n", expected, prediction);
    if (prediction == expected)
      ++correct;
  }
  printf("mnist_fixture_accuracy=%d/10 "
         "architecture=conv5x5x16-relu-avgpool8x8-dense10\n",
         correct);
  return correct == MNIST_CNN_CLASSES ? 0 : 1;
}
