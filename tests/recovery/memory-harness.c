// Native host oracle; neither a Udon runtime nor final host-ABI test.
extern int array_sum(int seed, int delta);
extern int array_alias(int value, int selector);
extern float array_matrix(float value);
int main(void) {
  for (int seed = -20; seed <= 20; ++seed)
    for (int delta = -5; delta <= 5; ++delta) {
      int expected = 0;
      for (int i = 0; i < 8; ++i) expected += seed + i + delta;
      if (array_sum(seed, delta) != expected) return 1;
    }
  for (int value = -12; value <= 12; ++value)
    for (int selector = -16; selector <= 16; ++selector)
      if (array_alias(value, selector) != value + (selector & 3) + 20) return 2;
  for (int i = -16; i <= 16; ++i) {
    float value = (float)i * 0.25f;
    if (array_matrix(value) != value) return 3;
  }
  return 0;
}
