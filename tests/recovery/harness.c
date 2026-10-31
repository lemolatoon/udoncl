// Native host differential test only. This does not validate the Udon ABI.
extern int sum_even(int n);
extern unsigned scalar_ops(unsigned a, unsigned b);
extern float float_ops(float a, float b);
extern int checked_inc(int a);
extern unsigned unsigned_divrem(unsigned a, unsigned b);
extern int stack_add(int a);
int main(void) {
  for (int n = -5; n <= 129; ++n) {
    int expected = 0;
    for (int i = 0; i < n; ++i)
      if ((i & 1) == 0)
        expected += i;
    if (sum_even(n) != expected)
      return 1;
  }
  unsigned seed = 0x12483655u;
  for (int i = 0; i < 256; ++i) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    unsigned a = seed;
    unsigned b = seed * 1664525u + 1013904223u;
    unsigned sum = a + b;
    unsigned expected = a < b ? (((sum ^ a) >> 3) & 65535u) : sum;
    if (scalar_ops(a, b) != expected)
      return 2;
  }
  for (int a = -8; a <= 8; ++a)
    for (int b = -8; b <= 8; ++b)
      if (float_ops((float)a * 0.5f, (float)b * 0.5f) !=
          -((float)a * (float)b * 0.25f + 0.5f))
        return 3;
  for (int i = 0; i < 100; ++i)
    if (checked_inc(i) != i + 1)
      return 4;
  for (unsigned a = 0; a < 100; ++a)
    for (unsigned b = 1; b < 17; ++b)
      if (unsigned_divrem(a, b) != a / b + a % b)
        return 6;
  for (int i = -100; i <= 100; ++i)
    if (stack_add(i) != i + 3)
      return 5;
  return 0;
}
