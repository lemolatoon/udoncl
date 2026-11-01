# Reinforcement-learning policy

This fixture is a small continuous-control actor network compiled entirely by
`UdonOperatorExecutionProvider`. It uses standard ONNX Gemm, Add, Tanh,
Sigmoid, Mul, Sub, and Softmax nodes. The compiler lowers them to tiled
UdonBLAS SGEMM calls and cuDNN-compatible UdonDNN activation, OpTensor, and
Softmax calls without a model-specific execution provider.

The repository Device gate compiles the generated C Host itself and executes
all ten calls through UdonCL's fragment backend. Three action probabilities are
compared with an independent CPU policy evaluation.

```text
python examples/reinforcement-learning/generate_onnx.py -o policy.onnx
python runtime/onnx/udon_onnx.py policy.onnx -o policy.json \
  --execution-provider operators --emit-host-uasm policy.uasm
```
