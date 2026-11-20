// RUN: %vrc-opt --verify-diagnostics --vrc-evaluate-host-constructors=step-budget=1000 %s

// Evaluation is bounded: a constructor that does not terminate is an error,
// not a compiler hang.

// expected-note @below {{while evaluating @spins}}
func.func @spins() attributes {vrc.static_constructor} {
  cf.br ^loop
^loop:
  // expected-error @below {{cannot be evaluated at build time: exceeds the evaluation step budget}}
  cf.br ^loop
}
