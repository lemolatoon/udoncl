# Separate prefill validation and conversation artifacts

The Prefill Validation Artifact exposes the real model's complete generated
47-input/45-output ABI for oracle comparison, while the user-facing
Conversation Artifact exposes only lifecycle and conversation operations.
Keeping these artifacts separate preserves exhaustive numerical testing
without making model tensors, K/V buffers, or physical buffer identifiers part
of the normal user API; an explicit test/debug build may retain full output
observation.
