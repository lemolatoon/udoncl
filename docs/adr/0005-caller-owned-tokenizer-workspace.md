# Keep tokenizer state caller-owned

`UdonTokenizers` exposes immutable generated Tokenizer Models and operates on
caller-owned bounded token buffers, string builders, and explicit incremental
decoder state. It has its own status domain, no mutable global tokenizer
session, no dynamic allocation, and no runtime JSON parsing, so multiple
Conversation Instances share tables while retaining deterministic Udon and
Native storage and behavior.
