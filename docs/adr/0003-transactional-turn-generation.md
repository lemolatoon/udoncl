# Generate and commit each turn transactionally

`AppendUserText` records one Pending User Turn, while may-suspend `Generate`
owns chat-template application, tokenization, inference, greedy selection, and
decoding. Generation preserves committed tokens, response, and K/V while two
working K/V banks ping-pong; success promotes the Working Turn atomically and
failure retains the pending text and prior state. This three-bank initial path
trades Device memory for rollback and Fragment Shader read/write legality; a
later proven append-only optimization may reduce storage without changing the
transaction boundary.
