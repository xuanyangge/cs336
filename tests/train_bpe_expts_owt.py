# TODO I need to implement this.


# (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary
# size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection.
# What is the longest token in the vocabulary? Does it make sense?
# Resource requirements: ≤ 12 hours (no GPUs), ≤ 100 GB RAM
# Deliverable: A one-to-two sentence response.
# (b) Compare and contrast the tokenizer that you get training on TinyStories versus
# OpenWebText.
# Deliverable: A one-to-two sentence response.

from .bpe import run_train_bpe
from .common import DATA_PATH
import os
import json

input_path = DATA_PATH / "owt_train.txt"

if __name__ == "__main__":
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        num_chunks=1000,
    )

    os.makedirs("outputs", exist_ok=True)

    longest = max(vocab.items(), key=lambda x: len(x[1]))
    print(longest)
    with open("outputs/openweb_vocab.json", "w") as f:
        json.dump({str(k): list(v) for k, v in vocab.items()}, f)

    # merges: list[tuple[bytes, bytes]] → store as list[list[list[int]]]
    with open("outputs/openweb_merges.json", "w") as f:
        json.dump([[list(a), list(b)] for a, b in merges], f)
