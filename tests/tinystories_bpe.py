from .bpe import run_train_bpe
from .common import FIXTURES_PATH
import tracemalloc
import json
import os

input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"

if __name__ == "__main__":
    tracemalloc.start()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/tiny_stories_vocab.json", "w") as f:
        json.dump({str(k): list(v) for k, v in vocab.items()}, f)

    # merges: list[tuple[bytes, bytes]] → store as list[list[list[int]]]
    with open("outputs/tiny_stories_merges.json", "w") as f:
        json.dump([[list(a), list(b)] for a, b in merges], f)
