from collections.abc import Iterable
import time
from collections import Counter, defaultdict
from functools import partial, cmp_to_key
from multiprocessing import Pool
from typing import IO, Any, BinaryIO, Iterator
import regex as re
import os
from collections import deque
from .common import split_by_pat, split_by_special_tokens


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.vocab_look_up = {b: i for (i, b) in self.vocab.items()}
        self.merges_look_up = {tup: i for (i, tup) in enumerate(merges)}
        self.special_tokens_map = {}
        self.special_tokens = None
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, reverse=True)
            special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]
            self.special_tokens_map = {b: self.vocab_look_up[b] for b in special_tokens_bytes}

    # Construct a tokenizer from a given
    # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
    # the following parameters:
    # vocab: dict[int, bytes]
    # merges: list[tuple[bytes, bytes]]
    # special_tokens: list[str] | None = None

    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    # # Class method
    # # that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
    # # same format that your BPE training code output) and (optionally) a list of special tokens.
    # # This method should accept the following additional parameters:
    # # vocab_filepath: str
    # # merges_filepath: str
    # # special_tokens: list[str] | None = None
    #     return Tokenizer(
    #         ...
    #     )

    def encode_word(self, word: str):
        word_bytes = word.encode("utf-8")
        cur_tokens = [self.vocab_look_up[bytes([b])] for b in word_bytes]
        # A larger token must have a subtoken with lower rank before it.
        # Thus we can do the greedy algorithm here.
        while len(cur_tokens) > 1:
            min_rank = len(self.merges_look_up)
            next_tuple_bytes = None
            ind = -1
            for i in range(len(cur_tokens) - 1):
                pair = (self.vocab[cur_tokens[i]], self.vocab[cur_tokens[i + 1]])
                if pair in self.merges_look_up and self.merges_look_up[pair] < min_rank:
                    min_rank = self.merges_look_up[pair]
                    next_tuple_bytes = pair
                    ind = i

            if next_tuple_bytes is None:
                break

            cur_tokens = (
                cur_tokens[:ind]
                + [self.vocab_look_up[next_tuple_bytes[0] + next_tuple_bytes[1]]]
                + cur_tokens[ind + 2 :]
            )

        return cur_tokens

    def encoding_generator(self, non_speical_text: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for word_match in re.finditer(PAT, non_speical_text):
            yield self.encode_word(word_match.group())

    def encode(self, text: str) -> list[int]:
        res = []
        if self.special_tokens is None:
            for word in self.encoding_generator(text):
                res.extend(word)
            return res

        #  Encode an input text into a sequence of token IDs.
        re_pattern = "|".join([re.escape(special_token) for special_token in self.special_tokens])
        last = 0
        for m in re.finditer(re_pattern, text):
            start = m.start()
            group_s = m.group()
            group_b = group_s.encode("utf-8")
            for word in self.encoding_generator(text[last:start]):
                res.extend(word)
            res.append(self.special_tokens_map[group_b])
            last = m.end()

        if last < len(text):
            for word in self.encoding_generator(text[last:]):
                res.extend(word)

        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")

    # Decode a sequence of token IDs into text.
    # To test your Tokenizer against our provided tests, you will first need to implement the test
    # adapter at [adapters.get_tokenizer] . Then, run uv run pytest tests/test_tokenizer.py. Your
    # implementation should be able to pass all tests.


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)
