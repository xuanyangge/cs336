from collections.abc import Iterable
import time
from collections import Counter, defaultdict
from functools import partial, cmp_to_key
from multiprocessing import Pool
from typing import IO, Any, BinaryIO, Iterator
import regex as re
import os
from heapdict import heapdict
from collections import deque
from .common import split_by_pat, split_by_special_tokens

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_look_up = {b:i for (i, b) in self.vocab }  
        self.merges_look_up = {tup: i for (i, tup) in merges }
        special_tokens_bytes = [s.encode('utf-8') for s in special_tokens]
        self.special_tokens_map = {b: self.vocab_look_up[b] for b in special_tokens_bytes }

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
        cur_tokens = [self.vocab[b] for b in word_bytes]
        # A larger token must have a subtoken with lower rank before it.
        # Thus we can do the greedy algorithm here.
        p_q = heapdict()
        tokens_linked_list = deque(cur_tokens)
        for i in range(len(cur_tokens) - 1):
            tup = (cur_tokens[i], cur_tokens[i+1])
            if tup in self.merges_look_up:
                p_q[(tup, i)] = self.merges_look_up[tup]
            
        while p_q:
            top = p_q.popitem()
            tup = top[0]
            ind = tup[1]
            tokens_linked_list.remove(ind)
            tokens_linked_list[ind-1] =  
            tokens_linked_list[ind]





    def encoding_generator(self, non_speical_text:str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for word_match in re.finditer(non_speical_text, PAT):
            yield self.encode_word(word_match) 

    def encode(self, text: str) -> list[int]:        
        #  Encode an input text into a sequence of token IDs.
        re_pattern = "|".join([re.escape(special_token) for special_token in self.special_tokens])
        re_pattern = "(" + re_pattern + ")"
        res = []
        last = 0 
        for m in re.finditer(re_pattern, text):
            last = m.end
            group_s = m.goup()
            if group_s == '':
                continue
            
            group_b = group_s.encode('utf-8')
            if group_b in self.special_tokens_map:
                res.append(self.special_tokens_map[group_b])
            else:
                for word in self.encoding_generator(group_s):
                    # change here.
                    yield word

                res.append(self.encode_word())
        
        # this is some sudo code
        if last < len(text):
            for word in self.encoding_generator(text[last:]):
                yield word

        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of 
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
        # required for memory-efficient tokenization of large files that we cannot directly load into 
        # memory.
        return
    
    def decode(self, ids: list[int]) -> str:



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
    raise NotImplementedError
