from __future__ import annotations

import argparse

import torch
from torch import Tensor
from .tokenizer import get_tokenizer
from pathlib import Path
from .adapters import MyTransformerLM, run_softmax
import os
import json


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    return logits / temperature


def top_p_filter(probs: Tensor, p: float) -> Tensor:
    """Zero out the tail of the distribution so only the smallest set of
    tokens whose cumulative probability exceeds `p` is kept (nucleus sampling).

    Args:
        probs: FloatTensor of shape (..., vocab_size). Should already be a
            valid probability distribution (sums to 1 along the last dim).
        p: Threshold in (0, 1]. p == 1.0 means "keep everything".

    Returns:
        FloatTensor of the same shape, renormalized so it sums to 1 along the
        last dim.
    """

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    keep_sorted = cumsum > p
    keep_sorted[..., 1:] = keep_sorted[..., :-1].clone()
    keep_sorted[..., 0] = False

    remove = torch.zeros_like(keep_sorted)
    remove.scatter_(-1, sorted_idx, keep_sorted)

    probs_filtered = torch.where(remove, torch.zeros_like(probs), probs)
    return probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)


def sample_from(probs: Tensor) -> Tensor:
    """Draw one token id per row from a probability distribution.

    Args:
        probs: FloatTensor of shape (batch, vocab_size).

    Returns:
        LongTensor of shape (batch,).

    """
    return torch.multinomial(probs.squeeze(0), num_samples=1)


def decode(
    model: MyTransformerLM,
    prompt_ids: Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_id: int = 50256,
) -> Tensor:
    """Generate a completion for a single prompt.

    Args:
        model: Anything matching the `LanguageModel` protocol above.
        prompt_ids: LongTensor of shape (seq_len,) — the user-provided x_{1..t}.
        end_of_text_id: Token id that signals end of generation
            (your `<|endoftext|>`).
        max_new_tokens: Stop after generating this many tokens even if EOS
            has not been produced.
        temperature: Softmax temperature. 1.0 = no scaling.
        top_p: Nucleus sampling threshold in (0, 1]. 1.0 disables filtering.

    Returns:
        LongTensor of shape (seq_len + n_generated,) — the prompt followed by
        the generated tokens (including the EOS token if one was sampled).
    """
    context = prompt_ids

    res = prompt_ids
    for i in range(max_new_tokens):
        model_res = model(context)
        logits = model_res[-1:]
        logits = apply_temperature(logits, temperature)
        probs = run_softmax(logits, dim=-1)
        probs = top_p_filter(probs, top_p)
        next_id = sample_from(probs)
        # print(next_id.shape)
        # torch.Size([1]
        # torch.Size([8])
        if next_id == end_of_text_id:
            break
        context = torch.cat((context, next_id), 0)
        res = torch.cat((res, next_id), 0)
        if context.shape[0] > 255:
            context = context[-255:]

    return res


def parse_args() -> argparse.Namespace:
    """Read decoding parameters from the command line.

    Only `--prompt` is required; the rest have sensible defaults.

    Example:
        python decode.py --prompt "Once upon a time" \\
            --max-new-tokens 128 --temperature 0.8 --top-p 0.9
    """
    parser = argparse.ArgumentParser(description="Sample from a language model.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input string to condition the model on.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature; 1.0 = no scaling (default: 1.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold in (0, 1]; 1.0 disables filtering (default: 1.0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open("tests/outputs/tiny_stories_vocab.json") as f:
        tinystories_vocab = json.load(f)
        tinystories_vocab = {int(k): bytes(v) for k, v in tinystories_vocab.items()}

    with open("tests/outputs/tiny_stories_merges.json") as f:
        tinystories_merges = json.load(f)
        tinystories_merges = [(bytes(a), bytes(b)) for (a, b) in tinystories_merges]

    tinystories_tokenizer = get_tokenizer(tinystories_vocab, tinystories_merges, ["<|endoftext|>"])
    model = MyTransformerLM(10000, 256, 512, 1344, 4, 10000, 16, device="mps")
    ckpt = torch.load(args.checkpoint, map_location="mps")
    model.load_state_dict(ckpt["model_state"])
    input = tinystories_tokenizer.encode(args.prompt)
    input = torch.tensor(input, device="mps")

    end_of_text_id = tinystories_tokenizer.encode("<|endoftext|>")[0]
    res = decode(model, input, args.max_new_tokens, args.temperature, args.top_p, end_of_text_id)

    res_str = tinystories_tokenizer.decode(res.tolist())

    print(res_str)
