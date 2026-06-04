from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
from math import sqrt

import numpy.typing as npt
from numpy import random
import torch
import regex as re
import math
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

from collections.abc import Callable, Iterable
from typing import Optional


class MyLinear(torch.nn.Module):
    """
    A linear transformation module.

    Args:
        in_features (int): The size of the input dimension
        out_features (int): The size of the output dimension
        device (torch.device | None = None): Device to store the parameters on
        dtype (torch.dtype | None = None): Data type of the parameters
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.device = device
        self.dtype = dtype
        # self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weights.T
        return out


class MyEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.device = device
        self.dtype = dtype
        # self.reset_parameters()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        out = self.weights[token_ids]
        return out


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    layer = MyLinear(d_in, d_out)
    layer.load_state_dict(
        {
            "weights": weights,
        }
    )
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = MyEmbedding(vocab_size, d_model)
    embedding.load_state_dict({"weights": weights})
    return embedding(token_ids)


class MySwiglu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = torch.nn.Parameter(torch.empty(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w1_w3 = silu_w1_x * einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        out = einsum(self.w2, w1_w3, "d_model d_ff, ... d_ff -> ... d_model")
        return out


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    swiglu = MySwiglu(d_model, d_ff)

    swiglu.load_state_dict({"w1": w1_weight, "w2": w2_weight, "w3": w3_weight})

    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    Q_K_T = einsum(Q, K, "... queries d_k, ... keys d_k-> ... queries keys")
    d_k = Q.shape[-1]
    Q_K_T = Q_K_T / sqrt(d_k)
    if mask is not None:
        infinity_matrix = torch.full_like(Q_K_T, -1 * torch.inf)
        Q_K_T = torch.where(mask, Q_K_T, infinity_matrix)
    soft_maxed_Q_K_t = run_softmax(Q_K_T, -1)
    return einsum(soft_maxed_Q_K_t, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MyMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None, rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.rope = rope
        self.q_proj = torch.nn.Parameter(torch.empty(d_model, self.d_model))
        self.k_proj = torch.nn.Parameter(torch.empty(d_model, self.d_model))
        self.v_proj = torch.nn.Parameter(torch.empty(d_model, self.d_model))
        self.output_proj = torch.nn.Parameter(torch.empty(self.d_model, self.d_model))

    def forward(self, in_features: torch.Tensor, token_positions) -> torch.Tensor:
        v_proj = rearrange(
            self.v_proj, "(num_heads d_v) d_model -> num_heads d_v d_model", num_heads=self.num_heads, d_v=self.d_v
        )
        k_proj = rearrange(
            self.k_proj, "(num_heads d_k) d_model -> num_heads d_k d_model", num_heads=self.num_heads, d_k=self.d_k
        )
        q_proj = rearrange(
            self.q_proj, "(num_heads d_k) d_model -> num_heads d_k d_model", num_heads=self.num_heads, d_k=self.d_k
        )

        w_v_x = einsum(v_proj, in_features, "num_heads d_v d_model, ... d_model -> num_heads ... d_v")
        w_k_x = einsum(k_proj, in_features, "num_heads d_k d_model, ... d_model -> num_heads ... d_k")
        w_q_x = einsum(q_proj, in_features, "num_heads d_k d_model, ... d_model -> num_heads ... d_k")
        if self.rope is not None:
            w_k_x = self.rope(w_k_x, token_positions)
            w_q_x = self.rope(w_q_x, token_positions)
        mask = torch.tril(torch.full((in_features.shape[-2], in_features.shape[-2]), 1)).bool()
        attention = run_scaled_dot_product_attention(w_q_x, w_k_x, w_v_x, mask)
        # attenion is num_heads queries d_v
        concatted_attention = rearrange(attention, "num_heads ... queries d_v -> ... queries (num_heads d_v)")
        return einsum(
            self.output_proj, concatted_attention, "d_model d_model2, ... seq_len d_model2 -> ... seq_len d_model"
        )


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    head_attention = MyMultiHeadAttention(d_model, num_heads)
    head_attention.load_state_dict(
        {
            "q_proj": q_proj_weight,
            "k_proj": k_proj_weight,
            "v_proj": v_proj_weight,
            "output_proj": o_proj_weight,
        }
    )
    return head_attention(in_features, token_positions=None)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    # use torch.triu to construct mask.
    # Rope applied  to q v fo each head.
    d_k = d_model // num_heads
    rope = MyRope(theta, d_k, max_seq_len)
    head_attention = MyMultiHeadAttention(d_model, num_heads, rope=rope)
    head_attention.load_state_dict(
        {
            "q_proj": q_proj_weight,
            "k_proj": k_proj_weight,
            "v_proj": v_proj_weight,
            "output_proj": o_proj_weight,
        },
        strict=False,
    )

    return head_attention(in_features, token_positions=token_positions)


class MyRope(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # torch.arange works like Python's range but returns a tensor:

        #   torch.arange(5)        # tensor([0, 1, 2, 3, 4])
        #   torch.arange(2, 8)     # tensor([2, 3, 4, 5, 6, 7])
        #   torch.arange(0, 1, 0.2)  # tensor([0.0, 0.2, 0.4, 0.6, 0.8])

        #   torch.outer takes two 1D tensors and produces a 2D matrix where each entry (i, j) is a[i] * b[j]:

        #   a = torch.tensor([1, 2, 3])
        #   b = torch.tensor([10, 20])
        #   torch.outer(a, b)
        #   # tensor([[10, 20],
        #   #         [20, 40],
        #   #         [30, 60]])

        arr = torch.arange(1.0, d_k // 2 + 1.0, 1)
        arr = self.theta ** (-1 * (2 * arr - 2) / d_k)
        seq = torch.arange(0, max_seq_len * 1.0, 1)
        # (max_seq_len, d_k / 2)
        angle_matrix = torch.outer(seq, arr)

        cos = torch.cos(angle_matrix)
        sin = torch.sin(angle_matrix)
        top = torch.stack([cos, -1 * sin], dim=2)
        bottom = torch.stack([sin, cos], dim=2)
        # (max_seq_len, d_k / 2, 2, 2)
        # top = (cos, -sin)
        # bottom = (sin, cos)
        # if by dim = 3 it would create (cos sin) instead
        #                               (-sin cos)
        self.register_buffer("rotation_matrix", torch.stack([top, bottom], dim=2))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Transform x into " ... sequence_length (2 d_k/2)" from " ... sequence_length d_k"
        # x is already Qx which is already query mutliplied by x.
        # rotation matrix: (max_seq_len, d_k / 2, 2,)
        x = rearrange(x, "... sequence_length (out two) -> ... sequence_length out two", two=2)
        # Needs ... in the first opeartor because token positions could be tensor as well.
        # The torch syntax is super rich.
        res = einsum(
            self.rotation_matrix[token_positions],
            x,
            "... sequence_length out two_i two_j, ... sequence_length out two_j -> ... sequence_length out two_i",
        )
        res = rearrange(res, "... sequence_length out two -> ... sequence_length (out two)")
        return res

        # Syntax:
        # w1_x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """

    my_rope = MyRope(
        theta,
        d_k,
        max_seq_len,
    )

    return my_rope(in_query_or_key, token_positions)


class MyTransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, rope: MyRope, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = MyRMSNorm(d_model)
        self.attn = MyMultiHeadAttention(d_model, num_heads, rope=rope)
        self.ffn = MySwiglu(d_model, d_ff)
        self.ln2 = MyRMSNorm(d_model)
        self.rope = rope

    def forward(self, in_features: torch.Tensor, token_positions) -> torch.Tensor:
        x = in_features
        t = self.ln1(x)
        t = self.attn(t, token_positions)
        y = x + t
        ff = self.ln2(y)
        ff = self.ffn(ff)
        return ff + y


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    d_k = d_model // num_heads
    rope = MyRope(theta, d_k, max_seq_len)

    transformer_block = MyTransformerBlock(d_model, d_ff, rope, num_heads)

    converted = {}
    for key, value in weights.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[2] == "weight":
            converted[f"{parts[0]}.{parts[1]}"] = value
        else:
            converted[key] = value

    transformer_block.load_state_dict(converted, strict=False)

    token_positions = torch.arange(in_features.shape[-2]).unsqueeze(0).expand(in_features.shape[:-1])
    return transformer_block(in_features, token_positions)


class MyTransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, d_ff, num_layers, rope_theta, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        d_k = d_model // num_heads
        rope = MyRope(rope_theta, d_k, context_length)
        self.rope = rope
        self.layers = torch.nn.ModuleList(
            [MyTransformerBlock(d_model, d_ff, rope, num_heads) for _ in range(num_layers)]
        )

        self.token_embeddings = MyEmbedding(vocab_size, d_model)
        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = MyLinear(d_model, vocab_size)

    def forward(self, in_features: torch.Tensor, token_positions) -> torch.Tensor:
        x = self.token_embeddings(in_features)
        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    model = MyTransformerLM(vocab_size, context_length, d_model, d_ff, num_layers, rope_theta, num_heads)

    converted = {}
    for key, value in weights.items():
        if key in ("token_embeddings.weight", "lm_head.weight"):
            converted[key + "s"] = value
        elif key.endswith(
            (
                ".q_proj.weight",
                ".k_proj.weight",
                ".v_proj.weight",
                ".output_proj.weight",
                ".w1.weight",
                ".w2.weight",
                ".w3.weight",
            )
        ):
            converted[key.rsplit(".weight", 1)[0]] = value
        else:
            converted[key] = value

    model.load_state_dict(converted, strict=False)

    token_positions = torch.arange(in_indices.shape[-1]).unsqueeze(0).expand(in_indices.shape)
    return model(in_indices, token_positions)


class MyRMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weight
        return x.to(in_dtype)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_norm = MyRMSNorm(d_model, eps)
    rms_norm.load_state_dict({"g": weights})
    return rms_norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.


    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    sample_start_ind = random.choice(len(dataset) - context_length, batch_size)
    device = torch.device(device)

    input_tensor = torch.tensor([dataset[i : i + context_length] for i in sample_start_ind], device=device)
    output_tensor = torch.tensor([dataset[i + 1 : i + 1 + context_length] for i in sample_start_ind], device=device)

    return (input_tensor, output_tensor)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    dim_max = torch.amax(in_features, dim=dim, keepdim=True)
    in_features_subtraced = in_features - dim_max
    in_features_subtraced = torch.exp(in_features_subtraced)
    in_features_subtraced_sum = torch.sum(in_features_subtraced, dim=dim, keepdim=True)
    return in_features_subtraced / in_features_subtraced_sum


def run_cross_entropy(
    inputs: Float[Tensor, "... batch_size vocab_size"], targets: Int[Tensor, "... batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

    # expanding the loss function, the first term is just the sum of all logits for the target index, the second term is the sum of the sum of logits for each sample.
    total_sample = math.prod(inputs.shape[:-1])
    one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[-1]).float()
    result = (inputs * one_hot).sum(-1)
    first_term = -1 * einsum(result, "... b->")
    max_inputs = torch.max(inputs, dim=-1, keepdim=True).values
    inputs = inputs - max_inputs
    exp_sum = einsum(torch.exp(inputs), "... b v ->... b")
    second_term_tensor = torch.log(exp_sum)

    second_term = einsum(second_term_tensor, "... b ->")
    third_term = einsum(max_inputs, "...->")
    return (first_term + second_term + third_term) / total_sample


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # grads = [p.grad for p in parameters if p.grad is not None]
    # if not grads:
    #     return
    # # total_norm = torch.sqrt(sum([(g**2).sum() for g in grads]))
    # total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    # if total_norm >= max_l2_norm:
    #     scale = max_l2_norm / (total_norm + 1e-6)
    #     for g in grads:
    #         g.mul(scale)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    total_norm = torch.sqrt(sum([(g.detach() ** 2).sum() for g in grads]))

    eps = 1e-6
    clip_coef = max_l2_norm / (total_norm + eps)
    if total_norm > max_l2_norm:
        for g in grads:
            g.detach().mul_(clip_coef)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                # learning_rate
                lr_t = lr * sqrt(1 - b2 ** (t + 1)) / (1 - b1 ** (t + 1))
                p.data -= lr * p.data * weight_decay
                state["m"] = b1 * m + (1 - b1) * grad
                state["v"] = b2 * v + (1 - b2) * (grad**2)
                state["t"] = t + 1
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
        return loss


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    t_w = warmup_iters
    t_c = cosine_cycle_iters

    if it < t_w:
        return it / t_w * max_learning_rate
    elif it <= t_c:
        angle = (it - t_w) / (t_c - t_w) * math.pi
        return min_learning_rate + 1 / 2 * (1 + math.cos(angle)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """

    torch.save(
        {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "iteration": iteration}, out
    )


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]
