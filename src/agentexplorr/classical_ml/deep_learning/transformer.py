"""
Transformer from Scratch — "Attention Is All You Need"
========================================================

A complete, from-scratch implementation of the Transformer architecture using
only basic PyTorch primitives (nn.Linear, nn.LayerNorm, nn.Embedding). We do
NOT use nn.Transformer or nn.MultiheadAttention — we build everything ourselves
to understand every single piece.

THIS IS THE MOST IMPORTANT FILE IN THE ENTIRE PROJECT FOR YOUR RESUME.
It demonstrates deep understanding of the architecture behind GPT, BERT,
LLaMA, and every modern LLM.

THE TRANSFORMER ARCHITECTURE:
    Introduced in "Attention Is All You Need" (Vaswani et al., 2017).
    Before Transformers, sequence models used RNNs/LSTMs which process tokens
    ONE AT A TIME (slow, hard to parallelize, forget long-range dependencies).

    The Transformer's key innovation: SELF-ATTENTION, which lets every token
    look at every other token in PARALLEL. This enables:
      1. Parallelized training (no sequential dependency like RNNs)
      2. Long-range dependencies (token 1 can directly attend to token 1000)
      3. Scalability (the foundation for GPT-4, Claude, LLaMA, etc.)

SELF-ATTENTION: THE CORE MECHANISM

    The "Query-Key-Value" analogy (think of a library):

    QUERY (Q): "What am I looking for?"
        Each token generates a query vector representing what information it needs.
        Like a person walking into a library with a specific question.

    KEY (K): "What information do I contain?"
        Each token generates a key vector advertising what information it has.
        Like the titles and descriptions on library books.

    VALUE (V): "Here is my actual content."
        Each token generates a value vector containing its information.
        Like the actual content inside each book.

    ATTENTION COMPUTATION:
        1. Compute relevance: score = Q * K^T / sqrt(d_k)
           (How well does my query match each key?)
        2. Normalize: weights = softmax(scores)
           (Convert scores to probabilities that sum to 1)
        3. Retrieve: output = weights * V
           (Weighted sum of values, emphasizing relevant tokens)

    WHY SCALE BY sqrt(d_k)?
        Without scaling, dot products grow with dimension (d_k), pushing
        softmax into saturation (all probability mass on one token).
        Dividing by sqrt(d_k) keeps the variance of scores at ~1,
        preserving informative gradients.

    ASCII DIAGRAM OF MULTI-HEAD ATTENTION:

        Input: (batch, seq_len, d_model)
                    |
            +-------+-------+
            |       |       |
        Q = XW_Q  K = XW_K  V = XW_V     <-- Linear projections
            |       |       |
        +---+---+---+---+---+---+
        |Head 1 |Head 2 |Head h |         <-- Split into h heads
        | QKV   | QKV   | QKV   |         <-- Each head attends independently
        +---+---+---+---+---+---+
            |       |       |
            +-------+-------+
                    |
                Concat + W_O               <-- Concatenate heads, project back
                    |
            Output: (batch, seq_len, d_model)

    WHY MULTIPLE HEADS?
        Each head can attend to different types of relationships:
          - Head 1 might attend to syntactic relationships (subject-verb)
          - Head 2 might attend to semantic relationships (synonyms)
          - Head 3 might attend to positional relationships (nearby tokens)
        Multiple perspectives give a richer representation.

POSITIONAL ENCODING:

    WHY DO WE NEED POSITIONAL ENCODING?
        Self-attention is permutation-invariant — it treats "dog bites man"
        and "man bites dog" identically because it has no notion of order.
        We must explicitly inject position information.

    SINUSOIDAL POSITIONAL ENCODING (from the original paper):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Where:
          - pos = position in the sequence (0, 1, 2, ...)
          - i = dimension index
          - d_model = embedding dimension

        WHY SINUSOIDAL?
          1. Each position gets a unique encoding
          2. The encoding is deterministic (no learned parameters)
          3. Relative positions can be expressed as linear transformations:
             PE(pos + k) can be represented as a linear function of PE(pos)
          4. Generalizes to unseen sequence lengths (unlike learned encodings)

FULL TRANSFORMER BLOCK:

    Input -> LayerNorm -> MultiHeadAttention -> + (residual) -> LayerNorm -> FFN -> + (residual) -> Output
             |                                  ^               |                   ^
             +----------------------------------+               +-------------------+
                     (skip connection)                            (skip connection)

    RESIDUAL (SKIP) CONNECTIONS:
        output = x + sublayer(norm(x))
        The "+ x" part is the skip connection. It lets gradients flow directly
        from output to input, preventing vanishing gradients in deep networks.
        This is what makes training 100+ layer networks possible.

    LAYER NORMALIZATION:
        Normalizes activations across the feature dimension (not the batch).
        More stable than BatchNorm for variable-length sequences.
        We use "Pre-LN" (normalize before attention/FFN) — more stable than
        the original "Post-LN" (normalize after).

    FEED-FORWARD NETWORK (FFN):
        Two linear layers with a non-linearity in between:
            FFN(x) = Linear2(GELU(Linear1(x)))
        Linear1 expands: d_model -> d_ff (typically 4 * d_model)
        Linear2 compresses: d_ff -> d_model
        This gives each position a chance to "think" independently.

COMPLETE ARCHITECTURE (ASCII):

    +---------------------------------------------------------+
    |  Input: [4, 12, 7, 3, 1]  (token IDs)                  |
    |      |                                                   |
    |  Embedding: each ID -> d_model vector                   |
    |      |                                                   |
    |  + Positional Encoding (sinusoidal)                     |
    |      |                                                   |
    |  +-------------------------------------+                |
    |  | Transformer Block 1                  |                |
    |  |  LayerNorm -> MultiHead Attention    |                |
    |  |  + Residual                          |                |
    |  |  LayerNorm -> FFN                    |                |
    |  |  + Residual                          |                |
    |  +-------------------------------------+                |
    |      |                                                   |
    |  +-------------------------------------+                |
    |  | Transformer Block 2                  |                |
    |  |  (same structure)                    |                |
    |  +-------------------------------------+                |
    |      |                                                   |
    |  ... (N blocks)                                          |
    |      |                                                   |
    |  LayerNorm (final)                                       |
    |      |                                                   |
    |  Linear(d_model -> vocab_size) -> Logits                |
    |      |                                                   |
    |  Output: probability distribution over vocabulary        |
    +---------------------------------------------------------+

PAPER REFERENCE:
    Vaswani, A., et al. (2017). "Attention Is All You Need."
    arXiv:1706.03762. https://arxiv.org/abs/1706.03762

LEARNING RESOURCES:
    - The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
    - The Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
    - VIDEO: "Attention in Transformers, Visually Explained" (3Blue1Brown) — https://www.youtube.com/watch?v=eMlx5fFNoYc
    - VIDEO: "Let's build GPT from scratch" (Andrej Karpathy) — https://www.youtube.com/watch?v=kCc8FmEb1nY
    - VIDEO: "Transformer Neural Networks Explained" (CodeEmporium) — https://www.youtube.com/watch?v=TQQlZhbC5ps
    - Stanford CS224n Lecture on Transformers: https://web.stanford.edu/class/cs224n/
    - GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    - BERT paper: https://arxiv.org/abs/1810.04805
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# COMPONENT 1: Multi-Head Self-Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention -- The heart of the Transformer.

    WHAT IS ATTENTION?
      Attention answers: "For each word, how much should I focus on every
      other word?" It produces a weighted combination of all words, where
      the weights represent relevance.

    THE Q, K, V ANALOGY:
      Think of it like a search engine:
      - **Query (Q)**: Your search query ("what am I looking for?")
      - **Key (K)**: The index/title of each document ("what does this contain?")
      - **Value (V)**: The actual content of each document ("here's the info")

      Attention score = how well Q matches K
      Output = weighted sum of V, weighted by attention scores

    WHY MULTI-HEAD?
      A single attention head learns ONE type of relationship.
      Multiple heads learn DIFFERENT types simultaneously:
      - Head 1 might learn syntactic relationships (subject-verb)
      - Head 2 might learn semantic relationships (pronouns-antecedents)
      - Head 3 might learn positional relationships (adjacent words)

    THE MATH:
      Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

      WHY divide by sqrt(d_k)?
        Without scaling, the dot products grow with d_k, pushing softmax
        into regions with very small gradients (saturated softmax).
        Dividing by sqrt(d_k) keeps the variance at 1, maintaining
        healthy gradients. This is "scaled" dot-product attention.

    Args:
        d_model: Total dimension of the model (e.g., 512).
        n_heads: Number of attention heads (e.g., 8).
            Must evenly divide d_model. Each head operates on d_model/n_heads dims.
        dropout: Dropout rate for attention weights.

    Example:
        >>> mha = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
        >>> output = mha(x, x, x)  # self-attention: Q=K=V=x
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        # d_model must be divisible by n_heads so we can split evenly
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # --- Linear projections for Q, K, V ---
        # These are learned transformations that project the input into
        # query, key, and value spaces. Each is a (d_model x d_model) matrix.
        #
        # WHY SEPARATE PROJECTIONS?
        #   The same token might be "querying" for syntactic info while its
        #   "key" advertises semantic info. Separate projections let Q, K, V
        #   encode different aspects of the same token.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection: concatenated heads -> d_model
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout applied to attention weights (prevents over-focusing on specific tokens)
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization (set during forward pass)
        self.attention_weights: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        STEP-BY-STEP:
            1. Project input to Q, K, V using learned weight matrices
            2. Reshape to separate heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)
            3. Compute scaled dot-product attention for each head
            4. Concatenate heads and project back to d_model

        For SELF-attention, query = key = value = x (the same input).
        For CROSS-attention (encoder-decoder), key & value come from the encoder.

        Args:
            query: (batch_size, seq_len, d_model) -- what we're looking for
            key:   (batch_size, seq_len, d_model) -- what's available
            value: (batch_size, seq_len, d_model) -- the actual content
            mask:  (batch_size, 1, 1, seq_len) or (1, 1, seq_len, seq_len)
                Positions with 0 are MASKED (not attended to).
                Used for:
                  - Padding mask: ignore <PAD> tokens
                  - Causal mask: prevent attending to future tokens

        Returns:
            (batch_size, seq_len, d_model) -- attended output.
        """
        batch_size = query.size(0)

        # Step 1: Project Q, K, V through linear layers
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Split into multiple heads
        # Reshape: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        # We rearrange dimensions so each head operates independently
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Compute scaled dot-product attention
        # scores shape: (batch, n_heads, seq_len, seq_len)
        # Each entry [b, h, i, j] = "how much should position i attend to position j
        #                             in batch b, head h?"
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 4: Apply mask (if provided)
        # For causal/autoregressive models, we mask future positions
        # so the model can only attend to previous tokens.
        # We set masked positions to -inf BEFORE softmax, so they get ~0 probability.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: Softmax to get attention weights (probabilities that sum to 1)
        attention_weights = F.softmax(scores, dim=-1)

        # Store for visualization (useful for debugging and understanding what
        # the model is "looking at" for each token)
        self.attention_weights = attention_weights.detach()

        # Apply dropout to attention weights
        # This randomly zeros out some attention connections during training,
        # preventing the model from relying too heavily on any single connection.
        attention_weights = self.dropout(attention_weights)

        # Step 6: Multiply attention weights by values
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k)
        # -> (batch, n_heads, seq_len, d_k)
        # This produces a weighted combination of values for each position.
        attended = torch.matmul(attention_weights, V)

        # Step 7: Concatenate all heads
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads * d_k) = (batch, seq_len, d_model)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # Step 8: Final linear projection
        # This mixes information across heads, allowing them to communicate.
        return self.W_o(attended)


# =============================================================================
# COMPONENT 2: Positional Encoding (Sinusoidal)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding -- Giving the Model a Sense of Order.

    WHY IS THIS NEEDED?
      Self-attention is permutation-invariant -- it treats "The cat sat" the
      same as "sat cat The". The model has NO notion of word order!

      Positional encoding adds information about each token's position so
      the model knows that "The" is first, "cat" is second, etc.

    WHY SINUSOIDAL?
      The original Transformer uses sine and cosine functions:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

      This works because:
      1. Each position gets a unique encoding
      2. Relative positions can be computed via linear transformation
         (the model can learn "3 positions to the left")
      3. It generalizes to sequence lengths not seen during training

    NOTE:
      Modern models (GPT, LLaMA) use LEARNED positional embeddings instead
      of sinusoidal ones. RoPE (Rotary Position Embedding) is now standard.
      We use sinusoidal here because it's easier to understand and was the
      original design in "Attention Is All You Need".

    Args:
        d_model: Dimension of the model.
        max_len: Maximum sequence length to pre-compute encodings for.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute positional encodings for all positions up to max_len
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1] as a column vector
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division term: 10000^(2i/d_model) computed in log space for stability
        # exp(2i * -log(10000) / d_model) = 1 / 10000^(2i/d_model)
        # This creates different frequencies for different dimensions:
        #   - Low dimensions (small i) = high frequency (captures fine positions)
        #   - High dimensions (large i) = low frequency (captures coarse positions)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even dimensions (0, 2, 4, ...) get sine
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd dimensions (1, 3, 5, ...) get cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as a buffer (not a parameter -- we don't want gradients for PE)
        # Buffers are saved with the model state but NOT updated by the optimizer.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (batch_size, seq_len, d_model) -- token embeddings.

        Returns:
            (batch_size, seq_len, d_model) -- embeddings + positional info.
        """
        # Add positional encoding (broadcasting handles the batch dimension)
        # We only take the first seq_len positions from the pre-computed PE
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# =============================================================================
# COMPONENT 3: Position-wise Feed-Forward Network
# =============================================================================

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    WHAT THIS DOES:
      After attention computes relationships BETWEEN tokens, the feed-forward
      network processes EACH token INDEPENDENTLY. Think of it as:
      - Attention = "gather information from other tokens"
      - Feed-forward = "process the gathered information"

    THE ARCHITECTURE:
      Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model) -> Dropout

    WHY d_ff > d_model?
      The hidden dimension (d_ff) is typically 4x the model dimension.
      This expansion gives the network more capacity to learn complex
      transformations before projecting back down. Think of it as
      briefly "thinking in a larger space" before compressing back.

    WHY GELU (not ReLU)?
      GELU (Gaussian Error Linear Unit) = x * Phi(x) where Phi is the
      standard Gaussian CDF. Unlike ReLU (which has a hard cutoff at 0),
      GELU is smooth everywhere. This gives slightly better performance
      in Transformers. GPT-2, BERT, and most modern models use GELU.

    Args:
        d_model: Model dimension (input/output size).
        d_ff: Feed-forward hidden dimension (typically 4 * d_model).
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),     # Expand: d_model -> d_ff
            nn.GELU(),                     # Non-linearity (smoother than ReLU)
            nn.Dropout(dropout),           # Regularization
            nn.Linear(d_ff, d_model),      # Compress: d_ff -> d_model
            nn.Dropout(dropout),           # Regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network to each position independently.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        return self.net(x)


# =============================================================================
# COMPONENT 4: Transformer Block (Attention + FFN + Residuals + LayerNorm)
# =============================================================================

class TransformerBlock(nn.Module):
    """A Single Transformer Block -- Attention + Feed-Forward + Residuals.

    THE BLOCK STRUCTURE (Pre-LN):
      +----------------+
      |    Input x     |
      +-------+--------+
              |
      +-------v--------+
      |  LayerNorm     |   <-- Pre-norm (modern) vs Post-norm (original)
      |  Multi-Head    |
      |  Attention     |
      |  + Dropout     |
      +-------+--------+
              |
        x + --|          <-- Residual connection: output = f(x) + x
              |
      +-------v--------+
      |  LayerNorm     |
      |  FeedForward   |
      |  + Dropout     |
      +-------+--------+
              |
        x + --|          <-- Another residual connection
              |
      +-------v--------+
      |    Output      |
      +----------------+

    WHY RESIDUAL CONNECTIONS?
      Without residuals, deep networks suffer from vanishing gradients --
      the signal gets weaker as it passes through more layers. Residual
      connections create a "highway" for gradients to flow directly from
      output back to input, enabling very deep networks (100+ layers).

      Formally: output = f(x) + x
      Gradient: d(output)/dx = df/dx + 1  (the +1 prevents vanishing!)

    WHY LAYER NORMALIZATION?
      LayerNorm normalizes across the feature dimension for each sample.
      This stabilizes training by ensuring activations don't grow or shrink
      too much between layers. Without normalization, Transformers are
      extremely hard to train.

    PRE-NORM vs POST-NORM:
      Original paper: Add & Norm AFTER attention/FFN (post-norm)
      Modern practice: Norm BEFORE attention/FFN (pre-norm)
      Pre-norm is more stable and easier to train. We use pre-norm here.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Pre-norm: normalize BEFORE attention and feed-forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Process input through one Transformer block.

        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask (1=attend, 0=mask).

        Returns:
            (batch_size, seq_len, d_model)
        """
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        # Pre-LN: normalize BEFORE the sublayer
        normed = self.norm1(x)
        # Self-attention: query = key = value (all the same input)
        attended = self.attention(normed, normed, normed, mask)
        # Residual connection: add input to sublayer output
        x = x + self.dropout(attended)

        # --- Sub-layer 2: Feed-Forward Network ---
        normed = self.norm2(x)
        fed_forward = self.feed_forward(normed)
        # Residual connection
        x = x + self.dropout(fed_forward)

        return x


# =============================================================================
# COMPONENT 5: The Full MiniTransformer
# =============================================================================

class MiniTransformer(nn.Module):
    """A complete Transformer model -- built entirely from scratch.

    This is a DECODER-ONLY Transformer (like GPT) designed for next-token
    prediction (language modeling). It uses causal (autoregressive) masking
    so each token can only attend to previous tokens and itself.

    ARCHITECTURE SUMMARY:
      Token IDs -> Embedding -> Positional Encoding -> N x TransformerBlock -> LayerNorm -> Linear -> Logits

    MODEL SIZE CALCULATION:
      For d_model=256, n_heads=4, n_layers=4, vocab_size=10000:
      - Embedding: 10000 x 256 = 2.56M params
      - Per block: ~0.8M params (attention + FFN + norms)
      - Total: ~2.56M + 4 x 0.8M + 0.8M ~ 6.6M parameters

      For reference:
      - GPT-2 Small: 117M params (d=768, 12 heads, 12 layers)
      - GPT-3: 175B params (d=12288, 96 heads, 96 layers)
      - Our MiniTransformer: ~6.6M (perfect for learning on a laptop!)

    LEARNING NOTE:
      This implementation follows the same architecture as GPT-2/3, just smaller.
      If you understand this code, you understand GPT. The only differences at
      scale are: more layers, wider dimensions, and training data.

    CAUSAL MASKING:
        For language modeling, token at position t should only attend to
        positions 0, 1, ..., t (not future tokens). We enforce this with a
        causal mask (lower-triangular matrix of ones):

        Mask for seq_len=4:
        [[1, 0, 0, 0],    <-- token 0 sees only itself
         [1, 1, 0, 0],    <-- token 1 sees tokens 0-1
         [1, 1, 1, 0],    <-- token 2 sees tokens 0-2
         [1, 1, 1, 1]]    <-- token 3 sees tokens 0-3

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Dimension of embeddings and hidden states.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        d_ff: Feed-forward hidden dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.

    Example:
        >>> model = MiniTransformer(vocab_size=10000, d_model=256, n_heads=4, n_layers=4)
        >>> input_ids = torch.randint(0, 10000, (2, 32))  # (batch=2, seq_len=32)
        >>> logits = model(input_ids)  # (2, 32, 10000)
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int | None = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding: maps token IDs to dense vectors
        # Each of the vocab_size tokens gets its own d_model-dimensional vector.
        # These vectors are LEARNED during training -- tokens with similar
        # meanings will end up with similar embeddings.
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: adds position information
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Stack of N Transformer blocks
        # nn.ModuleList (not a regular Python list!) ensures PyTorch can
        # find and track all parameters in the blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm (pre-norm architecture requires this at the end)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: maps hidden states to vocabulary logits
        # This is the "language model head" -- it predicts the next token.
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Weight tying: share weights between token embedding and output projection
        # WHY? The output projection is doing the REVERSE of embedding:
        #   embedding: token_id -> vector
        #   output:    vector -> token_id (logits)
        # Sharing weights reduces parameters and improves performance.
        # Paper: "Using the Output Embedding to Improve Language Models" (2016)
        self.output_projection.weight = self.token_embedding.weight

        # Initialize weights (crucial for stable training)
        self._init_weights()

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "mini_transformer_initialized",
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff or 4 * d_model,
            max_seq_len=max_seq_len,
            total_params=total_params,
        )

    def _init_weights(self) -> None:
        """Initialize weights for stable training.

        WHY INITIALIZATION MATTERS:
          Poor initialization can make training impossible:
          - Too large -> activations and gradients explode
          - Too small -> activations and gradients vanish
          - All zeros -> all neurons learn the same thing (symmetry problem)

          Xavier uniform initialization keeps the variance of activations
          roughly constant across layers, enabling stable training.
          Embedding layers use normal(0, 0.02) -- a common choice in GPT models.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform: maintains activation variance across layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Small normal initialization for embeddings (GPT convention)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: bias=0, weight=1 (standard)
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask that prevents attending to future tokens.

        WHY CAUSAL MASKING?
          In language modeling, we predict the next token. The model should
          only see PREVIOUS tokens, not future ones. The causal mask sets
          future positions to 0 so that masked_fill replaces them with -inf
          in the attention scores, giving them zero probability after softmax.

          For seq_len=4:
            [[1, 0, 0, 0],    token 0 can only see token 0
             [1, 1, 0, 0],    token 1 can see tokens 0-1
             [1, 1, 1, 0],    token 2 can see tokens 0-2
             [1, 1, 1, 1]]    token 3 can see tokens 0-3

        Args:
            seq_len: Length of the sequence.
            device: Device to create the mask on.

        Returns:
            (1, 1, seq_len, seq_len) -- causal attention mask.
        """
        # torch.tril creates a lower triangular matrix (upper triangle = 0)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Add batch and head dimensions for broadcasting
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: token IDs -> next-token logits.

        Args:
            input_ids: (batch_size, seq_len) -- integer token IDs.

        Returns:
            (batch_size, seq_len, vocab_size) -- logits for each position.
            The logits at position i predict the token at position i+1.
            Apply softmax to get probabilities: probs = torch.softmax(logits, dim=-1)
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Token embedding
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embedding(input_ids)

        # Scale embeddings by sqrt(d_model) -- this is from the original paper.
        # It ensures embedding magnitudes are in the same range as positional
        # encodings, preventing one from drowning out the other.
        x = x * math.sqrt(self.d_model)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Create causal mask (autoregressive -- can't see future)
        mask = self._create_causal_mask(seq_len, input_ids.device)

        # Step 4: Pass through N Transformer blocks
        # Each block refines the representation through attention and FFN.
        for block in self.blocks:
            x = block(x, mask)

        # Step 5: Final layer norm
        x = self.final_norm(x)

        # Step 6: Project to vocabulary logits
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        logits = self.output_projection(x)

        return logits

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Useful for comparing model sizes and estimating memory requirements.
        Each float32 parameter = 4 bytes, so total memory ~ params * 4 bytes.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_parameters(self) -> dict[str, int]:
        """Count parameters by component.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        counts: dict[str, int] = {
            "token_embedding": sum(
                p.numel() for p in self.token_embedding.parameters()
            ),
            "positional_encoding": 0,  # No learned parameters (sinusoidal)
            "transformer_blocks": sum(
                p.numel() for block in self.blocks for p in block.parameters()
            ),
            "final_norm": sum(
                p.numel() for p in self.final_norm.parameters()
            ),
            "output_projection": 0,  # Weight-tied with embedding (not double-counted)
        }
        counts["TOTAL"] = sum(p.numel() for p in self.parameters())

        return counts

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        HOW AUTOREGRESSIVE GENERATION WORKS:
          1. Feed the input tokens through the model
          2. Get the logits for the LAST position (next token prediction)
          3. Sample a token from the logits
          4. Append the sampled token to the input
          5. Repeat steps 1-4

        TEMPERATURE:
            Controls randomness. Applied before softmax:
              logits_scaled = logits / temperature

            temperature = 1.0: normal sampling (model's learned distribution)
            temperature < 1.0: more confident/deterministic (sharper distribution)
            temperature > 1.0: more creative/random (flatter distribution)
            temperature -> 0: greedy decoding (always pick the most likely token)
            temperature -> inf: uniform random (all tokens equally likely)

        TOP-K SAMPLING:
            Only consider the top K most likely tokens. Zeroes out all others.
            This prevents the model from generating very unlikely tokens while
            still allowing diversity among plausible options.

            k=1: greedy (always pick the top token)
            k=50: choose among top 50 tokens (common default)
            k=None: consider all tokens

        Args:
            input_ids: (batch, seq_len) -- starting token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Top-k filtering (None = no filtering).

        Returns:
            (batch, seq_len + max_new_tokens) -- input + generated tokens.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate to max sequence length (positional encoding limit)
            # If the sequence is longer than 512, we slide the window
            input_seq = input_ids[:, -512:]

            # Get logits from the model
            logits = self(input_seq)

            # Only need the last position's logits (next-token prediction)
            next_token_logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                # Find the k-th largest value
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                # Set all logits below the k-th largest to -inf
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample from the distribution (multinomial sampling)
            # This is what makes generation stochastic (non-deterministic)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ---------------------------------------------------------------------------
# Quick demo -- run this file directly to see the Transformer in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("MINI TRANSFORMER FROM SCRATCH")
    print("Architecture: Decoder-only (GPT-style)")
    print("Reference: 'Attention Is All You Need' (Vaswani et al., 2017)")
    print("=" * 70)

    # --- Create a small Transformer ---
    model = MiniTransformer(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=512,
        dropout=0.1,
    )

    # --- Print architecture ---
    print("\nModel Architecture:")
    print(model)

    # --- Parameter count ---
    print("\nParameter Counts:")
    params = model.get_num_parameters()
    for name, count in params.items():
        if name == "TOTAL":
            print(f"  {'_' * 45}")
        print(f"  {name}: {count:,}")

    # --- Test forward pass ---
    print("\nForward Pass Test:")
    # Random token IDs (simulating a batch of 2 sequences, each length 20)
    dummy_tokens = torch.randint(0, 10000, (2, 20))
    logits = model(dummy_tokens)
    print(f"  Input shape:  {dummy_tokens.shape}  (batch=2, seq_len=20)")
    print(f"  Output shape: {logits.shape}  (batch=2, seq_len=20, vocab=10000)")

    # --- Test generation ---
    print("\nGeneration Test:")
    prompt = torch.randint(0, 10000, (1, 5))  # 5-token prompt
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"  Prompt length:    {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    # --- Attention weights ---
    print("\nAttention Weight Shape (from last block):")
    last_block = model.blocks[-1]
    if last_block.attention.attention_weights is not None:
        attn_shape = last_block.attention.attention_weights.shape
        print(f"  {attn_shape}  (batch, heads, seq_len, seq_len)")

    print("\nTransformer is ready! Pair with Trainer from training_loop.py to train.")
    print("For a real language model, you'd need a tokenizer and a text corpus.")
