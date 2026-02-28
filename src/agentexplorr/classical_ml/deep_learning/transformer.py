"""
Transformer From Scratch — The Architecture That Changed Everything
====================================================================

WHAT IS A TRANSFORMER?
  The Transformer is the architecture behind GPT, BERT, LLaMA, Claude,
  and virtually every modern LLM. Understanding it deeply is arguably
  the single most important thing for an AI career.

  Before Transformers (2017), NLP used RNNs (LSTMs, GRUs) which processed
  text sequentially — one word at a time. Transformers process ALL words
  in PARALLEL using "self-attention", which is dramatically faster and
  captures long-range dependencies better.

THE KEY INSIGHT — SELF-ATTENTION:
  For the sentence "The cat sat on the mat because it was tired":
  - An RNN processes: "The" → "cat" → "sat" → ... → "tired" (sequential)
  - A Transformer processes ALL words simultaneously and learns:
    "it" → refers to "cat" (not "mat") via attention weights

  Self-attention computes: "For each word, how relevant is every other word?"

THE ARCHITECTURE (simplified):
  ┌────────────────────────────────────────┐
  │            Input Tokens                │
  │         [The, cat, sat, ...]           │
  └───────────────┬────────────────────────┘
                  │
                  ▼
  ┌────────────────────────────────────────┐
  │    Token Embedding + Positional Enc.   │  ← Words → Vectors + Position info
  └───────────────┬────────────────────────┘
                  │
                  ▼
  ┌────────────────────────────────────────┐
  │        Transformer Block × N           │
  │  ┌──────────────────────────────────┐  │
  │  │  Multi-Head Self-Attention       │  │  ← "Which words matter for each word?"
  │  │  + Residual Connection           │  │
  │  │  + Layer Normalization           │  │
  │  ├──────────────────────────────────┤  │
  │  │  Feed-Forward Network            │  │  ← Process each position independently
  │  │  + Residual Connection           │  │
  │  │  + Layer Normalization           │  │
  │  └──────────────────────────────────┘  │
  └───────────────┬────────────────────────┘
                  │
                  ▼
  ┌────────────────────────────────────────┐
  │         Output Projection              │  ← Vectors → Token Probabilities
  └────────────────────────────────────────┘

PAPER:
  "Attention Is All You Need" — Vaswani et al., 2017
  https://arxiv.org/abs/1706.03762

  This paper introduced the Transformer architecture and is one of the
  most cited papers in all of computer science. The title is not hyperbole —
  they showed that attention alone (without recurrence or convolution)
  achieves state-of-the-art results on machine translation.

LEARNING RESOURCES:
  - "The Illustrated Transformer" (Jay Alammar):
    https://jalammar.github.io/illustrated-transformer/
  - "Attention Is All You Need" paper: https://arxiv.org/abs/1706.03762
  - VIDEO: "Attention in Transformers" (3Blue1Brown):
    https://www.youtube.com/watch?v=eMlx5fFNoYc
  - VIDEO: "Let's build GPT from scratch" (Andrej Karpathy):
    https://www.youtube.com/watch?v=kCc8FmEb1nY
  - VIDEO: "Transformer Neural Networks Explained" (CodeEmporium):
    https://www.youtube.com/watch?v=TQQlZhbC5ps
  - "The Annotated Transformer" (Harvard NLP):
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention — The heart of the Transformer.

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
        dropout: Dropout rate for attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        # d_model must be divisible by n_heads so we can split evenly
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Each projects from d_model to d_model (all heads combined)
        # We'll split into heads after projection
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection: combines all heads back to d_model
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            query: (batch_size, seq_len, d_model) — what we're looking for
            key:   (batch_size, seq_len, d_model) — what's available
            value: (batch_size, seq_len, d_model) — the actual content
            mask:  (batch_size, 1, 1, seq_len) — attention mask (0 = attend, -inf = ignore)

        Returns:
            (batch_size, seq_len, d_model) — attended output
        """
        batch_size = query.size(0)

        # Step 1: Project Q, K, V through linear layers
        # Shape: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Split into multiple heads
        # Reshape: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        # We rearrange dimensions so each head operates independently
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Compute scaled dot-product attention
        # scores shape: (batch, n_heads, seq_len, seq_len)
        # Each entry [i,h,j,k] = "how much should position j attend to position k in head h?"
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 4: Apply mask (if provided)
        # For causal/autoregressive models, we mask future positions
        # so the model can only attend to previous tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: Softmax to get attention weights (probabilities that sum to 1)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 6: Multiply attention weights by values
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k)
        # → (batch, n_heads, seq_len, d_k)
        attended = torch.matmul(attention_weights, V)

        # Step 7: Concatenate all heads
        # (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Step 8: Final linear projection
        return self.W_o(attended)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding — Giving the Model a Sense of Order.

    WHY IS THIS NEEDED?
      Self-attention is permutation-invariant — it treats "The cat sat" the
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
        max_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create the positional encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division term: 10000^(2i/d_model) computed as exp(2i * -log(10000)/d_model)
        # This creates different frequencies for different dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even dimensions get sine, odd dimensions get cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) → (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter — no gradients, but saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (batch_size, seq_len, d_model) — token embeddings.

        Returns:
            (batch_size, seq_len, d_model) — embeddings + positional info.
        """
        # Add positional encoding (broadcasting handles batch dimension)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    WHAT THIS DOES:
      After attention computes relationships BETWEEN tokens, the feed-forward
      network processes EACH token INDEPENDENTLY. Think of it as:
      - Attention = "gather information from other tokens"
      - Feed-forward = "process the gathered information"

    THE ARCHITECTURE:
      Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)

    WHY d_ff > d_model?
      The hidden dimension (d_ff) is typically 4× the model dimension.
      This expansion gives the network more capacity to learn complex
      transformations before projecting back down.

    WHY GELU (not ReLU)?
      GELU (Gaussian Error Linear Unit) is smoother than ReLU and works
      better for Transformers. It was introduced in the BERT paper and is
      now standard in all modern Transformer variants.

    Args:
        d_model: Model dimension.
        d_ff: Feed-forward hidden dimension (typically 4 * d_model).
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network to each position independently.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """A Single Transformer Block — Attention + Feed-Forward + Residuals.

    THE BLOCK STRUCTURE:
      ┌──────────────┐
      │    Input x    │
      ├──────────────┤
      │  LayerNorm   │  ← Pre-norm (modern) vs Post-norm (original)
      │  Attention   │
      │  + Dropout   │
      │  + x (resid) │  ← Residual connection: output = f(x) + x
      ├──────────────┤
      │  LayerNorm   │
      │  FeedForward │
      │  + Dropout   │
      │  + x (resid) │  ← Another residual connection
      └──────────────┘

    WHY RESIDUAL CONNECTIONS?
      Without residuals, deep networks suffer from vanishing gradients —
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
            mask: Optional attention mask.

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Self-attention with pre-norm and residual connection
        normed = self.norm1(x)
        attended = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attended)  # Residual connection

        # Feed-forward with pre-norm and residual connection
        normed = self.norm2(x)
        fed_forward = self.feed_forward(normed)
        x = x + self.dropout(fed_forward)  # Residual connection

        return x


class MiniTransformer(nn.Module):
    """A complete Transformer model — built entirely from scratch.

    This is a decoder-only Transformer (like GPT) that can generate text.
    It stacks N TransformerBlocks and adds token/positional embeddings.

    ARCHITECTURE SUMMARY:
      Token IDs → Embedding → Positional Encoding → N × TransformerBlock → LayerNorm → Linear → Logits

    MODEL SIZE CALCULATION:
      For d_model=256, n_heads=4, n_layers=4, vocab_size=10000:
      - Embedding: 10000 × 256 = 2.56M params
      - Per block: ~0.8M params (attention + FFN + norms)
      - Total: ~2.56M + 4 × 0.8M + 0.8M ≈ 6.6M parameters

      For reference:
      - GPT-2 Small: 117M params (d=768, 12 heads, 12 layers)
      - GPT-3: 175B params (d=12288, 96 heads, 96 layers)
      - Our MiniTransformer: ~6.6M (perfect for learning on a laptop!)

    LEARNING NOTE:
      This implementation follows the same architecture as GPT-2/3, just smaller.
      If you understand this code, you understand GPT. The only differences at
      scale are: more layers, wider dimensions, and training data.

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
        >>> logits = model(input_ids)  # (2, 32, 10000) — probability over vocab
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

        # Token embedding: maps token IDs to dense vectors
        # Each of the vocab_size tokens gets its own d_model-dimensional vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: adds position information
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Stack of N Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm (pre-norm architecture)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: maps hidden states to vocabulary logits
        # This is the "language model head" — it predicts the next token
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Weight tying: share weights between token embedding and output projection
        # WHY? The output projection is doing the REVERSE of embedding:
        # embedding: token_id → vector, output: vector → token_id
        # Sharing weights reduces parameters and improves performance.
        # Paper: "Using the Output Embedding to Improve Language Models" (2016)
        self.output_projection.weight = self.token_embedding.weight

        # Initialize weights (Xavier uniform is standard for Transformers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization.

        WHY INITIALIZATION MATTERS:
          Poor initialization can make training impossible:
          - Too large → gradients explode
          - Too small → gradients vanish
          Xavier initialization keeps the variance of activations
          constant across layers, enabling stable training.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask that prevents attending to future tokens.

        WHY CAUSAL MASKING?
          In language modeling, we predict the next token. The model should
          only see PREVIOUS tokens, not future ones. The causal mask sets
          future positions to -infinity, so softmax gives them zero weight.

          For seq_len=4:
            [[1, 0, 0, 0],    token 0 can only see token 0
             [1, 1, 0, 0],    token 1 can see tokens 0-1
             [1, 1, 1, 0],    token 2 can see tokens 0-2
             [1, 1, 1, 1]]    token 3 can see tokens 0-3

        Args:
            seq_len: Length of the sequence.
            device: Device to create the mask on.

        Returns:
            (1, 1, seq_len, seq_len) — causal attention mask.
        """
        # torch.tril creates a lower triangular matrix (upper triangle = 0)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Add batch and head dimensions for broadcasting
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: token IDs → next-token logits.

        Args:
            input_ids: (batch_size, seq_len) — integer token IDs.

        Returns:
            (batch_size, seq_len, vocab_size) — logits for each position.
            The logits at position i predict the token at position i+1.
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Token embedding
        # (batch, seq_len) → (batch, seq_len, d_model)
        x = self.token_embedding(input_ids)

        # Scale embeddings by sqrt(d_model) — this is from the original paper
        # It prevents the embedding values from being too small relative to
        # the positional encoding values
        x = x * math.sqrt(self.d_model)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Create causal mask (autoregressive — can't see future)
        mask = self._create_causal_mask(seq_len, input_ids.device)

        # Step 4: Pass through N Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Step 5: Final layer norm
        x = self.final_norm(x)

        # Step 6: Project to vocabulary logits
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.output_projection(x)

        return logits

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Useful for comparing model sizes and estimating memory requirements.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
          - temperature=1.0: normal sampling (model's default distribution)
          - temperature<1.0: more confident/deterministic (sharper distribution)
          - temperature>1.0: more creative/random (flatter distribution)
          - temperature→0: greedy decoding (always pick most likely)

        TOP-K:
          Only consider the top-k most likely tokens. This prevents the model
          from picking very unlikely tokens. k=50 is a common default.

        Args:
            input_ids: (batch, seq_len) — starting token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering (None = no filtering).

        Returns:
            (batch, seq_len + max_new_tokens) — input + generated tokens.
        """
        for _ in range(max_new_tokens):
            # Get logits from the model
            logits = self(input_ids)

            # Only need the last position's logits
            next_token_logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                # Set all logits outside top-k to -inf
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
