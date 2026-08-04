# Tiny Shakespeare — nano-GPT

## Summary

A from-scratch implementation of a decoder-only Transformer language model trained on the TinyShakespeare dataset.  
The model learns to generate Shakespearean text at the character level, progressively building up every core concept of the GPT architecture — from raw bytes to autoregressive generation.

---

## Description

This project is a step-by-step, pedagogical notebook that constructs a miniature GPT model entirely in PyTorch without relying on any high-level abstractions.  
It follows Andrej Karpathy's *nanoGPT* approach and is intended as a ground-up understanding of how large language models actually work internally.  

### Dataset

**TinyShakespeare** is a ~1 MB plain-text file containing the complete works of Shakespeare concatenated together.  
It is the canonical small dataset for teaching character-level language models because:

- It is small enough to train in minutes on a CPU or Apple Silicon GPU.
- It has rich, recognisable structure (speaker names, dialogue, acts/scenes) that makes it easy to evaluate output quality by eye.

### Architecture

The model is a **decoder-only Transformer** — the same fundamental architecture as GPT-2/3, just much smaller (~1 million parameters).

```text
Input tokens (B, T)
    ↓  Token Embedding  +  Positional Embedding     → (B, T, n_embd)
    ↓  N × Transformer Block
        ├─ LayerNorm  →  Multi-Head Self-Attention  →  residual add
        └─ LayerNorm  →  Feed-Forward Network       →  residual add
    ↓  Final LayerNorm
    ↓  Linear LM Head                               → (B, T, vocab_size)
    ↓  Cross-Entropy Loss / Softmax Sampling
```

### Key Concepts Covered

| Concept                          | Description                                                                                                   |
|----------------------------------|---------------------------------------------------------------------------------------------------------------|
| **Character-level tokenizer**    | Maps each of 65 unique characters to an integer; no sub-word BPE required                                     |
| **Token + Positional Embeddings**| Two learnable lookup tables summed together to encode *what* and *where*                                      |
| **Scaled Dot-Product Attention** | Q·Kᵀ / √d_k with a causal (lower-triangular) mask to prevent future peeking                                   |
| **Multi-Head Attention**         | N parallel attention heads concatenated and projected; each head specialises in different relational patterns |
| **Feed-Forward Network**         | Per-token two-layer MLP with 4× expansion: `n_embd → 4·n_embd → n_embd`                                       |
| **Transformer Block**            | Pre-LayerNorm residual block: `x = x + MHA(LN(x))` then `x = x + FFN(LN(x))`                                  |
| **Autoregressive Generation**    | Iteratively samples the next character from the softmax distribution over the vocabulary                      |
| **AdamW optimiser**              | Adam with weight decay (L2 regularisation on weights, not biases)                                             |

### Hyperparameters

| Parameter       | Value                  |
|-----------------|------------------------|
| `vocab_size`    | 65 (unique characters) |
| `block_size`    | 64 (context window)    |
| `batch_size`    | 64                     |
| `n_embd`        | 128                    |
| `n_head`        | 4                      |
| `n_layer`       | 4                      |
| `learning_rate` | 1e-3                   |
| `max_iters`     | 1 500                  |

### Notebook Structure

| Cell  | Purpose                                                          |
|-------|------------------------------------------------------------------|
| 1     | Imports (`torch`, `urllib`, `os`)                                |
| 2     | Download TinyShakespeare dataset                                 |
| 3     | Build character-level tokenizer (encode / decode)                |
| 4     | Train/val split (90/10) and `get_batch` data loader              |
| 5     | Token + Positional Embedding layer (standalone demo)             |
| 6     | Single-head scaled dot-product self-attention (standalone demo)  |
| 7     | *(Marker)* Complete code                                         |
| 8     | Full model assembly + training loop                              |
| 9     | Unconditional generation (blank prompt)                          |
| 10    | Save model weights to `mini_shakespeare_model.pt`                |
| 11    | Conditional (prompt-based) generation (e.g. `"ROMEO:"`)          |
| 12–13 | Additional generation / analysis cells                           |

### Output

After training, the model generates text that structurally resembles Shakespeare — proper speaker labels, dialogue formatting, and plausible English words —  
even though the content is stochastically sampled.

**Example (unconditional generation after training):**

```text
ROMEO:
What means this, my lord?

JULIET:
The heavens have bound thee here
To speak of what thou hast not seen.
```

### Files

| File                                  | Description                           |
|---------------------------------------|---------------------------------------|
| `nano-gpt/notebook.ipynb`             | Main Jupyter notebook with all cells  |
| `nano-gpt/shakespeare.txt`            | TinyShakespeare training corpus       |
| `nano-gpt/mini_shakespeare_model.pt`  | Saved model weights (state dict)      |

### Device Support

The training loop auto-selects the best available compute device:

- `cuda` — NVIDIA GPU
- `mps` — Apple Silicon GPU
- `cpu` — fallback
