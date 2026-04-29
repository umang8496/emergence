# The LLM Pipeline

## From Text to Tokens to Thought: How LLMs Process and Generate Language.  

### Table of Content

- [The LLM Pipeline](#the-llm-pipeline)
  - [Table of Content](#table-of-content)
  - [\[Phase 1\] Input Processing](#phase-1-input-processing)
    - [Vocabulary](#vocabulary)
    - [Tokenization](#tokenization)
    - [Vectors](#vectors)
      - [How are vectors built?](#how-are-vectors-built)
      - [What is Meaning?](#what-is-meaning)
    - [Embeddings Matrix](#embeddings-matrix)
    - [What Does "Model Size" Actually Mean?](#what-does-model-size-actually-mean)
    - [How Vector Dimensionality Depends on Model Size?](#how-vector-dimensionality-depends-on-model-size)
  - [\[Phase 2\] Transformer Core](#phase-2-transformer-core)
    - [Let's understand phase 2 with an example](#lets-understand-phase-2-with-an-example)
      - [Create Q, K and V](#create-q-k-and-v)
        - [ATTENTION SCORE](#attention-score)
        - [SOFTMAX → WEIGHTS](#softmax--weights)
        - [WEIGHTED SUM](#weighted-sum)
        - [FFN TRANSFORMATION](#ffn-transformation)
        - [RESIDUAL CONNECTION](#residual-connection)
        - [Repeat for Every Token](#repeat-for-every-token)
      - [Final Output After Phase 2](#final-output-after-phase-2)
        - [Clean Mental Model](#clean-mental-model)
  - [\[Phase 3\] Output Generation](#phase-3-output-generation)

---

## [Phase 1] Input Processing

- Different models have different tokenizers and they tokenize the text differently.
- Tokens are compression units, not linguistic units.
- Tokens are not same as words, because language is messy and humans refuse to cooperate.
- The same text can generate different tokens which would have different IDs and hence have different embeddings.
- Every model has a fixed vocabulary,around 30k to 100k tokens.
- Once the model is trained and is frozen the vocabulary becomes final and immutable.
- Token IDs are meaningless, these are just an index for a token, nothing more.
- Meaning only appears after `ID → embedding lookup → vector`.
- Because of subword tokenization, unknown or unseen words are no longer a challenge or problem.
- New text must be broken into existing tokens, once the training is done for the model.
- Each token id would correspond to a vector through the embedding lookup process.
- These vectors (list of numbers) go into the transformers.

```text
Converting messy human language into structured numerical input that a transformer can digest.  
No intelligence yet.  
Just encoding reality into numbers.  
```

### Vocabulary

Vocabulary is a set of all tokens a model understands.  
Vocabulary defines the input/output space of the model.  
The model can only, **read tokens from Vocabulary** and **generate tokens from Vocabulary**.  

### Tokenization

This is basically, `Convert text → tokens from vocabulary`.  
For example:  
`"playing football"` upon tokenization becomes `["play", "ing", " football"]`.  

We do not use words for tokens, because:

- infinite words exist
- new words appear
- storage explodes

So we use subword units.  

Tokens often include spaces: `" love" ≠ "love"`.  
Yes, it's weird. Yes, it matters.

### Vectors

A vector is a list of numbers, for example `v = [0.12, -0.98, 0.33, 1.45, ...]`.  
Dimension could be 64, 128, 768, 4096 depending on model size.  
They're not random (after training).  
They are **learned numerical representations of tokens**.  
Meaning:

- similar words → similar vectors
- different meanings → different directions in space

#### How are vectors built?

Inside the model, we have an embedding matrix.

```text
Embedding Matrix E
Shape: (vocab_size × embedding_dim)
```

```text
E = [
  [0.1, 0.2, ...],   ← token 0
  [0.5, -0.3, ...],  ← token 1
  ...
]
```

After that, for each token `Token ID → Vector`.  

```text
vector = E[token_id]
```

No computation. Just raw **row lookup**.

During training:

- model predicts next token
- computes loss
- backprop updates embedding matrix too

So embeddings evolve like:

```text
"king" → vector close to "queen"
"dog"  → vector close to "cat"
```

These vectors represent **statistical relationships in language**.  
These vectors do not represent **dictionary definitions** and **human-readable meaning**.  

For Example, If in data, we have:

```text
"I love AI"
"I love music"
"I love coding"
```

Then, "AI", "music", "coding" vectors become similar (because they appear in similar contexts).  

#### What is Meaning?

```text
Meaning = how a word behaves, not what it “means”
```

Each vector encodes multiple dimensions like:

- syntactic role (noun, verb)
- semantic similarity
- usage patterns
- context compatibility

But NOT explicitly. There is no: `dimension 42 = “is a verb”`.  
It's all distributed.

### Embeddings Matrix

Shape: `vocab_size × d_model`  

Lets say: `vocab_size = 50,000` and `d_model = 128`, then `E = (50000 × 128)`.  

Parameter Count Contribution: `embedding_params = vocab_size × d_model`  
So, `50,000 × 128 = 6.4 million parameters`  

Embedding matrix is often a huge chunk of total parameters.  

If we strip everything down:

An LLM is just a giant system that learns how to move vectors around in a very structured way.

### What Does "Model Size" Actually Mean?

> Model size = total number of trainable parameters

That includes:

- embedding matrix
- attention weights (Q, K, V, projections)
- feed-forward layers
- output projection

A **1M parameter model** means:

- All weights combined ≈ 1,000,000 numbers
- Each of those gets updated during training.

`More parameters → more capacity to learn patterns`  
`Less parameters → simpler patterns only`

So:

- 1M params → toy model
- 7B params → serious model
- 100B+ → "we spent millions training this"

### How Vector Dimensionality Depends on Model Size?

Embedding Dimension `(d_model)` is the size of each token vector.  
Bigger models usually have:

- larger d_model
- more layers
- more heads

Because, more dimensions = more expressive representation.  
Think of vector dimensions as `number of features the model can use to describe a token`.  

| Model     | d_model | Params   |
| --------- | ------- | -------- |
| Tiny      | 64      | ~1M      |
| Small     | 256     | ~10M     |
| GPT-scale | 4096+   | billions |

Increasing `d_model`:

- increases embedding size
- increases attention computation
- increases FFN size

So model size grows fast.

---

## [Phase 2] Transformer Core

- So far, we have converted text into number for better processing. No intelligence is there.
- This phase answers how the model understand the relationships between the tokens.
- We have `vec = [v1, v2, v3, v4, ...]` after the phase 1.
- Each token is **independent** and **unaware of others**.
- Transformer's job is to **Make each token aware of every other token**. That is the whole game.
- The problem with embeddings is that they don't know the order or relationships among the tokens.
- To the model `"I love AI" == "AI love I"`, which is not ideal or correct.
- So here each token gets mapped to the `token_embedding` + `position_embedding`, which is `meaning + position`.
- Position is not extra info, rather it is the only way the model knows sequence order exists.
- From here on, the problem to be addressed is "Which other tokens should I care about, and how much?"
- Formally, for each input `x`, the system computes:

    ```text
    Q = xWq   (Query)
    K = xWk   (Key)
    V = xWv   (Value)
    ```

    Here `Q` stands for **what this token is looking for**, `K` **what each token offers** and `V` **actual information to pass**.
- Then some `weight` is computed using these values. This gets done through a process called `Self-Attention`.
- The same self-attention process is executed parallely which enables the each token to understand semantics, relationship, and dependency.
- This process provides **multiple perspectives on the same sentence**.
- After the attention process, **FFN** (Feed Forward Network) comes into the picture, it transforms the result computed so far.
- Then a series of such layers get added for better output generation, namely Residual Connections, Layer Normalization.
- This same process is applied multiple time 5 or 6, where each layer reshapes the meaning and sharpens the relationships.
- Transformer iteratively refines the token representations using context, and does not try to understand the language itself.

### Let's understand phase 2 with an example

Prompt: `"I love AI"`.  
Token IDs: `[I → 101, love → 2057, AI → 999]`.  
Embedding Lookup:

```text
v_I    = E[101]
v_love = E[2057]
v_AI   = E[999]
```

Now: `X = [v_I, v_love, v_AI]`. These are context-free vectors.  

#### Create Q, K and V

Each vector is transformed:

```text
Q = XWq
K = XWk
V = XWv
```

Now each token has: a query version, a key version, a value version.  
Each token now asks:

- **What am I looking for?** (Q)
- **What do others offer?** (K)
- **What info do I give?** (V)

##### ATTENTION SCORE

For token `"love"`:

```text
score(love, I)
score(love, love)
score(love, AI)
```

This gives: `[0.8, 0.2, 0.9]`.  

##### SOFTMAX → WEIGHTS

`[0.4, 0.1, 0.5]`

##### WEIGHTED SUM

`v_love' = 0.4*v_I + 0.1*v_love + 0.5*v_AI`

##### FFN TRANSFORMATION

`v_love'' = FFN(v_love')`. This:

- reshapes information
- introduces non-linearity

##### RESIDUAL CONNECTION

`v_love_final = v_love + v_love''` So:

- original meaning preserved
- context added

##### Repeat for Every Token

Repeat the same process for all the tokens in the input.

#### Final Output After Phase 2

```text
[
  v_I_final,
  v_love_final,
  v_AI_final
]
```

Each vector now encodes:

- its original meaning
- its role in the sentence
- its relationship with others

```text
At no point does the model “understand” words like humans.  
It does vector transformations that preserve and reshape statistical relationships.  
```

##### Clean Mental Model

| Stage           | What Vector Represents |
| --------------- | ---------------------- |
| After embedding | isolated meaning       |
| After attention | context-aware meaning  |
| After FFN       | refined representation |

---

## [Phase 3] Output Generation

---
