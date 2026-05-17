<!-- markdownlint-configure-file {"MD036": false, "MD040": false} -->

# Overfitting, Underfitting, Regularisation, and Batch Normalisation

## A Comprehensive Guide with Mathematics and Python

---

## Table of Contents

01. [The Core Problem — Generalisation](#1-the-core-problem--generalisation)
02. [Overfitting](#2-overfitting)
03. [Underfitting](#3-underfitting)
04. [Bias-Variance Tradeoff](#4-bias-variance-tradeoff)
05. [Train, Validation, and Test Split](#5-train-validation-and-test-split)
06. [Learning Curves — Diagnosing the Problem](#6-learning-curves--diagnosing-the-problem)
07. [L2 Regularisation — Weight Decay](#7-l2-regularisation--weight-decay)
08. [L1 Regularisation — Lasso](#8-l1-regularisation--lasso)
09. [L1 vs L2 — Key Differences](#9-l1-vs-l2--key-differences)
10. [Dropout](#10-dropout)
11. [Batch Normalisation](#11-batch-normalisation)
12. [Early Stopping](#12-early-stopping)
13. [Combining Techniques — Decision Guide](#13-combining-techniques--decision-guide)
14. [Full PyTorch Implementation](#14-full-pytorch-implementation)

---

## 1. The Core Problem — Generalisation

A model is trained on a finite dataset. The real objective is never to perform well on that training data — it is to perform well on **new, unseen data** that the model has never encountered.

This ability is called **generalisation.**

The gap between training performance and real-world performance is the central problem in machine learning. Every concept in this document exists to close that gap.

```text
What we want:
    Low training loss   AND   Low validation loss

What can go wrong:
    High training loss  AND   High validation loss  →  Underfitting
    Low training loss   AND   High validation loss  →  Overfitting
```

---

## 2. Overfitting

### Definition

Overfitting occurs when a model learns the training data **too well** — including its noise, outliers, and random fluctuations — instead of learning the underlying pattern.

The model performs excellently on training data and poorly on new data.

### Analogy

A student memorises every question and answer from last year's exam paper word for word. They score perfectly on a mock test using the same paper. On the real exam with new questions, they fail — because they memorised, not learned.

### How It Looks

```
Training loss   →  very low   (e.g. 0.01)
Validation loss →  much higher (e.g. 0.45)
```

The gap between training and validation loss is the signal.

### Why It Happens

- Model has too many parameters relative to the amount of training data
- Model trained for too many epochs
- Insufficient regularisation
- Training data is not representative

### Visual Representation

```
True pattern:   y = 0.5x + 2

Underfit model: y = 3           (horizontal line — ignores data)
Good model:     y = 0.5x + 2   (captures the pattern)
Overfit model:  y = x⁸ - 3x⁷ + 2x⁶ - ...  (passes through every point exactly)
```

---

## 3. Underfitting

### Definition

Underfitting occurs when a model is **too simple** to capture the underlying pattern in the data. It performs poorly on both training and validation data.

### Analogy

A student barely studied and learned only one rule — "all answers are C." They perform badly on both the mock test and the real exam.

### How It Looks

```
Training loss   →  high  (e.g. 0.45)
Validation loss →  high  (e.g. 0.47)
```

Both are high. No significant gap between them.

### Why It Happens

- Model has too few parameters (too shallow or too narrow)
- Trained for too few epochs
- Learning rate too high — training never converged
- Over-regularisation — regularisation is too aggressive

---

## 4. Bias-Variance Tradeoff

This is the theoretical framework that explains both overfitting and underfitting.

### Definitions

**Bias** — error introduced by approximating a complex real-world problem with a simplified model. A high-bias model makes strong, often wrong, assumptions about the data.

```
High Bias → model too simple → underfitting
```

**Variance** — error introduced by the model's sensitivity to small fluctuations in the training data. A high-variance model reacts to noise as if it were a real signal.

```
High Variance → model too complex → overfitting
```

### The Decomposition

For any model, the expected prediction error on unseen data can be decomposed as:

```
Expected Error = Bias² + Variance + Irreducible Noise
```

Where:

- **Bias²** — systematic error from wrong assumptions
- **Variance** — error from sensitivity to training data
- **Irreducible Noise** — inherent randomness in the data. Cannot be eliminated regardless of model quality.

### Mathematics

Let `f(x)` be the true function and `f̂(x)` be the model's prediction.

```
Bias     = E[f̂(x)] - f(x)
           (expected prediction minus true value)

Variance = E[(f̂(x) - E[f̂(x)])²]
           (spread of predictions around their own mean)

MSE      = Bias² + Variance + σ²
           where σ² is the irreducible noise variance
```

### The Tradeoff

You cannot minimise both simultaneously with a fixed amount of data.

```
Error
  │
  │         Total Error
  │        /────────────
  │       /          ___/──   Variance
  │      /        __/
  │     /      __/
  │  __/______/              Bias²
  │ /
  │__________________________________ Model Complexity

  ↑                    ↑
 Underfit           Overfit
             ↑
          Sweet spot
```

Increasing model complexity reduces bias but increases variance. The optimal model minimises the **sum** of both.

### Python — Demonstrating Bias-Variance

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(42)

# True function: y = sin(x) + noise
def true_function(x):
    return np.sin(x)

n_samples    = 30
X            = np.sort(np.random.uniform(0, 2 * np.pi, n_samples))
y            = true_function(X) + np.random.normal(0, 0.3, n_samples)
X_plot       = np.linspace(0, 2 * np.pi, 300)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees   = [1, 4, 15]
titles    = ['Underfit (degree=1)', 'Good Fit (degree=4)', 'Overfit (degree=15)']

for ax, degree, title in zip(axes, degrees, titles):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('lr',   LinearRegression())
    ])
    model.fit(X.reshape(-1, 1), y)
    y_plot = model.predict(X_plot.reshape(-1, 1))

    ax.scatter(X, y, color='black', s=20, label='Data')
    ax.plot(X_plot, true_function(X_plot), 'g--', label='True function')
    ax.plot(X_plot, y_plot, 'r-', label=f'Model (deg={degree})')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('bias_variance.png', dpi=150)
plt.show()
```

---

## 5. Train, Validation, and Test Split

### Why Three Sets?

| Set          | Purpose                     | Used For                                       |
|--------------|-----------------------------|------------------------------------------------|
| Training     | Model learns weights        | Forward pass, backprop, weight updates         |
| Validation   | Monitor generalisation      | Detect overfitting, tune hyperparameters       |
| Test         | Final unbiased evaluation   | Report final performance — used exactly once   |

**Critical rule:** The test set must never influence any training decision. If you use the test set to tune hyperparameters, it becomes a second validation set — and your reported performance is optimistic and unreliable.

### Recommended Split Ratios

| Dataset Size   | Train   | Validation   | Test   |
|----------------|---------|--------------|--------|
| < 10K          | 60%     | 20%          | 20%    |
| 10K – 100K     | 70%     | 15%          | 15%    |
| 100K – 1M      | 80%     | 10%          | 10%    |
| 1M+            | 90%     | 5%           | 5%     |

### Python Implementation

```python
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_dataset(dataset, ratios=(0.70, 0.15, 0.15), seed=42):
    """
    Splits dataset into train, val, test.

    Args:
        dataset : PyTorch Dataset
        ratios  : (train, val, test) — must sum to 1.0
        seed    : for reproducibility
    """
    assert abs(sum(ratios) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    n       = len(dataset)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


# Example
X       = np.random.randn(10_000, 50).astype(np.float32)
y       = np.random.randint(0, 2, (10_000, 1)).astype(np.float32)
dataset = TabularDataset(X, y)

train_set, val_set, test_set = split_dataset(dataset)
print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False)
```

### Cross-Validation (for Small Datasets)

When data is scarce, a single train/val split is unreliable — the split itself might be lucky or unlucky. K-Fold Cross-Validation gives a more robust estimate.

```python
from sklearn.model_selection import KFold

def k_fold_split(X, y, n_splits=5, seed=42):
    """
    K-Fold cross-validation split.

    Use when:
        - Dataset < 5000 samples
        - You need a reliable estimate of generalisation

    Each fold uses (k-1)/k of data for training and 1/k for validation.
    Final performance = average across all k folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        yield fold, X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# Usage
for fold, X_train, X_val, y_train, y_val in k_fold_split(X, y, n_splits=5):
    print(f"Fold {fold+1}: Train={len(X_train)}, Val={len(X_val)}")
    # Train and evaluate model on each fold
```

---

## 6. Learning Curves — Diagnosing the Problem

Learning curves plot training and validation loss (and accuracy) across epochs. They are the primary diagnostic tool.

### Patterns and Their Meaning

**Healthy training:**

```
Loss
  │  train ──\
  │           \──────────────  } small gap = generalising
  │  val   ────────────────────
  │
  └──────────────────────────── epochs
```

**Overfitting:**

```
Loss
  │           /─────────────── val loss rising
  │          /
  │  val ───/
  │
  │  train ──\──────────────── train loss falling
  │
  └──────────────────────────── epochs
               ↑
           divergence point
```

**Underfitting:**

```
Loss
  │  train ──────────────────  both high
  │  val   ──────────────────  and flat
  │
  └──────────────────────────── epochs
```

**Over-regularisation (too much penalty):**

```
Loss
  │  train ──────────────────  both high
  │  val   ──────────────────  but close together
  │
  └──────────────────────────── epochs
  (looks like underfitting — check regularisation strength)
```

### Python — Plotting Learning Curves

```python
import matplotlib.pyplot as plt

def plot_learning_curves(history: dict, save_path: str = None):
    """
    Plots training and validation loss and accuracy.

    Args:
        history   : dict with keys train_loss, val_loss,
                    train_acc, val_acc
        save_path : optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'],   label='Val Loss',   linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(history['val_acc'],   label='Val Acc',   linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# Example
history = {
    'train_loss': [0.9, 0.7, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02],
    'val_loss'  : [0.9, 0.72, 0.55, 0.40, 0.38, 0.42, 0.50, 0.61],
    'train_acc' : [0.55, 0.65, 0.75, 0.85, 0.92, 0.96, 0.98, 0.99],
    'val_acc'   : [0.55, 0.64, 0.73, 0.79, 0.80, 0.79, 0.77, 0.74],
}
plot_learning_curves(history)
```

---

## 7. L2 Regularisation — Weight Decay

### The Idea

Add the sum of squared weights as a penalty term to the loss function. This discourages large weights — which are associated with overfitting.

### Mathematics

Standard loss:

```
L = Loss(ŷ, y)
```

L2 regularised loss:

```
L_reg = Loss(ŷ, y) + λ/2 × Σ wᵢ²
```

Where:

- `λ` (lambda) is the regularisation strength — a hyperparameter
- `Σ wᵢ²` is the sum of squares of all weights in the network
- `1/2` is a convenience factor that cancels with the 2 from the derivative

### Effect on Gradient

Without L2, the gradient for weight `w` is:

```
∂L/∂w = ∂Loss/∂w
```

With L2, the gradient becomes:

```
∂L_reg/∂w = ∂Loss/∂w + λ × w
```

The weight update rule:

```
w_new = w_old - η × (∂Loss/∂w + λ × w)
      = w_old - η × ∂Loss/∂w - η × λ × w
      = w_old × (1 - η × λ) - η × ∂Loss/∂w
```

The term `(1 - η × λ)` is the decay factor. Every step, weights are multiplied by a number slightly less than 1 — they shrink towards zero. Hence the name **weight decay**.

### Lambda Selection

| λ Value        | Effect                                          |
|----------------|-------------------------------------------------|
| 0              | No regularisation                               |
| 1e-5 to 1e-4   | Light regularisation (default starting point)   |
| 1e-3           | Moderate                                        |
| 1e-2           | Strong — may cause underfitting                 |

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ── Method 1: Via optimiser weight_decay (recommended) ────────────────────────
# PyTorch's AdamW and SGD apply weight decay correctly and efficiently

optimiser = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4       # L2 penalty λ = 0.0001
)

# ── Method 2: Manual L2 penalty in loss (for explicit control) ────────────────

def l2_loss(model, lambda_l2=1e-4):
    """
    Computes L2 penalty: λ/2 × Σ wᵢ²
    Excludes biases — regularising biases is rarely beneficial.
    """
    l2_penalty = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name:                        # exclude biases
            l2_penalty += torch.sum(param ** 2)
    return (lambda_l2 / 2) * l2_penalty


# In training loop:
# loss = criterion(y_pred, y_true) + l2_loss(model, lambda_l2=1e-4)


# ── Note on Adam vs AdamW ──────────────────────────────────────────────────────
# torch.optim.Adam with weight_decay does NOT apply true L2 regularisation.
# It conflates weight decay with gradient scaling — the two are not equivalent.
# torch.optim.AdamW fixes this. Always use AdamW when weight decay is needed.

optimiser_correct   = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
optimiser_incorrect = optim.Adam(model.parameters(),  lr=0.001, weight_decay=1e-4)
# Use optimiser_correct
```

---

## 8. L1 Regularisation — Lasso

### The Idea

Add the sum of absolute values of weights as a penalty. Unlike L2, L1 drives many weights to **exactly zero** — effectively removing those connections.

### Mathematics

L1 regularised loss:

```
L_reg = Loss(ŷ, y) + λ × Σ |wᵢ|
```

Effect on gradient:

```
∂L_reg/∂w = ∂Loss/∂w + λ × sign(w)
```

Where:

```
sign(w) =  1   if w > 0
          -1   if w < 0
           0   if w = 0
```

The penalty is constant in magnitude regardless of the weight's size.  
This causes small weights to be driven to exactly zero — unlike L2 which just makes them small.  

### Why L1 Produces Sparsity — Geometric Intuition

```
L2 constraint region: circle (all weights penalised proportionally)
L1 constraint region: diamond (corners lie on axes)

The optimal solution (minimum loss + constraint) is more likely to
touch the diamond at a corner — where one or more weights = 0.
```

### Python Implementation

```python
def l1_loss(model, lambda_l1=1e-4):
    """
    Computes L1 penalty: λ × Σ |wᵢ|
    Excludes biases.
    """
    l1_penalty = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_penalty += torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty


def combined_l1_l2_loss(model, lambda_l1=1e-5, lambda_l2=1e-4):
    """
    Elastic Net — combines L1 and L2.
    Gets sparsity from L1 and stability from L2.
    """
    l1 = l1_loss(model, lambda_l1)
    l2 = l2_loss(model, lambda_l2)
    return l1 + l2


# In training loop:
# loss = criterion(y_pred, y_true) + combined_l1_l2_loss(model)
```

---

## 9. L1 vs L2 — Key Differences

| Property              | L1 (Lasso)                         | L2 (Ridge / Weight Decay)                  |       |               |
|-----------------------|------------------------------------|--------------------------------------------|-------|---------------|
| Penalty term          | λ × Σ\                             | wᵢ\                                        |       | λ/2 × Σ wᵢ²   |
| Effect on weights     | Drives many to exactly zero        | Drives all towards zero but rarely exact   |       |               |
| Result                | Sparse model (feature selection)   | Dense but small weights                    |       |               |
| Gradient              | Constant magnitude ± λ             | Proportional to weight value λw            |       |               |
| Differentiability     | Not at zero                        | Everywhere                                 |       |               |
| Use in DL             | Rare                               | Standard (via weight_decay)                |       |               |
| Use in classical ML   | Lasso regression                   | Ridge regression                           |       |               |

### When to Use Which

```
Use L2 (weight decay):
    → Default choice for all neural networks
    → Stable, differentiable everywhere
    → Works well with Adam and SGD

Use L1:
    → High-dimensional input with many irrelevant features
    → Interpretability matters (zero weights = feature removed)
    → Classical ML pipelines (Lasso regression)

Use Both (Elastic Net):
    → When you want sparsity but also stability
    → When L1 alone is too aggressive
```

---

## 10. Dropout

### The Idea

During each forward pass of training, randomly set a fraction of neuron outputs to zero. This prevents neurons from co-adapting — the network cannot rely on any specific neuron and must develop redundant, robust representations.

### Mathematics

For a hidden layer with output vector `a = [a₁, a₂, ..., aₙ]`:

**During training:**

```
mask = Bernoulli(1 - p)    ← binary vector, each element 1 with prob (1-p)
a_dropped = a ⊙ mask       ← element-wise multiplication
a_scaled  = a_dropped / (1-p)   ← scale to maintain expected value
```

Where:

- `p` is the dropout probability (fraction of neurons dropped)
- `⊙` is element-wise multiplication
- Scaling by `1/(1-p)` keeps the expected sum of activations unchanged

**During inference:**

```
a_out = a    ← all neurons active, no masking, no scaling
```

PyTorch implements this **inverted dropout** — scaling happens during training, inference is unchanged.

### Why Scaling Matters

Without scaling, if you drop 50% of neurons during training but use all neurons during inference, the expected activation magnitude doubles. This would cause the model to behave differently at test time.

By scaling up during training, inference output matches training output in expectation.

### Effect on the Network

Each training step uses a different random subnetwork:

```
Step 1:  neurons [1, 2, 4, 6] active   (3, 5 dropped)
Step 2:  neurons [1, 3, 5, 6] active   (2, 4 dropped)
Step 3:  neurons [2, 3, 4, 5] active   (1, 6 dropped)
```

The network must learn useful representations without relying on any specific neuron. This acts as an ensemble of many thinned networks, averaged at inference time.

### Dropout Rate Recommendations

| Rate        | Use When                                    |
|-------------|---------------------------------------------|
| 0.1 – 0.2   | Mild regularisation needed, large dataset   |
| 0.3 – 0.4   | General purpose default                     |
| 0.5         | Severe overfitting, smaller dataset         |
| > 0.5       | Rarely — kills too much network capacity    |

### Where to Apply

```
Input layer    → rarely (loses raw input information)
Hidden layers  → yes — after activation function
Output layer   → never
```

### Python Implementation

```python
import torch
import torch.nn as nn

# ── Built-in Dropout ───────────────────────────────────────────────────────────

class ANNWithDropout(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super().__init__()
        layers = []
        prev   = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))   # after activation
            prev = size

        layers.append(nn.Linear(prev, output_size))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ── Manual Dropout (for understanding) ────────────────────────────────────────

def manual_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    Manual implementation of inverted dropout.

    Args:
        x        : input tensor
        p        : dropout probability
        training : True during training, False during inference
    """
    if not training or p == 0.0:
        return x                                    # no dropout during eval

    mask  = torch.bernoulli(torch.full_like(x, 1 - p))   # 1 with prob (1-p)
    return x * mask / (1 - p)                              # drop and scale


# ── Critical: train() vs eval() ───────────────────────────────────────────────

model = ANNWithDropout(100, [64, 32], 1, dropout_rate=0.3)

model.train()                               # dropout ACTIVE
output_train = model(torch.randn(1, 100))

model.eval()                                # dropout DISABLED
with torch.no_grad():
    output_eval = model(torch.randn(1, 100))

# Always call model.eval() before inference
# Always call model.train() before training resumes
```

---

## 11. Batch Normalisation

### The Problem — Internal Covariate Shift

During training, as weights in earlier layers update, the distribution of inputs to later layers keeps changing. Each layer must continuously adapt to a moving input distribution — this slows training and makes deep networks unstable.

This phenomenon is called **Internal Covariate Shift**.

### The Solution

After each linear transformation and before the activation function, normalise the output of that layer across the current mini-batch — then apply learnable scale and shift parameters.

### Mathematics

For a mini-batch of size `m`, and a layer with pre-activation values `z = [z₁, z₂, ..., zₘ]`:

**Step 1 — Compute batch statistics:**

```
μ_B  = (1/m) × Σ zᵢ               ← batch mean
σ²_B = (1/m) × Σ (zᵢ - μ_B)²     ← batch variance
```

**Step 2 — Normalise:**

```
ẑᵢ = (zᵢ - μ_B) / √(σ²_B + ε)
```

Where `ε` (epsilon) is a small constant (e.g. 1e-5) to prevent division by zero.

**Step 3 — Scale and shift with learnable parameters:**

```
yᵢ = γ × ẑᵢ + β
```

Where:

- `γ` (gamma) — learnable scale parameter, initialised to 1
- `β` (beta)  — learnable shift parameter, initialised to 0

`γ` and `β` allow the network to undo the normalisation if optimal — ensuring Batch Norm does not restrict the network's representational capacity.

### During Inference

During training: `μ_B` and `σ²_B` are computed per batch.

During inference: a **running mean** and **running variance** accumulated during training are used:

```
running_mean = momentum × running_mean + (1 - momentum) × μ_B
running_var  = momentum × running_var  + (1 - momentum) × σ²_B
```

Default momentum in PyTorch = 0.1. The running statistics stabilise as training progresses.

This is why `model.eval()` must be called during inference — it switches Batch Norm from batch statistics to running statistics.

### Where to Place Batch Norm

**Standard order (original paper):**

```
Linear → BatchNorm → Activation
```

**Alternative (common in practice):**

```
Linear → Activation → BatchNorm
```

Both work. The original order is more commonly used.

### Benefits

- Allows higher learning rates — training is faster
- Reduces sensitivity to weight initialisation
- Acts as mild regularisation — reduces need for dropout slightly
- Makes very deep networks (10+ layers) trainable

### Limitations

- Requires batch size > 1 (does not work with batch_size=1)
- Adds two parameters per feature (γ and β)
- Behaviour differs between training and inference — requires careful model.train()/model.eval() management

### Python Implementation

```python
import torch
import torch.nn as nn

# ── Built-in BatchNorm ─────────────────────────────────────────────────────────

class ANNWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev   = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.BatchNorm1d(size))         # after linear, before activation
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev = size

        layers.append(nn.Linear(prev, output_size))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ── Manual Batch Norm (for understanding) ─────────────────────────────────────

class ManualBatchNorm(nn.Module):
    """
    Manual implementation of 1D Batch Normalisation.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps      = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))    # scale
        self.beta  = nn.Parameter(torch.zeros(num_features))   # shift

        # Running statistics (not learned, not gradient-tracked)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Compute batch statistics
            mu  = x.mean(dim=0)                         # shape: (num_features,)
            var = x.var(dim=0, unbiased=False)           # shape: (num_features,)

            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mu
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            )
        else:
            # Use accumulated running statistics during inference
            mu  = self.running_mean
            var = self.running_var

        # Normalise
        x_norm = (x - mu) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta


# ── Inspect Batch Norm Parameters ─────────────────────────────────────────────

def inspect_batchnorm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            print(f"\nLayer: {name}")
            print(f"  gamma (scale) mean : {module.weight.data.mean():.4f}")
            print(f"  beta  (shift) mean : {module.bias.data.mean():.4f}")
            print(f"  running mean  mean : {module.running_mean.mean():.4f}")
            print(f"  running var   mean : {module.running_var.mean():.4f}")
```

---

## 12. Early Stopping

### The Idea

Stop training when the validation loss stops improving. Restore the weights from the best epoch.

### Why It Works

The network overfits gradually. There is a specific epoch where it transitions from learning to memorising. Early stopping finds and returns to that transition point.

### The Algorithm

```
best_val_loss = infinity
patience_counter = 0

For each epoch:
    train for one epoch
    compute val_loss

    if val_loss < best_val_loss - min_delta:
        best_val_loss    = val_loss
        patience_counter = 0
        save model weights          ← checkpoint the best model

    else:
        patience_counter += 1
        if patience_counter >= patience:
            stop training
            restore saved weights   ← return to best model
```

### Patience Selection

| Scenario                         | Patience                      |
|----------------------------------|-------------------------------|
| Small dataset, fast epochs       | 15 – 20                       |
| Medium dataset                   | 10                            |
| Large dataset, slow epochs       | 5                             |
| Learning rate scheduler in use   | Match to scheduler patience   |

### Python Implementation

```python
import torch
import numpy as np

class EarlyStopping:
    """
    Monitors validation loss and stops training when improvement stalls.

    Args:
        patience  : epochs to wait after last improvement
        min_delta : minimum change to count as an improvement
        mode      : 'min' for loss (lower is better)
                    'max' for accuracy (higher is better)
        path      : path to save the best model weights
    """

    def __init__(self,
                 patience  : int   = 10,
                 min_delta : float = 1e-4,
                 mode      : str   = 'min',
                 path      : str   = 'best_model.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.path       = path
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, score: float, model: torch.nn.Module):
        if self.best_score is None:
            self.best_score = score
            self._save(model)
            return

        if self._is_improvement(score):
            self.best_score = score
            self.counter    = 0
            self._save(model)
        else:
            self.counter += 1
            print(f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                print(f"  EarlyStopping: triggered. Best score: {self.best_score:.6f}")
                self.stop = True

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def _save(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.path)

    def restore(self, model: torch.nn.Module):
        """Load best weights back into model."""
        model.load_state_dict(torch.load(self.path))
        print(f"  Restored best model from {self.path}")


# Usage in training loop
early_stopping = EarlyStopping(patience=10, min_delta=1e-4, mode='min')

for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_loss   = evaluate(...)

    early_stopping(val_loss, model)

    if early_stopping.stop:
        break

early_stopping.restore(model)   # restore best weights
```

---

## 13. Combining Techniques — Decision Guide

### Diagnosis → Treatment

| Symptom                                   | Diagnosis                   | Treatment                                   |
|-------------------------------------------|-----------------------------|---------------------------------------------|
| Train loss high, val loss high, similar   | Underfitting                | More neurons/layers, more epochs, lower λ   |
| Train loss low, val loss much higher      | Overfitting                 | L2, dropout, early stopping, more data      |
| Training unstable, loss fluctuating       | Unstable gradients          | Batch norm, lower LR, gradient clipping     |
| Loss not decreasing                       | LR too low or too high      | Try 10× in either direction                 |
| Val loss > train loss from epoch 1        | Data leakage or bad split   | Check preprocessing pipeline                |
| Both losses plateau early                 | LR too high                 | Reduce LR or use scheduler                  |

### What to Add and When

```
Start with:
    AdamW optimiser (weight_decay=1e-4)    ← always
    Early stopping (patience=10)           ← always
    Train/Val/Test split                   ← always

If overfitting persists:
    Add Dropout (0.3)                      ← first addition
    Increase weight_decay to 1e-3          ← second
    Reduce network size                    ← third

If training is unstable or network is deep (3+ layers):
    Add Batch Normalisation                ← always for deep nets

If dataset is very small (< 5000):
    Use K-Fold cross-validation
    Use heavier dropout (0.4 – 0.5)
    Reduce network size significantly
```

### Interaction Between Techniques

| Combination                     | Notes                                                            |
|---------------------------------|------------------------------------------------------------------|
| Batch Norm + Dropout            | Use both but reduce dropout rate slightly (0.2 instead of 0.3)   |
| Batch Norm + L2                 | Batch Norm reduces need for strong L2. Keep λ small (1e-5)       |
| Early Stopping + LR Scheduler   | Let scheduler reduce LR first; early stopping terminates         |
| L1 + L2 (Elastic Net)           | Good when many features are irrelevant                           |

---

## 14. Full PyTorch Implementation

A complete, production-ready training pipeline incorporating all techniques.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import time


# ── Dataset ────────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Normaliser ─────────────────────────────────────────────────────────────────

class Normaliser:
    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ── Model ──────────────────────────────────────────────────────────────────────

class RegularisedANN(nn.Module):
    """
    ANN with Batch Normalisation and Dropout.

    Args:
        input_size    : number of input features
        hidden_sizes  : list of hidden layer sizes e.g. [256, 128, 64]
        output_size   : 1 for binary/regression, N for multi-class
        dropout_rate  : dropout probability (0 = disabled)
        use_batch_norm: whether to apply batch normalisation
    """

    def __init__(self,
                 input_size    : int,
                 hidden_sizes  : list,
                 output_size   : int,
                 dropout_rate  : float = 0.3,
                 use_batch_norm: bool  = True):
        super().__init__()
        layers = []
        prev   = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev = size

        layers.append(nn.Linear(prev, output_size))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, path='best_model.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter    = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model: nn.Module):
        model.load_state_dict(torch.load(self.path, map_location='cpu'))


# ── Training Functions ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimiser, device, clip_norm=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)

        optimiser.zero_grad()
        loss.backward()
        if clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimiser.step()

        total_loss += loss.item() * len(X_batch)
        correct    += ((y_pred > 0.5) == y_batch).sum().item()
        total      += len(X_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred      = model(X_batch)
        loss        = criterion(y_pred, y_batch)
        total_loss += loss.item() * len(X_batch)
        correct    += ((y_pred > 0.5) == y_batch).sum().item()
        total      += len(X_batch)

    return total_loss / total, correct / total


# ── Full Training Pipeline ─────────────────────────────────────────────────────

def train(model, train_loader, val_loader, config, device):
    criterion     = nn.BCELoss()
    optimiser     = optim.AdamW(
                        model.parameters(),
                        lr           = config['lr'],
                        weight_decay = config['weight_decay']
                    )
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
                        optimiser, mode='min', factor=0.5, patience=5
                    )
    early_stopper = EarlyStopping(
                        patience  = config['patience'],
                        min_delta = 1e-4
                    )

    history = {'train_loss': [], 'val_loss': [],
               'train_acc' : [], 'val_acc' : []}

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimiser, device
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(vl_loss)
        early_stopper(vl_loss, model)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if epoch % config.get('log_every', 10) == 0 or epoch == 1:
            print(f"{epoch:>6} | {tr_loss:>10.6f} | {tr_acc:>8.2%} | "
                  f"{vl_loss:>8.6f} | {vl_acc:>6.2%} | "
                  f"{time.time()-t0:.1f}s")

        if early_stopper.stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    early_stopper.restore(model)
    print(f"Best val loss: {early_stopper.best_score:.6f}")
    return history


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Config
    CONFIG = {
        'n_samples'     : 50_000,
        'n_features'    : 100,
        'hidden_sizes'  : [128, 64, 32],
        'dropout_rate'  : 0.3,
        'batch_norm'    : True,
        'lr'            : 0.001,
        'weight_decay'  : 1e-4,
        'batch_size'    : 128,
        'epochs'        : 200,
        'patience'      : 10,
        'log_every'     : 10,
        'seed'          : 42,
    }

    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    X = np.random.randn(CONFIG['n_samples'], CONFIG['n_features']).astype(np.float32)
    y = np.random.randint(0, 2, (CONFIG['n_samples'], 1)).astype(np.float32)

    normaliser = Normaliser()
    X          = normaliser.fit_transform(X)

    dataset              = TabularDataset(X, y)
    n                    = len(dataset)
    n_train, n_val       = int(n * 0.70), int(n * 0.15)
    n_test               = n - n_train - n_val
    generator            = torch.Generator().manual_seed(CONFIG['seed'])
    train_set, val_set, \
        test_set         = random_split(
                               dataset,
                               [n_train, n_val, n_test],
                               generator=generator
                           )

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'],
                              shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG['batch_size'] * 2,
                              shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG['batch_size'] * 2,
                              shuffle=False, pin_memory=True)

    # Model
    model = RegularisedANN(
        input_size     = CONFIG['n_features'],
        hidden_sizes   = CONFIG['hidden_sizes'],
        output_size    = 1,
        dropout_rate   = CONFIG['dropout_rate'],
        use_batch_norm = CONFIG['batch_norm']
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")

    # Train
    history = train(model, train_loader, val_loader, CONFIG, device)

    # Final test evaluation
    criterion        = nn.BCELoss()
    test_loss, \
        test_acc     = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.6f} | Test Accuracy: {test_acc:.2%}")

    # Plot
    plot_learning_curves(history)
```

---

## Summary

```
Overfitting        → train loss low, val loss high
                     Fix: L2, Dropout, Early Stopping, more data

Underfitting       → both losses high
                     Fix: bigger network, more epochs, less regularisation

Bias-Variance      → Total Error = Bias² + Variance + Noise
                     Goal: minimise both simultaneously

L2 Regularisation  → penalise large weights, shrink towards zero
                     Loss += λ/2 × Σ wᵢ²

L1 Regularisation  → drive irrelevant weights to exactly zero
                     Loss += λ × Σ |wᵢ|

Dropout            → randomly zero out p fraction of neurons each step
                     Scale by 1/(1-p) to maintain expected magnitude

Batch Norm         → normalise layer outputs per batch
                     ẑ = (z - μ) / √(σ² + ε)   then   y = γẑ + β

Early Stopping     → monitor val loss, save best weights, stop on plateau

Default stack      → AdamW (weight_decay=1e-4) + Dropout (0.3)
                     + BatchNorm (deep networks) + EarlyStopping (patience=10)
```
