<!-- markdownlint-configure-file {"MD036": false, "MD040": false} -->

# PyTorch ANN — Implementation Guide

A scale-ready, implementation reference for Artificial Neural Networks using PyTorch.  
Covers architecture design, activation functions, loss functions, gradient descent variants, backpropagation, and training — with recommendations for every decision point.  

---

## Table of Contents

01. [Environment Setup](#1-environment-setup)
02. [Data Handling — Any Scale](#2-data-handling--any-scale)
03. [Architecture Design — Layers and Neurons](#3-architecture-design--layers-and-neurons)
04. [Activation Functions](#4-activation-functions)
05. [Loss Functions](#5-loss-functions)
06. [Gradient Descent Variants](#6-gradient-descent-variants)
07. [Optimisers](#7-optimisers)
08. [Backpropagation in PyTorch](#8-backpropagation-in-pytorch)
09. [Training Loop — Full Implementation](#9-training-loop--full-implementation)
10. [Evaluation](#10-evaluation)
11. [GPU Support](#11-gpu-support)
12. [Putting It All Together — Generic Reusable Template](#12-putting-it-all-together--generic-reusable-template)
13. [Recommendations and Decision Guide](#13-recommendations-and-decision-guide)

---

## 1. Environment Setup

### Installation

```bash
# CPU only
pip install torch torchvision

# GPU (CUDA 11.8) — check https://pytorch.org for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Device — use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device    : {device}")
```

---

## 2. Data Handling — Any Scale

For large datasets, never load everything into memory.  
PyTorch's `Dataset` and `DataLoader` handle this via lazy loading and batching.

### 2.1 Custom Dataset Class

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class TabularDataset(Dataset):
    """
    Generic dataset for tabular data.
    Works with in-memory numpy arrays or disk-backed files.

    Args:
        X : numpy array or file path (features)
        y : numpy array or file path (labels)
    """

    def __init__(self, X, y):
        # Accept numpy arrays directly
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        # Accept file paths for large datasets (memory mapped)
        elif isinstance(X, str):
            self.X = torch.tensor(np.load(X, mmap_mode='r'), dtype=torch.float32)
            self.y = torch.tensor(np.load(y, mmap_mode='r'), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Usage — in-memory
X = np.random.randn(100_000, 1000).astype(np.float32)   # 100K samples, 1000 features
y = np.random.randint(0, 2, size=(100_000, 1)).astype(np.float32)

dataset = TabularDataset(X, y)
print(f"Dataset size: {len(dataset)}")
print(f"Sample shape: {dataset[0][0].shape}")
```

### 2.2 Train / Validation / Test Split

```python
from torch.utils.data import random_split

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits dataset into train, validation, and test sets.

    Recommendation:
        70% train / 15% val / 15% test  — general purpose
        80% train / 10% val / 10% test  — when data is scarce
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Ratios must sum to 1.0"

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_set, val_set, test_set


train_set, val_set, test_set = split_dataset(dataset)
print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
```

### 2.3 DataLoader — Batching and Shuffling

```python
def create_dataloaders(train_set, val_set, test_set, batch_size=256, num_workers=4):
    """
    Creates DataLoaders for train, val, and test sets.

    batch_size recommendations:
        32   — small datasets, limited GPU memory
        64   — default safe choice
        128  — medium datasets
        256  — large datasets with sufficient GPU memory
        512+ — very large datasets, monitor GPU memory

    num_workers:
        0    — single process (safe for debugging)
        2-4  — standard for most machines
        8+   — high-core machines with NVMe storage

    Recommendation: always pin_memory=True when using GPU.
    It speeds up CPU→GPU data transfer.
    """
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,               # always shuffle training data
        num_workers=num_workers,
        pin_memory=True             # faster CPU→GPU transfer
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,  # no gradients → larger batch is fine
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = create_dataloaders(
    train_set, val_set, test_set, batch_size=256
)
print(f"Batches per epoch: {len(train_loader)}")
```

### 2.4 Normalisation

```python
class Normaliser:
    """
    Fits on training data only.
    Applies same transformation to val and test.

    CRITICAL: Never fit on val or test — that leaks information.
    """

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0) + 1e-8    # epsilon avoids division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# Usage
X_train_raw = np.random.randn(70_000, 1000).astype(np.float32)
X_val_raw   = np.random.randn(15_000, 1000).astype(np.float32)
X_test_raw  = np.random.randn(15_000, 1000).astype(np.float32)

normaliser  = Normaliser()
X_train     = normaliser.fit_transform(X_train_raw)  # fit on train
X_val       = normaliser.transform(X_val_raw)         # transform only
X_test      = normaliser.transform(X_test_raw)         # transform only
```

---

## 3. Architecture Design — Layers and Neurons

### 3.1 Rules of Thumb

| Factor           | Guidance                                                        |
|------------------|-----------------------------------------------------------------|
| Input neurons    | Fixed — equals number of features                               |
| Output neurons   | Fixed — 1 for binary/regression, N for N-class                  |
| Hidden layers    | Start with 1–2, add only if underfitting                        |
| Hidden neurons   | Between input and output size; powers of 2 (32, 64, 128, 256)   |
| Depth vs Width   | Deeper → more abstraction; Wider → more capacity per layer      |

**Dataset size vs network size:**

| Dataset Size   | Suggested Architecture                |
|----------------|---------------------------------------|
| < 1K samples   | 1 hidden layer, ≤ 64 neurons          |
| 1K – 10K       | 1–2 hidden layers, 64–128 neurons     |
| 10K – 100K     | 2–3 hidden layers, 128–256 neurons    |
| 100K – 1M      | 3–4 hidden layers, 256–512 neurons    |
| 1M+            | 4+ hidden layers, 512–1024+ neurons   |

### 3.2 Automatic Architecture Suggester

```python
def suggest_architecture(n_samples: int,
                          n_features: int,
                          problem_type: str = 'binary') -> dict:
    """
    Suggests a starting architecture based on dataset dimensions.

    Args:
        n_samples    : number of data points
        n_features   : number of input features
        problem_type : 'binary', 'multiclass', 'regression'

    Returns:
        dict with suggested architecture

    Note:
        These are starting points — not optimal solutions.
        Always validate with experiments.
    """
    # Determine depth
    if n_samples < 1_000:
        n_hidden_layers = 1
        base_neurons    = 32
    elif n_samples < 10_000:
        n_hidden_layers = 2
        base_neurons    = 64
    elif n_samples < 100_000:
        n_hidden_layers = 2
        base_neurons    = 128
    elif n_samples < 1_000_000:
        n_hidden_layers = 3
        base_neurons    = 256
    else:
        n_hidden_layers = 4
        base_neurons    = 512

    # Cap neurons to avoid excessive parameters
    max_neurons = min(base_neurons, n_features * 2)
    max_neurons = max(max_neurons, 16)   # floor at 16

    # Funnel: decrease neurons each layer
    hidden_sizes = []
    size = max_neurons
    for _ in range(n_hidden_layers):
        hidden_sizes.append(size)
        size = max(size // 2, 16)

    # Output
    if problem_type == 'binary':
        output_size       = 1
        output_activation = 'sigmoid'
        loss              = 'BCELoss'
    elif problem_type == 'multiclass':
        output_size       = 'N (number of classes)'
        output_activation = 'softmax'
        loss              = 'CrossEntropyLoss'
    else:   # regression
        output_size       = 1
        output_activation = 'none'
        loss              = 'MSELoss'

    return {
        'input_size'        : n_features,
        'hidden_sizes'      : hidden_sizes,
        'output_size'       : output_size,
        'hidden_activation' : 'ReLU',
        'output_activation' : output_activation,
        'suggested_loss'    : loss,
        'suggested_lr'      : 0.001,
        'suggested_batch'   : min(256, n_samples // 10),
        'suggested_epochs'  : 100
    }


# Example
arch = suggest_architecture(n_samples=100_000, n_features=1000, problem_type='binary')
for k, v in arch.items():
    print(f"  {k:<25}: {v}")
```

### 3.3 Generic ANN Model

```python
import torch
import torch.nn as nn

class ANN(nn.Module):
    """
    Generic fully connected ANN.
    Supports any number of layers, neurons, and activation functions.

    Args:
        input_size        : number of input features
        hidden_sizes      : list of neuron counts per hidden layer e.g. [256, 128, 64]
        output_size       : number of output neurons
        hidden_activation : activation for hidden layers (see Section 4)
        output_activation : activation for output layer
        dropout_rate      : dropout probability (0 = disabled)
                            Recommendation: 0.2–0.5 for regularisation
        use_batch_norm    : apply batch normalisation after each hidden layer
                            Recommendation: True for deep networks (3+ layers)
    """

    ACTIVATIONS = {
        'relu'      : nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        'elu'       : nn.ELU(),
        'tanh'      : nn.Tanh(),
        'sigmoid'   : nn.Sigmoid(),
        'selu'      : nn.SELU(),
        'gelu'      : nn.GELU(),
        'none'      : nn.Identity()
    }

    def __init__(self,
                 input_size        : int,
                 hidden_sizes      : list,
                 output_size       : int,
                 hidden_activation : str   = 'relu',
                 output_activation : str   = 'sigmoid',
                 dropout_rate      : float = 0.0,
                 use_batch_norm    : bool  = False):
        super(ANN, self).__init__()

        assert hidden_activation in self.ACTIVATIONS, \
            f"Unknown activation: {hidden_activation}. Choose from {list(self.ACTIVATIONS)}"
        assert output_activation in self.ACTIVATIONS, \
            f"Unknown activation: {output_activation}. Choose from {list(self.ACTIVATIONS)}"

        layers = []
        prev_size = input_size

        # Hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(self.ACTIVATIONS[hidden_activation])
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(self.ACTIVATIONS[output_activation])

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """
        He initialisation for ReLU-based networks.
        Xavier initialisation for Tanh/Sigmoid-based networks.

        Recommendation:
            Use He   initialisation with ReLU, LeakyReLU, ELU
            Use Xavier initialisation with Tanh, Sigmoid
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage
model = ANN(
    input_size        = 1000,
    hidden_sizes      = [256, 128, 64],
    output_size       = 1,
    hidden_activation = 'relu',
    output_activation = 'sigmoid',
    dropout_rate      = 0.3,
    use_batch_norm    = True
)
print(model)
print(f"\nTotal trainable parameters: {model.count_parameters():,}")
```

---

## 4. Activation Functions

### When to Use What

| Activation      | Use In                         | Avoid When        | Notes                                    |
|-----------------|--------------------------------|-------------------|------------------------------------------|
| ReLU            | Hidden layers (default)        | Output layer      | Fast, simple, works well generally       |
| Leaky ReLU      | Hidden layers                  | Output layer      | Better than ReLU when neurons die        |
| ELU             | Hidden layers                  | Output layer      | Smooth, handles negative values          |
| GELU            | Hidden layers (Transformers)   | Simple networks   | Used in BERT, GPT                        |
| Tanh            | Hidden layers (RNNs)           | Deep networks     | Saturates — causes vanishing gradients   |
| Sigmoid         | Binary output layer only       | Hidden layers     | Saturates badly in hidden layers         |
| Softmax         | Multi-class output layer       | Anywhere else     | Converts scores to probabilities         |
| None/Identity   | Regression output              | Classification    | Raw continuous output                    |

### Dead Neuron Problem (ReLU)

ReLU outputs zero for all negative inputs. If a neuron consistently receives negative inputs, its gradient becomes permanently zero — it stops learning. This is called a **dead neuron**.

**Solution:** Use Leaky ReLU or ELU if you observe many dead neurons.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def plot_activations():
    z = torch.linspace(-4, 4, 200)

    activations = {
        'ReLU'       : nn.ReLU(),
        'Leaky ReLU' : nn.LeakyReLU(0.1),
        'ELU'        : nn.ELU(),
        'Tanh'       : nn.Tanh(),
        'Sigmoid'    : nn.Sigmoid(),
        'GELU'       : nn.GELU(),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for i, (name, fn) in enumerate(activations.items()):
        with torch.no_grad():
            out = fn(z)
        axes[i].plot(z.numpy(), out.numpy(), linewidth=2)
        axes[i].set_title(name)
        axes[i].axhline(0, color='gray', linewidth=0.5)
        axes[i].axvline(0, color='gray', linewidth=0.5)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('activations.png', dpi=150)
    plt.show()

plot_activations()
```

### Manual Activation Functions (for reference)

```python
import torch

def relu(z):
    return torch.clamp(z, min=0)

def leaky_relu(z, alpha=0.01):
    return torch.where(z > 0, z, alpha * z)

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def tanh(z):
    return (torch.exp(z) - torch.exp(-z)) / (torch.exp(z) + torch.exp(-z))

def softmax(z, dim=-1):
    exp_z = torch.exp(z - z.max(dim=dim, keepdim=True).values)  # numerical stability
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

# Test
z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"ReLU    : {relu(z)}")
print(f"Sigmoid : {sigmoid(z)}")
print(f"Tanh    : {tanh(z).round(decimals=3)}")
```

---

## 5. Loss Functions

### Selection Guide

| Problem                       | Loss Function                      | Output Activation                            |
|-------------------------------|------------------------------------|----------------------------------------------|
| Binary classification         | `BCELoss` or `BCEWithLogitsLoss`   | Sigmoid (or none for logits)                 |
| Multi-class classification    | `CrossEntropyLoss`                 | None (logits — softmax applied internally)   |
| Regression                    | `MSELoss`                          | None                                         |
| Regression (outlier robust)   | `L1Loss` (MAE)                     | None                                         |
| Regression (balanced)         | `HuberLoss`                        | None                                         |

**Recommendation:** Prefer `BCEWithLogitsLoss` over `BCELoss`. It combines sigmoid and BCE in one numerically stable operation.

### Implementation

```python
import torch
import torch.nn as nn

# ── Binary Classification ──────────────────────────────────────────────────────

# Option 1: BCELoss — requires sigmoid applied beforehand
criterion_bce = nn.BCELoss()
y_pred = torch.sigmoid(torch.tensor([1.5, -0.5, 2.0]))
y_true = torch.tensor([1.0, 0.0, 1.0])
loss = criterion_bce(y_pred, y_true)
print(f"BCELoss: {loss.item():.4f}")

# Option 2: BCEWithLogitsLoss — preferred, numerically stable
# Pass raw logits (no sigmoid needed beforehand)
criterion_bce_logits = nn.BCEWithLogitsLoss()
y_logits = torch.tensor([1.5, -0.5, 2.0])   # raw network output
loss = criterion_bce_logits(y_logits, y_true)
print(f"BCEWithLogitsLoss: {loss.item():.4f}")

# ── Multi-class Classification ─────────────────────────────────────────────────

# CrossEntropyLoss — expects raw logits, NOT softmax output
# Applies log_softmax internally
criterion_ce = nn.CrossEntropyLoss()
y_logits_mc  = torch.tensor([[1.2, 0.5, -0.3],
                               [0.1, 2.1,  0.4]])   # batch of 2, 3 classes
y_true_mc    = torch.tensor([0, 1])                  # class indices
loss = criterion_ce(y_logits_mc, y_true_mc)
print(f"CrossEntropyLoss: {loss.item():.4f}")

# ── Regression ─────────────────────────────────────────────────────────────────

criterion_mse   = nn.MSELoss()     # sensitive to outliers
criterion_mae   = nn.L1Loss()      # robust to outliers
criterion_huber = nn.HuberLoss()   # balanced — MSE for small errors, MAE for large

y_pred_reg = torch.tensor([2.5, 0.0, 2.0, 8.0])
y_true_reg = torch.tensor([3.0, -0.5, 2.0, 7.0])

print(f"MSE  : {criterion_mse(y_pred_reg, y_true_reg).item():.4f}")
print(f"MAE  : {criterion_mae(y_pred_reg, y_true_reg).item():.4f}")
print(f"Huber: {criterion_huber(y_pred_reg, y_true_reg).item():.4f}")


# ── Class Imbalance — Weighted BCE ─────────────────────────────────────────────
# When one class dominates the dataset (e.g. 95% class 0, 5% class 1)

pos_weight = torch.tensor([19.0])   # ~ratio of negatives to positives
criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## 6. Gradient Descent Variants

### Conceptual Difference

```text
Batch GD      : one weight update per epoch (uses all data at once)
SGD           : one weight update per sample (noisy but frequent)
Mini-Batch GD : one weight update per batch  (balance of both) ← standard
```

### Implementation — All Three Variants

```python
import torch
import torch.nn as nn
import numpy as np

def train_batch_gd(model, X, y, criterion, lr=0.01, epochs=100):
    """
    Batch Gradient Descent.
    Uses entire dataset for each weight update.

    Use when:
        - Dataset fits entirely in memory
        - Stable convergence is more important than speed
        - Dataset is small (< 10K samples)

    Avoid when:
        - Dataset is large — one step per epoch is too slow
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_t)
        loss   = criterion(y_pred, y_t)

        model.zero_grad()    # clear previous gradients
        loss.backward()      # compute gradients

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad    # manual SGD step

        if epoch % 10 == 0:
            print(f"[Batch GD]  Epoch {epoch:4d} | Loss: {loss.item():.6f}")


def train_sgd(model, X, y, criterion, lr=0.01, epochs=10):
    """
    Stochastic Gradient Descent.
    One weight update per sample.

    Use when:
        - You want frequent updates and fast early learning
        - Dataset is too large to batch
        - Escaping local minima is important

    Avoid when:
        - Loss needs to converge smoothly
        - Noisy updates cause instability
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n   = len(X)

    for epoch in range(epochs):
        model.train()
        indices    = np.random.permutation(n)
        total_loss = 0.0

        for idx in indices:
            xi = X_t[idx].unsqueeze(0)
            yi = y_t[idx].unsqueeze(0)

            y_pred = model(xi)
            loss   = criterion(y_pred, yi)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad

            total_loss += loss.item()

        avg_loss = total_loss / n
        if epoch % 2 == 0:
            print(f"[SGD]       Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f}")


def train_mini_batch(model, train_loader, criterion, optimiser, epochs=50, device='cpu'):
    """
    Mini-Batch Gradient Descent — the standard approach.
    One weight update per batch.

    Use when:
        - Default choice for any dataset size
        - GPU training (GPUs are optimised for batch operations)
        - You want balance between stability and speed

    Batch size recommendations:
        32  — small datasets or limited memory
        64  — safe default
        128 — medium datasets
        256 — large datasets
        512 — very large datasets with high memory GPU
    """
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss   = 0.0
        total_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)
            loss   = criterion(y_pred, y_batch)

            # Backward pass
            optimiser.zero_grad()   # CRITICAL: clear gradients before each step
            loss.backward()          # compute gradients via backprop
            optimiser.step()         # update weights

            total_loss    += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        if epoch % 10 == 0:
            print(f"[Mini-Batch] Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f}")
```

---

## 7. Optimisers

### When to Use What

| Optimiser        | Use When                                  | Learning Rate   | Notes                                   |
|------------------|-------------------------------------------|-----------------|-----------------------------------------|
| SGD              | Simple problems, good LR scheduler used   | 0.01–0.1        | Generalises well with momentum          |
| SGD + Momentum   | Most problems requiring SGD               | 0.01–0.1        | Standard improvement over plain SGD     |
| RMSProp          | RNNs, non-stationary problems             | 0.001           | Adaptive per-weight LR                  |
| Adam             | Default for most problems                 | 0.001           | Fast convergence, widely used           |
| AdamW            | When weight decay matters                 | 0.001           | Adam with proper regularisation         |
| LBFGS            | Small datasets, full-batch only           | 1.0             | Second-order, very fast on small data   |

**Recommendation:** Start with **AdamW lr=0.001**. Switch to SGD with momentum + scheduler if you need better generalisation (Adam sometimes overfits more).

```python
import torch.optim as optim

# model must be defined first
model = ANN(input_size=1000, hidden_sizes=[256, 128], output_size=1)

# ── Adam ───────────────────────────────────────────────────────────────────────
optimiser_adam = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),     # momentum terms — rarely need to change
    eps=1e-8,
    weight_decay=0          # L2 regularisation — use AdamW instead for this
)

# ── AdamW (recommended default) ────────────────────────────────────────────────
optimiser_adamw = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4       # L2 penalty — prevents overfitting
)

# ── SGD with Momentum ──────────────────────────────────────────────────────────
optimiser_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,           # standard value
    weight_decay=1e-4,
    nesterov=True           # lookahead momentum — slightly better than standard
)

# ── RMSProp ────────────────────────────────────────────────────────────────────
optimiser_rms = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99,
    weight_decay=1e-4
)

# ── Learning Rate Schedulers ───────────────────────────────────────────────────

# Step decay: reduce LR by factor every N epochs
scheduler_step = optim.lr_scheduler.StepLR(
    optimiser_adamw, step_size=20, gamma=0.5
)

# Reduce on plateau: reduce LR when val loss stops improving
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimiser_adamw,
    mode='min',
    factor=0.5,
    patience=5,             # wait 5 epochs before reducing
    min_lr=1e-6
)

# Cosine annealing: smoothly decay LR following cosine curve
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimiser_adamw,
    T_max=100,              # number of epochs
    eta_min=1e-6
)
```

---

## 8. Backpropagation in PyTorch

PyTorch uses **automatic differentiation** (autograd) to compute gradients.
Every operation on a tensor is tracked in a **computation graph**.
Calling `.backward()` traverses this graph in reverse and computes gradients.

### How It Works

```python
import torch

# Simple example — manual trace of what PyTorch does internally
x = torch.tensor([0.5, 0.3, 0.8], requires_grad=False)
w = torch.tensor([0.4, 0.3, 0.2], requires_grad=True)   # weights — track gradient
b = torch.tensor(0.1,              requires_grad=True)   # bias

# Forward pass
z    = torch.dot(w, x) + b
a    = torch.sigmoid(z)
loss = -(torch.tensor(1.0) * torch.log(a))   # BCE, y=1

print(f"z    : {z.item():.4f}")
print(f"a    : {a.item():.4f}")
print(f"loss : {loss.item():.4f}")

# Backward pass — computes all gradients automatically
loss.backward()

print(f"\nGradient of loss w.r.t. w : {w.grad}")
print(f"Gradient of loss w.r.t. b : {b.grad.item():.4f}")

# These are the exact values backprop would compute manually
```

### Critical Rules for Backpropagation in Training

```python
# RULE 1: Always call zero_grad() before backward()
# Gradients accumulate by default — not clearing them corrupts the update
optimiser.zero_grad()    # ← MUST be called every iteration
loss.backward()
optimiser.step()

# RULE 2: Use torch.no_grad() during inference
# Disables gradient tracking — saves memory and speeds up computation
model.eval()
with torch.no_grad():
    predictions = model(X_val)

# RULE 3: model.train() vs model.eval()
model.train()   # enables dropout and batch norm in training mode
model.eval()    # disables dropout, uses running stats for batch norm

# RULE 4: Gradient clipping — prevents exploding gradients in deep networks
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Call this AFTER loss.backward() and BEFORE optimiser.step()
```

### Gradient Flow Inspection

```python
def check_gradients(model):
    """
    Inspect gradient magnitudes per layer.
    Use this to detect vanishing or exploding gradients.

    Healthy gradient range: 1e-4 to 1e-1
    Too small (< 1e-6)    : vanishing gradient — early layers not learning
    Too large (> 10)      : exploding gradient — use gradient clipping
    """
    print("\nGradient magnitudes:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            status = "OK"
            if grad_norm < 1e-6:
                status = "VANISHING"
            elif grad_norm > 10:
                status = "EXPLODING"
            print(f"  {name:<40} | norm: {grad_norm:.2e}  [{status}]")
```

---

## 9. Training Loop — Full Implementation

```python
import torch
import torch.nn as nn
import time

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Args:
        patience  : epochs to wait before stopping
        min_delta : minimum improvement to count as progress
        mode      : 'min' for loss, 'max' for accuracy

    Recommendation:
        patience=10 for small datasets
        patience=5  for large datasets (expensive epochs)
    """

    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, score, model, path='best_model.pt'):
        if self.best_score is None:
            self.best_score = score
            self._save(model, path)
        elif self._is_improvement(score):
            self.best_score = score
            self.counter    = 0
            self._save(model, path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def _save(self, model, path):
        torch.save(model.state_dict(), path)


def train_epoch(model, train_loader, criterion, optimiser, device,
                clip_grad_norm=None):
    """Single training epoch."""
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Forward pass
        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)

        # Backward pass
        optimiser.zero_grad()
        loss.backward()

        # Gradient clipping (optional but recommended for deep networks)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimiser.step()

        # Accumulate metrics
        total_loss    += loss.item() * len(X_batch)
        total_samples += len(X_batch)

        # Binary accuracy
        preds          = (y_pred > 0.5).float()
        total_correct += (preds == y_batch).sum().item()

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on val or test set. No gradient computation."""
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)

        total_loss    += loss.item() * len(X_batch)
        total_samples += len(X_batch)

        preds          = (y_pred > 0.5).float()
        total_correct += (preds == y_batch).sum().item()

    return total_loss / total_samples, total_correct / total_samples


def train(model,
          train_loader,
          val_loader,
          criterion,
          optimiser,
          scheduler       = None,
          epochs          = 100,
          device          = 'cpu',
          early_stopping  = None,
          clip_grad_norm  = 1.0,
          log_every       = 10):
    """
    Full training loop with validation, early stopping,
    and learning rate scheduling.

    Returns:
        history : dict with train/val loss and accuracy per epoch
    """
    model.to(device)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss'  : [], 'val_acc'  : [],
        'lr'        : []
    }

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>6}")
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimiser, device, clip_grad_norm
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        current_lr = optimiser.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        elapsed = time.time() - t0

        if epoch % log_every == 0 or epoch == 1:
            print(f"{epoch:>6} | {train_loss:>10.6f} | {train_acc:>8.2%} | "
                  f"{val_loss:>8.6f} | {val_acc:>6.2%} | "
                  f"{current_lr:>8.2e} | {elapsed:>5.1f}s")

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.stop:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                print(f"Best val loss: {early_stopping.best_score:.6f}")
                # Restore best weights
                model.load_state_dict(torch.load('best_model.pt'))
                break

    return history
```

---

## 10. Evaluation

```python
import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Collect all predictions and true labels from a DataLoader."""
    model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        probs   = model(X_batch).cpu()
        preds   = (probs > 0.5).float()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(y_batch)

    return (torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy(),
            torch.cat(all_probs).numpy())


def evaluate_binary_classifier(model, test_loader, device):
    """
    Full evaluation report for binary classification.
    Reports accuracy, precision, recall, F1, ROC-AUC.
    """
    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)

    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))

    print("CONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred))

    auc = roc_auc_score(y_true, y_prob)
    print(f"\nROC-AUC Score: {auc:.4f}")
    print("=" * 50)

    return y_true, y_pred, y_prob
```

---

## 11. GPU Support

```python
import torch
import torch.nn as nn

# ── Device Setup ───────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)    # wraps model for multi-GPU

model = model.to(device)

# ── Mixed Precision Training (FP16) ───────────────────────────────────────────
# Halves memory usage, roughly doubles throughput on modern GPUs.
# Recommendation: use on any GPU with Tensor Cores (RTX, V100, A100)

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

def train_epoch_amp(model, train_loader, criterion, optimiser, device):
    """Training epoch with automatic mixed precision."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad()

        # Forward pass in FP16
        with autocast():
            y_pred = model(X_batch)
            loss   = criterion(y_pred, y_batch)

        # Backward pass — scaler handles FP16 gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ── Memory Management ──────────────────────────────────────────────────────────

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory — Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# Clear cache when running out of memory
torch.cuda.empty_cache()

# ── Model Save and Load ────────────────────────────────────────────────────────

def save_checkpoint(model, optimiser, epoch, loss, path='checkpoint.pt'):
    torch.save({
        'epoch'              : epoch,
        'model_state_dict'   : model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'loss'               : loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimiser, path='checkpoint.pt'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    epoch = checkpoint['epoch']
    loss  = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch}, loss {loss:.4f}")
    return epoch, loss
```

---

## 12. Putting It All Together — Generic Reusable Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG = {
    # Data
    'n_samples'          : 100_000,
    'n_features'         : 1_000,
    'problem_type'       : 'binary',       # 'binary' | 'multiclass' | 'regression'

    # Architecture
    'hidden_sizes'       : [512, 256, 128],
    'hidden_activation'  : 'relu',         # 'relu' | 'leaky_relu' | 'elu' | 'tanh' | 'gelu'
    'output_activation'  : 'sigmoid',      # 'sigmoid' | 'none' (regression/multiclass)
    'dropout_rate'       : 0.3,
    'use_batch_norm'     : True,

    # Training
    'batch_size'         : 256,
    'epochs'             : 100,
    'learning_rate'      : 0.001,
    'weight_decay'       : 1e-4,
    'clip_grad_norm'     : 1.0,

    # Early stopping
    'early_stopping'     : True,
    'patience'           : 10,

    # Misc
    'seed'               : 42,
    'num_workers'        : 4,
    'log_every'          : 10,
}

# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ── Device ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Synthetic Data (replace with your dataset) ─────────────────────────────────

X = np.random.randn(CONFIG['n_samples'], CONFIG['n_features']).astype(np.float32)
y = np.random.randint(0, 2, (CONFIG['n_samples'], 1)).astype(np.float32)

# Normalise
normaliser = Normaliser()
X = normaliser.fit_transform(X)

# Dataset and loaders
dataset                           = TabularDataset(X, y)
train_set, val_set, test_set      = split_dataset(dataset)
train_loader, val_loader, \
    test_loader                   = create_dataloaders(
                                        train_set, val_set, test_set,
                                        batch_size  = CONFIG['batch_size'],
                                        num_workers = CONFIG['num_workers']
                                    )

# ── Model ──────────────────────────────────────────────────────────────────────

model = ANN(
    input_size        = CONFIG['n_features'],
    hidden_sizes      = CONFIG['hidden_sizes'],
    output_size       = 1,
    hidden_activation = CONFIG['hidden_activation'],
    output_activation = CONFIG['output_activation'],
    dropout_rate      = CONFIG['dropout_rate'],
    use_batch_norm    = CONFIG['use_batch_norm']
).to(device)

print(f"Parameters: {model.count_parameters():,}")

# ── Loss and Optimiser ─────────────────────────────────────────────────────────

criterion = nn.BCELoss()

optimiser = optim.AdamW(
    model.parameters(),
    lr           = CONFIG['learning_rate'],
    weight_decay = CONFIG['weight_decay']
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5, patience=5
)

early_stopping = EarlyStopping(
    patience=CONFIG['patience']
) if CONFIG['early_stopping'] else None

# ── Train ──────────────────────────────────────────────────────────────────────

history = train(
    model          = model,
    train_loader   = train_loader,
    val_loader     = val_loader,
    criterion      = criterion,
    optimiser      = optimiser,
    scheduler      = scheduler,
    epochs         = CONFIG['epochs'],
    device         = device,
    early_stopping = early_stopping,
    clip_grad_norm = CONFIG['clip_grad_norm'],
    log_every      = CONFIG['log_every']
)

# ── Evaluate ───────────────────────────────────────────────────────────────────

y_true, y_pred, y_prob = evaluate_binary_classifier(model, test_loader, device)

# ── Save ───────────────────────────────────────────────────────────────────────

save_checkpoint(model, optimiser, epoch=CONFIG['epochs'],
                loss=min(history['val_loss']), path='final_model.pt')
```

---

## 13. Recommendations and Decision Guide

### Architecture

| Decision                | Recommendation                                             |
|-------------------------|------------------------------------------------------------|
| Starting architecture   | 1–2 hidden layers, neurons between input and output size   |
| Neuron count            | Power of 2 (32, 64, 128, 256, 512)                         |
| Depth                   | Add layers only when underfitting persists                 |
| Width vs Depth          | Wider first, then deeper                                   |
| Batch normalisation     | Use for 3+ hidden layers always                            |
| Dropout                 | 0.2–0.5 on hidden layers, never on output                  |

### Activation Functions

| Decision                | Recommendation                                       |
|-------------------------|------------------------------------------------------|
| Hidden layers           | ReLU (default)                                       |
| Dead neurons observed   | Switch to Leaky ReLU or ELU                          |
| Binary output           | Sigmoid                                              |
| Multi-class output      | None (CrossEntropyLoss applies softmax internally)   |
| Regression output       | None (Identity)                                      |

### Training

| Decision                | Recommendation                                            |
|-------------------------|-----------------------------------------------------------|
| Gradient descent type   | Mini-batch (always)                                       |
| Batch size              | 64–256 depending on GPU memory                            |
| Optimiser               | AdamW (default), SGD+Momentum for better generalisation   |
| Learning rate           | 0.001 for Adam/AdamW, 0.01 for SGD                        |
| LR scheduler            | ReduceLROnPlateau for general use                         |
| Gradient clipping       | max_norm=1.0 for deep networks                            |
| Early stopping          | patience=10, monitor val loss                             |
| Weight initialisation   | He (ReLU networks), Xavier (Tanh/Sigmoid)                 |

### Regularisation (Overfitting Prevention)

| Technique                  | When to Use                                 |
|----------------------------|---------------------------------------------|
| Dropout (0.3–0.5)          | When val loss >> train loss                 |
| Weight decay (1e-4)        | Default — always include                    |
| Batch normalisation        | Deep networks                               |
| Reduce model size          | When regularisation alone is insufficient   |
| More data / augmentation   | Best long-term solution                     |

### Common Issues and Fixes

| Symptom                               | Likely Cause                      | Fix                                        |
|---------------------------------------|-----------------------------------|--------------------------------------------|
| Loss not decreasing                   | LR too small or too large         | Try 10× up or down                         |
| Loss exploding                        | LR too large or no clipping       | Reduce LR, add gradient clipping           |
| Val loss >> train loss                | Overfitting                       | Add dropout, weight decay, reduce model    |
| Val loss = train loss and both high   | Underfitting                      | Deeper/wider network, more epochs          |
| Dead neurons                          | ReLU with bad init                | Switch to Leaky ReLU, use He init          |
| NaN loss                              | Exploding gradients or bad data   | Gradient clip, check input normalisation   |
| Slow training                         | CPU bottleneck                    | More num_workers, pin_memory=True          |
| Out of GPU memory                     | Batch too large                   | Halve batch size, use mixed precision      |
