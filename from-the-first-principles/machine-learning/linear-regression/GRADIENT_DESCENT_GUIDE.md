# Gradient Descent: Complete Guide with All Variants

## What is Gradient Descent?

Gradient Descent is an optimization algorithm that finds the minimum of a cost function by iteratively moving parameters in the direction of the negative gradient.

**Core Concept:**

```text
Imagine standing on a hill in fog. You can't see the bottom, but you know:
- The ground slopes downward in some direction
- If I follow the steepest downward slope, I'll reach the valley
- Repeat until I reach the lowest point

Gradient Descent = this hill descent process, mathematically formalized
```

**The Update Rule:**

```text
w := w - α * ∇J(w)
```

Where:

- `w` = parameters (weights)
- `α` = learning rate (step size)
- `∇J(w)` = gradient (slope of cost function)

---

## The Mathematics of Gradients

### What is a Gradient?

A gradient is the derivative of a function—it tells how much the function changes with respect to each parameter.

**Single variable (1D):**

```text
f(x) = x²
f'(x) = 2x

At x=3: f'(3) = 6 (slope is 6)
At x=1: f'(1) = 2 (slope is 2)
At x=0: f'(0) = 0 (at minimum)
```

**Multiple variables (vector):**

```text
J(w₁, w₂, w₃) = w₁² + 2w₂ + 3w₃²

∇J = [∂J/∂w₁, ∂J/∂w₂, ∂J/∂w₃]
   = [2w₁, 2, 6w₃]
```

The gradient is a vector pointing uphill (direction of steepest ascent). Negative gradient points downhill.

### Gradient for Linear Regression

**Cost function (MSE):**

```text
J(w, b) = (1/2m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²
        = (1/2m) Σᵢ₌₁ᵐ (w·xᵢ + b - yᵢ)²
```

**Gradients:**

```text
∂J/∂w = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ) * xᵢ
∂J/∂b = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)
```

**Intuition:**

- If prediction > actual: positive error, increase gradient
- If prediction < actual: negative error, decrease gradient
- Weighted by feature value (feature importance in gradient)

### Why Negative Gradient?

**If gradient is positive (uphill):**

```text
Parameter := Parameter - (positive value)
Parameter decreases (move downhill) ✓
```

**If gradient is negative (downhill):**

```text
Parameter := Parameter - (negative value)
           = Parameter + (positive value)
Parameter increases (move downhill) ✓
```

The negative sign ensures we always move downhill.

---

## The Learning Rate (α)

The learning rate controls how big each step is.

### Learning Rate Too Small

```python
learning_rate = 0.0001
```

**Effect:**

- Very small steps
- Takes many iterations (slow)
- May be too small to escape local minima
- Converges eventually but inefficient

**Visualization:**

```text
Iteration 0: Cost = 100
Iteration 100: Cost = 99.9
Iteration 200: Cost = 99.8
Iteration 300: Cost = 99.7
...
Iteration 100000: Cost = 0.1 (finally!)
```

### Learning Rate Too Large

```python
learning_rate = 1.0
```

**Effect:**

- Large steps, might overshoot
- Oscillates around minimum (bounces)
- May diverge (move away from minimum)
- Cost function explodes to infinity

**Visualization:**

```text
Iteration 0: Cost = 100
Iteration 1: Cost = 150 (overshot)
Iteration 2: Cost = 200 (worse)
Iteration 3: Cost = 500 (diverging!)
Iteration 4: Cost = inf (exploded)
```

### Learning Rate Just Right

```python
learning_rate = 0.01
```

**Effect:**

- Converges smoothly
- Reasonable number of iterations
- Cost decreases monotonically
- Reaches minimum efficiently

**Visualization:**

```text
Iteration 0: Cost = 100
Iteration 10: Cost = 50
Iteration 20: Cost = 25
Iteration 30: Cost = 12
Iteration 40: Cost = 0.5 (converged)
```

### Adaptive Learning Rates

Some algorithms adjust learning rate during training:

- Start large (fast progress)
- Decrease over time (fine-tuning)
- Example: Learning rate decay

---

## Batch Gradient Descent (Standard GD)

### Algorithm

**Definition:** Use ALL training examples to compute gradient in each iteration.

```text
for iteration in range(num_iterations):
    predictions = X @ w + b              # All m examples
    errors = predictions - y             # All m examples
    gradient = (1/m) * X.T @ errors      # Average over all m
    w := w - learning_rate * gradient
```

### Characteristics

**Pros:**

- Exact gradient (uses all data)
- Smooth convergence (no noise)
- Guaranteed convergence (for convex functions)
- Good for small to medium datasets

**Cons:**

- Computationally expensive for large datasets
- Loads entire dataset in memory
- Slow for very large m (millions of examples)
- Each update requires computing all m predictions

### Convergence Pattern

```text
Cost over iterations (typical):

100 |
    | \
 80 |  \
    |   \
 60 |    \
    |     \
 40 |      \
    |       \
 20 |        \___
    |            \___
  0 |________________\___
    0   10   20   30   40
           Iterations
```

Smooth, monotonic decrease.

### When to Use

- Small datasets (< 10K examples)
- When memory is not a constraint
- When exact gradient is important
- Offline learning scenarios

---

## Stochastic Gradient Descent (SGD)

### Algorithm

**Definition:** Use ONE random training example per iteration.

```text
for iteration in range(num_iterations):
    i = random(0, m)                    # Pick ONE random example
    prediction = X[i] @ w + b           # Single example
    error = prediction - y[i]            # Single error
    gradient = error * X[i]              # Single gradient
    w := w - learning_rate * gradient
```

### Characteristics

**Pros:**

- Very fast per update (uses 1 example)
- Memory efficient (load 1 example at a time)
- Scales to huge datasets
- Natural online learning (can update with new data continuously)
- Noise can help escape local minima

**Cons:**

- Noisy gradient (random variations)
- Convergence path is erratic
- May oscillate around minimum
- No guarantee of monotonic improvement
- Learning rate must decay over time

### Convergence Pattern

```text
Cost over iterations (typical):

100 |  *
    | * *
 80 | * * *
    |* * *  *
 60 | *  *   *
    |  *  *   *
 40 |    * *   *
    |     *  *  *
 20 |      * *   *
    |        *  * *
  0 |________________
    0   10   20   30
           Iterations
```

Noisy, oscillating, but generally trending downward.

### When to Use

- Large datasets (millions of examples)
- Online learning (new data arriving continuously)
- When memory is very limited
- When noise helps (escaping local minima)
- Deep learning (standard choice)

### Why the Noise Helps

In neural networks with multiple local minima:

```text
Cost landscape (non-convex):

    *
   / \     /\
  /   \   /  \
 /     \ /    \
/_______X______\
  local  global
  min    min

SGD bounces around due to noise, might escape local minima.
Batch GD gets stuck.
```

---

## Mini-Batch Gradient Descent

### Algorithm

**Definition:** Use a small batch (e.g., 32) of random examples per iteration.

```text
batch_size = 32
for iteration in range(num_iterations):
    indices = random_sample(0, m, batch_size)
    X_batch = X[indices]                 # 32 examples
    y_batch = y[indices]                 # 32 labels
    predictions = X_batch @ w + b
    errors = predictions - y_batch
    gradient = (1/batch_size) * X_batch.T @ errors
    w := w - learning_rate * gradient
```

### Characteristics

**Pros:**

- Balance between Batch GD and SGD
- Less noisy than SGD (averaged over batch)
- Faster than Batch GD (smaller computations)
- Scales to large datasets
- Efficient GPU utilization
- Industry standard choice

**Cons:**

- Still some noise (smaller than SGD)
- Need to tune batch size
- More complex than pure GD or SGD

### Convergence Pattern

```text
Cost over iterations (typical):

100 |  .
    | . .
 80 | . . .
    |. . . .
 60 | .  .  .
    |  .  .  .
 40 |   . .  .
    |    . .  .
 20 |     . . .
    |      . . .
  0 |____________
    0   10   20   30
           Iterations
```

Smooth-ish with small noise. Between Batch GD and SGD.

### Common Batch Sizes

```text
batch_size = 32    (typical, default for many frameworks)
batch_size = 64    (for GPUs)
batch_size = 128   (larger GPU memory)
batch_size = 256   (very large GPUs)
```

### 6.5 When to Use

- Standard choice for modern ML
- Large datasets
- Neural networks
- When GPU/hardware available
- Default for most frameworks (PyTorch, TensorFlow)

---

## Advanced Optimizers

### Momentum-Based Methods

#### Gradient Descent with Momentum (SGD + Momentum)

**Intuition:** A ball rolling downhill gains momentum, doesn't stop at slight bumps.

**Algorithm:**

```text
v = 0 (velocity/momentum)
for iteration in range(num_iterations):
   gradient = compute_gradient(w)
   v = β * v + (1 - β) * gradient        # Accumulate momentum
   w := w - learning_rate * v
```

Where `β` (beta) ≈ 0.9 (typical value).

**Effect:**

```text
Cost over iterations (with momentum):

100 |  .
    | .  .
 80 | .   .
    |.     .
 60 |       .
    |        .
 40 |         ..
    |           .
 20 |            .
    |             .
  0 |________________
    0   10   20   30
           Iterations
```

Converges faster, fewer oscillations.

**Why it works:**

- Accumulates gradient direction (if consistent, gains speed)
- Smooths out noise
- Helps escape local minima

---

#### Nesterov Accelerated Gradient (NAG)

**Intuition:** Look ahead before updating (like an intelligent ball).

**Algorithm:**

```text
v = 0
for iteration in range(num_iterations):
    gradient = compute_gradient(w - learning_rate * β * v)  # Look ahead
    v = β * v + (1 - β) * gradient
    w := w - learning_rate * v
```

**Advantage over Momentum:**

- Slightly better convergence
- "Looks ahead" before committing to direction

**When to use:**

- Similar use cases as momentum
- Slightly better for some problems

---

### Adaptive Learning Rate Methods

These adjust learning rate per parameter based on history.

#### AdaGrad (Adaptive Gradient)

**Algorithm:**

```text
cache = 0
for iteration in range(num_iterations):
    gradient = compute_gradient(w)
    cache += gradient ** 2                # Accumulate squared gradients
    w := w - (learning_rate / sqrt(cache)) * gradient
```

**Effect:**

- Parameters with large gradients get smaller updates
- Parameters with small gradients get larger updates
- Learning rate decreases over time (automatically)

**Pros:**

- No manual learning rate tuning
- Works well for sparse features

**Cons:**

- Learning rate decreases monotonically (eventually becomes tiny)
- Can stop learning late in training

---

#### RMSprop (Root Mean Square Propagation)

**Algorithm:**

```text
cache = 0
for iteration in range(num_iterations):
    gradient = compute_gradient(w)
    cache = β * cache + (1 - β) * (gradient ** 2)  # Exponential moving avg
    w := w - (learning_rate / sqrt(cache)) * gradient
```

Where `β` ≈ 0.9

**Advantage over AdaGrad:**

- Learning rate doesn't monotonically decrease
- Exponential moving average (recent gradients matter more)
- Remains effective throughout training

**When to use:**

- Neural networks
- When adaptive learning rate helps
- Good default for many problems

---

#### Adam (Adaptive Moment Estimation)

**Algorithm:**

```text
m = 0 (first moment, like momentum)
v = 0 (second moment, like RMSprop)
for iteration in range(num_iterations):
    gradient = compute_gradient(w)
    m = β₁ * m + (1 - β₁) * gradient           # Momentum
    v = β₂ * v + (1 - β₂) * (gradient ** 2)   # RMSprop
    m_corrected = m / (1 - β₁ᵗ)               # Bias correction
    v_corrected = v / (1 - β₂ᵗ)               # Bias correction
    w := w - learning_rate * m_corrected / sqrt(v_corrected + ε)
```

Where:

- `β₁` ≈ 0.9 (momentum parameter)
- `β₂` ≈ 0.999 (RMSprop parameter)
- `ε` ≈ 1e-8 (numerical stability)
- `t` = iteration number

**Characteristics:**

- Combines momentum (m) and adaptive learning (v)
- Includes bias correction (important early in training)
- Often requires less tuning
- Works well "out of the box"

**Pros:**

- Combines benefits of momentum and RMSprop
- Less sensitive to learning rate
- Widely used in deep learning
- Good default choice

**Cons:**

- More hyperparameters (β₁, β₂, ε)
- May overfit in some cases
- More complex (harder to debug)

**When to use:**

- Default for most deep learning
- When you want algorithm to work without tuning
- Neural networks, especially transformers

---

## Comparison Table: All Variants

| Method          | Type         | Speed      | Convergence   | Memory | Noise      | Best For         |
|-----------------|--------------|------------|---------------|--------|------------|------------------|
| Batch GD        | First-order  | Slow       | Smooth        | High   | None       | Small data       |
| SGD             | First-order  | Fast       | Noisy         | Low    | High       | Large data       |
| Mini-batch      | First-order  | Medium     | Smooth-ish    | Medium | Low        | Standard         |
| Momentum        | First-order  | Fast       | Smoother      | Low    | Reduced    | Standard         |
| NAG             | First-order  | Very Fast  | Smooth        | Low    | Reduced    | Standard         |
| AdaGrad         | Adaptive     | Medium     | Smooth        | Medium | Low        | Sparse data      |
| RMSprop         | Adaptive     | Fast       | Smooth        | Medium | Low        | Standard         |
| Adam            | Adaptive     | Fast       | Very Smooth   | Medium | Very Low   | Deep Learning    |
| AdamW           | Adaptive     | Fast       | Very Smooth   | Medium | Very Low   | Deep Learning    |

---

## Decision Tree: Which Optimizer to Use?

```text
START: What problem are I solving?

├─ Small dataset (< 10K examples)?
│  └─ Use: Batch Gradient Descent
│
├─ Large dataset, offline learning?
│  └─ Use: Mini-batch with Adam
│
├─ Online learning (stream of data)?
│  └─ Use: SGD with learning rate decay
│
├─ Deep neural network?
│  ├─ Computer vision?
│  │  └─ Use: Adam (default)
│  │
│  ├─ Transformer / NLP?
│  │  └─ Use: AdamW
│  │
│  └─ Other?
│     └─ Use: Adam
│
├─ Need interpretability / debugging?
│  └─ Use: Mini-batch GD or SGD (simpler)
│
└─ Sparse features (NLP embeddings)?
   └─ Use: AdaGrad or RMSprop
```

---

## Learning Rate Schedules

Learning rate can change during training.

### Constant Learning Rate

```python
lr = 0.001  # Stays same entire training
```

Simplest, works for many problems.

### Step Decay

```python
# Every N iterations, multiply by factor
if iteration % 10 == 0:
    lr *= 0.1
```

Example:

```text
Iterations 0-9:     lr = 0.001
Iterations 10-19:   lr = 0.0001
Iterations 20-29:   lr = 0.00001
```

### Exponential Decay

```python
# Decay continuously
lr = initial_lr * exp(-decay_rate * iteration)
```

Smooth, continuous reduction.

### Cosine Annealing

```python
# Learning rate follows cosine curve (popular in modern deep learning)
import numpy as np
lr = initial_lr * 0.5 * (1 + cos(π * iteration / total_iterations))
```

Decreases smoothly, can have sudden drops (helps escape local minima).

### Warm-up then Decay

```python
# Increase first, then decrease (common in transformers)
if iteration < warmup_steps:
    lr = initial_lr * (iteration / warmup_steps)
else:
    lr = initial_lr * decay_schedule(iteration - warmup_steps)
```

Helps optimization stability early in training.

---

## Convergence Diagnosis

### Loss Decreasing Smoothly?

Good sign. Model learning.

```text
Loss per iteration:
100, 80, 60, 40, 20, 10, 5, 2, 1, 0.5, 0.3...
```

**Continue training**  

### Loss Plateaus?

Model saturated. Options:

- Reduce learning rate
- Change optimizer
- Train more iterations

```text
Loss per iteration:
100, 50, 25, 12, 6, 3, 3, 3, 3, 3, 3...
```

**Investigate**  

### Loss Oscillating?

Learning rate too high.

```text
Loss per iteration:
100, 80, 60, 40, 50, 30, 60, 40, 70, 50...
```

**Reduce learning rate**  

### Loss Diverging?

Learning rate way too high or implementation bug.

```text
Loss per iteration:
100, 150, 200, 300, 500, 1000, inf, nan
```

**Reduce learning rate significantly**  

### Loss Not Changing?

Learning rate too low or stuck.

```text
Loss per iteration:
100, 100, 100, 100, 100, 100...
```

**Increase learning rate**  

---

## Summary: Quick Reference

| Scenario                  | Recommended       | Reason                          |
|---------------------------|-------------------|---------------------------------|
| Quick prototype           | Adam              | Works out of the box            |
| Maximum accuracy          | SGD + momentum    | Better generalization           |
| Deep learning             | Adam              | Industry standard               |
| Sparse features           | AdaGrad/RMSprop   | Handles sparsity                |
| Large batch (distributed) | LAMB              | Scales to many GPUs             |
| Want simplicity           | Mini-batch GD     | Straightforward                 |
| Online learning           | SGD               | Can update with one example     |

---

## Key Takeaways

1. **Gradient Descent is hill descent** — Move opposite to slope
2. **Learning rate critical** — Too small (slow), too large (diverge)
3. **Batch size matters** — Batch GD exact, SGD noisy, mini-batch balanced
4. **Momentum helps** — Accumulates gradient direction
5. **Adaptive methods adjust per-parameter** — Adam is popular default
6. **No one-size-fits-all** — Choose based on problem
7. **Decay learning rate** — Helps fine-tuning late in training

---
