# Cost Functions for Logistic Regression: Complete Guide

## Introduction

In logistic regression, we predict probabilities (values between 0 and 1). Our cost function must be designed to work with these probabilities, not arbitrary numerical values. This is different from linear regression where we used Mean Squared Error.

---

## Why Linear Regression Cost Functions Fail

**Linear regression uses:** Mean Squared Error (MSE)

```text
MSE = (1/m) * Σ(h - y)²
```

Why doesn't this work well for classification?

1. **Output mismatch:** MSE expects continuous values, but we have probabilities
2. **Penalty structure:** MSE doesn't penalize confident wrong predictions as heavily
3. **Convergence:** Gradient descent converges slowly with sigmoid + MSE

Example:

```text
If y=1 (actual positive) and h=0.01 (predicted almost negative):
  MSE loss = (0.01 - 1)² = 0.9801 (moderate penalty)
  But we confidently predicted wrong! Should have large penalty.
```

---

## Cost Function 1: Binary Cross-Entropy (Log Loss)

### Definition

```text
J = -(1/m) * Σ[y * log(h) + (1-y) * log(1-h)]
```

Where:

- `m` = number of training examples
- `y` = actual label (0 or 1)
- `h` = predicted probability (output of sigmoid)
- `log` = natural logarithm

### How It Works

When `y = 1` (positive class):

```text
Cost = -log(h)
```

- If h = 0.9: cost = -log(0.9) ≈ 0.105 (small, good)
- If h = 0.5: cost = -log(0.5) ≈ 0.693 (medium)
- If h = 0.1: cost = -log(0.1) ≈ 2.303 (large, bad)

When `y = 0` (negative class):

```text
Cost = -log(1 - h)
```

- If h = 0.9: cost = -log(0.1) ≈ 2.303 (large, bad)
- If h = 0.5: cost = -log(0.5) ≈ 0.693 (medium)
- If h = 0.1: cost = -log(0.1) ≈ 0.105 (small, good)

### Code Implementation

```python
def binary_cross_entropy(y_actual, y_predicted, m):
    """
    Binary Cross-Entropy cost function
    Args:
        y_actual: Actual labels (m x 1), values 0 or 1
        y_predicted: Predicted probabilities (m x 1), values 0-1
        m: Number of examples
    Returns:
        Cost (scalar)
    """
    # Clip to avoid log(0)
    y_pred_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
    
    # Binary cross-entropy formula
    cost = -(1/m) * np.sum(
        y_actual * np.log(y_pred_clipped) + 
        (1 - y_actual) * np.log(1 - y_pred_clipped)
    )
    
    return cost
```

### Characteristics

**Pros:**

- Derived from maximum likelihood estimation (theoretically sound)
- Convex function (single global minimum)
- Smooth gradients (easy to optimize)
- Natural for probability predictions
- Heavily penalizes confident wrong predictions

**Cons:**

- Undefined when h = 0 or h = 1 (need clipping)
- Can be sensitive to outliers (very wrong predictions)

**When to use:**

- Standard choice for all binary classification problems
- Default for logistic regression
- When you want calibrated probabilities

**Gradient:**

```text
∂J/∂w = (1/m) * Σ(h - y) * x
```

---

## Cost Function 2: Focal Loss

### Definition

```text
J = -(1/m) * Σ[α * (1-h)^γ * y * log(h) + (1-α) * h^γ * (1-y) * log(1-h)]
```

Where:

- `α` = weighting parameter (typically 0.25)
- `γ` (gamma) = focusing parameter (typically 2)

### How It Works

Focal loss adds a **focusing term** `(1-h)^γ` to the standard cross-entropy.

**When h is high (confident prediction):**

- `(1-h)` is small
- `(1-h)^γ` becomes very small
- Loss is down-weighted (we don't need to focus on confident correct predictions)

**When h is low (wrong prediction):**

- `(1-h)` is close to 1
- `(1-h)^γ` stays close to 1
- Loss is NOT down-weighted (we focus on wrong predictions)

### Example

With γ = 2:

```text
If h = 0.9 and y = 1 (correct, confident):
  (1-h)^2 = (0.1)^2 = 0.01
  Effective loss = 0.01 * (-log(0.9)) ≈ 0.001 (very small)

If h = 0.1 and y = 1 (wrong, confident):
  (1-h)^2 = (0.9)^2 = 0.81
  Effective loss = 0.81 * (-log(0.1)) ≈ 1.87 (large)
```

### Code Implementation

```python
def focal_loss(y_actual, y_predicted, m, alpha=0.25, gamma=2):
    """
    Focal Loss cost function (useful for imbalanced datasets)
    Args:
        y_actual: Actual labels (m x 1)
        y_predicted: Predicted probabilities (m x 1)
        m: Number of examples
        alpha: Weight parameter (default 0.25)
        gamma: Focusing parameter (default 2)
    Returns:
        Cost (scalar)
    """
    # Clip to avoid log(0)
    y_pred_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
    
    # Focal loss formula
    cost = -(1/m) * np.sum(
        alpha * ((1 - y_pred_clipped) ** gamma) * y_actual * np.log(y_pred_clipped) +
        (1 - alpha) * (y_pred_clipped ** gamma) * (1 - y_actual) * np.log(1 - y_pred_clipped)
    )
    
    return cost
```

### Characteristics

**Pros:**

- Handles class imbalance well (e.g., 99% negative, 1% positive)
- Focuses on hard examples (wrong predictions)
- Reduces loss contribution from easy examples
- Still convex

**Cons:**

- More hyperparameters to tune (α, γ)
- Slower computation than binary cross-entropy
- More complex to understand

**When to use:**

- Imbalanced binary classification (one class dominates)
- Object detection tasks
- When you want to focus on hard-to-classify examples

---

## Cost Function 3: Weighted Cross-Entropy

### Definition

```text
J = -(1/m) * Σ[w₁ * y * log(h) + w₀ * (1-y) * log(1-h)]
```

Where:

- `w₁` = weight for positive class
- `w₀` = weight for negative class
- Common choice: `w₁ = count(negative) / total`, `w₀ = count(positive) / total`

### How It Works

Weighted cross-entropy applies different penalties to different classes.

**Example with imbalanced data:**

```text
Dataset: 900 negative, 100 positive (99% vs 1%)

Weights:
  w₁ = 900 / 1000 = 0.9  (weight for positive class)
  w₀ = 100 / 1000 = 0.1  (weight for negative class)

Effect:
  - Errors on positive class (rare) are weighted heavily (0.9)
  - Errors on negative class (common) are weighted lightly (0.1)
  - Model focuses on learning positive class
```

### Code Implementation

```python
def weighted_cross_entropy(y_actual, y_predicted, m, w0=0.5, w1=0.5):
    """
    Weighted Binary Cross-Entropy (for imbalanced data)
    Args:
        y_actual: Actual labels (m x 1)
        y_predicted: Predicted probabilities (m x 1)
        m: Number of examples
        w0: Weight for negative class (default 0.5)
        w1: Weight for positive class (default 0.5)
    Returns:
        Cost (scalar)
    """
    # Clip to avoid log(0)
    y_pred_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
    
    # Weighted cross-entropy
    cost = -(1/m) * np.sum(
        w1 * y_actual * np.log(y_pred_clipped) +
        w0 * (1 - y_actual) * np.log(1 - y_pred_clipped)
    )
    
    return cost
```

### Characteristics

**Pros:**

- Handles class imbalance simply
- Easy to interpret weights
- Only two hyperparameters (w0, w1)

**Cons:**

- Need to choose weights carefully
- Affects all samples equally (vs focal loss which adapts)

**When to use:**

- Imbalanced binary classification
- When you know the misclassification costs
- Medical diagnosis (false negatives more costly than false positives)

---

## Cost Function 4: Hinge Loss

### Definition

```text
J = (1/m) * Σ max(0, 1 - y * ŷ)
```

Where:

- `y` ∈ {-1, +1} (note: not 0/1, but -1/+1)
- `ŷ` = predicted score (not probability, but raw output before sigmoid)

### How It Works

Hinge loss is primarily for Support Vector Machines, but can be adapted for logistic regression.

```text
If y=1 and ŷ=2 (correct, confident): Loss = max(0, 1-2) = 0 (no loss)
If y=1 and ŷ=0.5 (correct, not confident): Loss = max(0, 1-0.5) = 0.5
If y=1 and ŷ=-1 (wrong): Loss = max(0, 1-(-1)) = 2
If y=-1 and ŷ=2 (wrong): Loss = max(0, 1-(-1)*2) = max(0, 3) = 3
```

### Code Implementation

```python
def hinge_loss(y_actual, y_scores, m):
    """
    Hinge Loss (for margin-based classification)
    Note: y_actual should be {-1, +1} not {0, 1}
    Note: y_scores should be raw scores, not sigmoid probabilities
    Args:
        y_actual: Labels {-1, +1} (m x 1)
        y_scores: Predicted scores (m x 1)
        m: Number of examples
    Returns:
        Cost (scalar)
    """
    loss = (1/m) * np.sum(np.maximum(0, 1 - y_actual * y_scores))
    return loss
```

### Characteristics

**Pros:**

- Maximizes margin (creates separation between classes)
- Robust to outliers
- Efficient (only cares about misclassified examples)

**Cons:**

- Non-smooth at boundaries (hard to optimize)
- Requires label transformation (-1/+1)
- Doesn't output probabilities
- Less suitable for logistic regression

**When to use:**

- Support Vector Machines (SVM)
- Not recommended for logistic regression (better options available)

---

## Comparison Table

| Function               | Formula                                | Smooth | Probabilities | Handles Imbalance | Use Case         |
|------------------------|----------------------------------------|:------:|:-------------:|:-----------------:|------------------|
| Binary Cross-Entropy   | `-[y log(h) + (1-y) log(1-h)]`         |   ✓    |       ✓       |         ✗         | Standard         |
| Focal Loss             | Cross-entropy + `(1-h)^γ` weighting    |   ✓    |       ✓       |         ✓         | Imbalanced data  |
| Weighted Cross-Entropy | Weighted binary cross-entropy          |   ✓    |       ✓       |         ✓         | Imbalanced data  |
| Hinge Loss             | `max(0, 1 - y·ŷ)`                      |   ✗    |       ✗       |         ✓         | SVM/Margin-based |

---

## Recommendation for Different Scenarios

### Scenario 1: Balanced Binary Classification

**Use:** Binary Cross-Entropy

```text
Why: Standard choice, works well when classes are balanced
```

### Scenario 2: Imbalanced Binary Classification

**Use:** Weighted Cross-Entropy or Focal Loss

```text
Weighted CE: If you know exact misclassification costs
Focal Loss: If you want adaptive focusing on hard examples
```

### Scenario 3: Medical Diagnosis (False Negative = Bad)

**Use:** Weighted Cross-Entropy with high weight for positive class

```text
Why: False negatives (missing disease) are more costly than false positives
Set w1 >> w0
```

### Scenario 4: Spam Detection

**Use:** Binary Cross-Entropy or Weighted Cross-Entropy

```text
Why: Both false positives (blocking legitimate email) and false negatives
     (allowing spam) have costs. Weight based on business requirements.
```

### Scenario 5: Fraud Detection (Rare Fraud)

**Use:** Focal Loss

```text
Why: Fraud is rare (99.9% normal). Focal loss focuses on hard examples.
Prevents model from ignoring fraud cases.
```

---

## Gradients for Different Cost Functions

### Binary Cross-Entropy Gradient

```text
∂J/∂w = (1/m) * Σ(h - y) * x
∂J/∂b = (1/m) * Σ(h - y)
```

### Focal Loss Gradient

```text
∂J/∂w = (1/m) * Σ[(1-h)^γ * (h - y) - γ*h*(1-h)^(γ-1)*y*log(h)] * x
```

(More complex, requires careful implementation)

### Weighted Cross-Entropy Gradient

```text
∂J/∂w = (1/m) * Σ[w₁*(h - y)*y + w₀*(h - y)*(1-y)] * x
```

(Similar to binary cross-entropy but scaled by weights)

---

## Summary

For logistic regression:

1. **Start with Binary Cross-Entropy** — It's the standard, works well for most cases
2. **If classes are imbalanced** — Use Weighted Cross-Entropy or Focal Loss
3. **If optimization is slow** — Check your learning rate and feature scaling first
4. **Monitor gradients** — Ensure they're neither too large nor too small

The choice of cost function significantly impacts model performance, especially with imbalanced data.

---
