# Cost Functions in Machine Learning: Complete Guide

## What is a Cost Function?

A cost function is a mathematical measure that quantifies how poorly a model performs on a dataset.  
It takes two inputs:

- Actual values from the dataset (y)
- Predicted values from the model (ŷ)

And outputs a single scalar number representing the total error.  

**Intuition:** Think of it as a "report card" for the model. Lower score = better performance.  

```text
Cost = f(actual, predicted)
```

The machine learning algorithm's job is to minimize this cost.

---

## Why Multiple Cost Functions Exist

Different problems need different error measures because they care about different aspects:

- **MSE:** Penalizes large errors heavily (outlier-sensitive)
- **MAE:** Treats all errors equally (robust to outliers)
- **RMSE:** Interpretable like MSE but in original units
- **Huber:** Compromise between MSE and MAE
- **Log Loss:** For classification (probabilities)
- **Cross Entropy:** For multi-class classification

---

## Regression Cost Functions

### Mean Squared Error (MSE)

```text
MSE = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²
```

**Characteristics:**

- Quadratic penalty (error of 2 → loss of 4)
- Always non-negative
- Differentiable everywhere
- Heavily penalizes outliers

**When to use:**

- Standard regression problems
- When large errors are particularly bad
- When all errors matter equally

**Pros:**

- Mathematically smooth for optimization
- Widely understood
- Can use least squares closed-form solution

**Cons:**

- Outliers can dominate (squared term)
- Not interpretable in original units

**Closed-form solution (Normal Equation):**

```text
w = (X^T.X)^(-1).X^T y
```

This is an alternative to gradient descent—direct calculation without iteration.

---

### Root Mean Squared Error (RMSE)

```text
RMSE = √((1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²)
     = √(MSE)
```

**Characteristics:**

- Same behavior as MSE
- But expressed in original units
- Square root adds non-linearity

**When to use:**

- When interpretation matters (e.g., "average error is $50K")
- Same situations as MSE but need interpretability

**Pros:**

- Interpretable (same units as target)
- Optimization same as MSE
- Industry standard metric

**Cons:**

- Still outlier-sensitive
- Square root slightly slower to compute

**Relationship to MSE:**

```text
RMSE is always ≥ MSE (due to square root)
Example: MSE = 100 → RMSE = 10
```

---

### Mean Absolute Error (MAE)

```text
MAE = (1/m) Σᵢ₌₁ᵐ |ŷᵢ - yᵢ|
```

**Characteristics:**

- Linear penalty (error of 2 → loss of 2)
- All errors contribute equally
- Robust to outliers
- Non-differentiable at zero (kink)

**When to use:**

- Data with outliers
- When equal error penalty needed
- Financial predictions (care about size of error, not squared)

**Pros:**

- Robust to outliers
- Interpretable
- Represents median error trend

**Cons:**

- Not smooth at zero (harder to optimize)
- Gradient doesn't exist at zero
- May have multiple minima

**Gradient issues:**

```text
d|x|/dx = 1 if x > 0
d|x|/dx = -1 if x < 0
d|x|/dx = undefined if x = 0 (non-differentiable)
```

---

### Huber Loss

```text
            { (1/2)(ŷ - y)²           if |ŷ - y| ≤ δ
L(y, ŷ) = 
            { δ(|ŷ - y| - δ/2)        if |ŷ - y| > δ
```

**Interpretation:**

- δ (delta) is a threshold parameter
- Below δ: quadratic (like MSE)
- Above δ: linear (like MAE)

**Characteristics:**

- "Best of both worlds"
- Smooth (differentiable everywhere)
- Outlier-robust (linear tail)
- Need to tune δ parameter

**When to use:**

- Mixed data quality (some outliers but not many)
- When robustness AND optimization matter
- When interpretability needed

**Pros:**

- Smooth and differentiable
- Robust to outliers
- Compromises MSE and MAE well

**Cons:**

- Need to choose δ parameter
- More complex
- Gradient computation slightly harder

**Example with δ = 1:**

```text
Error = 00.5:  Loss = (1/2)*0.5² = 0.125 (quadratic)
Error = 01.0:  Loss = (1/2)*1² = 0.5 (quadratic boundary)
Error = 02.0:  Loss = 1*(2 - 0.5) = 1.5 (linear)
Error = 10.0:  Loss = 1*(10 - 0.5) = 9.5 (linear)
```

**Gradient:**

```text
        { (ŷ - y)           if |ŷ - y| ≤ δ
∂L/∂ŷ = 
        { δ * sign(ŷ - y)   if |ŷ - y| > δ
```

---

### Log-Cosh Loss

```text
L(y, ŷ) = (1/m) Σᵢ₌₁ᵐ log(cosh(ŷᵢ - yᵢ))
```

Where `cosh(x) = (e^x + e^(-x)) / 2`

**Characteristics:**

- Smooth approximation to MAE
- For small errors: approximately MSE
- For large errors: approximately MAE
- Twice-differentiable (smooth gradients)

**When to use:**

- When optimization with outliers matters
- Neural networks needing smooth, twice-differentiable loss

**Pros:**

- Smooth everywhere
- Approximately MSE for small errors
- Approximately MAE for large errors
- Good for neural network optimization

**Cons:**

- Computationally more expensive
- Less intuitive
- Less commonly used

---

## Classification Cost Functions

### Binary Cross-Entropy (Log Loss)

```text
BCE = -(1/m) Σᵢ₌₁ᵐ [yᵢ*log(ŷᵢ) + (1-yᵢ)*log(1-ŷᵢ)]
```

Where:

- y ∈ {0, 1} (actual class)
- ŷ ∈ (0, 1) (predicted probability)

**Interpretation:**

- Measures divergence between actual and predicted probability distributions
- Only active term depends on actual value:
  - If y=1: loss = -log(ŷ) (penalizes predicting low probability)
  - If y=0: loss = -log(1-ŷ) (penalizes predicting high probability)

**Characteristics:**

- Always non-negative
- Zero when prediction = actual
- Approaches infinity when confidence is wrong
- Smooth and differentiable

**When to use:**

- Binary classification (spam/not spam, disease/healthy)
- Logistic regression
- Any model outputting probabilities

**Pros:**

- Designed for probabilities
- Well-behaved gradients
- Natural for classification

**Cons:**

- Only for 0/1 targets
- Large penalty for confident wrong predictions

**Behavior:**

```text
If true=1, predict=0.9: loss = -log(0.9) ≈ 0.105 (small)
If true=1, predict=0.1: loss = -log(0.1) ≈ 2.303 (large)
If true=1, predict=0.01: loss = -log(0.01) ≈ 4.605 (very large)
```

---

### Categorical Cross-Entropy

```text
CCE = -(1/m) Σᵢ₌₁ᵐ Σⱼ₌₁ᶜ yᵢⱼ * log(ŷᵢⱼ)
```

Where:

- c = number of classes
- yᵢⱼ = 1 if example i belongs to class j, else 0 (one-hot encoded)
- ŷᵢⱼ = predicted probability for class j

**Interpretation:**

- Generalizes binary cross-entropy to multiple classes
- Only the term for true class contributes (rest are zero)

**Example with 3 classes:**

```text
True: [1, 0, 0] (class 0)
Predicted: [0.7, 0.2, 0.1]

Loss = -[1*log(0.7) + 0*log(0.2) + 0*log(0.1)]
     = -log(0.7) ≈ 0.357
```

**When to use:**

- Multi-class classification (3+ classes)
- One-hot encoded targets
- Softmax output layer

---

## Comparison Table

| Function        | Type           | Formula                            | Smooth | Outlier Robust | Interpretable | Differentiable |
|-----------------|----------------|------------------------------------|--------|----------------|---------------|----------------|
| MSE             | Regression     | Σ(e²)/m                            | Yes    | No             | No            | Yes            |
| RMSE            | Regression     | √(Σ(e²)/m)                         | Yes    | No             | Yes           | Yes            |
| MAE             | Regression     | Σ\|e\|/m                           | Yes    | Yes            | Yes           | No (at 0)      |
| Huber           | Regression     | Mixed                              | Yes    | Yes            | Partial       | Yes            |
| Log-Cosh        | Regression     | log(cosh(e))                       | Yes    | Yes            | No            | Yes            |
| BCE             | Classification | -[y log(ŷ) + (1-y) log(1-ŷ)]       | Yes    | N/A            | N/A           | Yes            |
| Categorical CE  | Classification | -Σ y log(ŷ)                        | Yes    | N/A            | N/A           | Yes            |

---

## How to Choose a Cost Function

### Decision Tree

**Start: What is the target variable?**

```bash
├─ Continuous (Regression)
│  ├─ No outliers, normal data distribution?
│  │  └─ Use MSE/RMSE
│  │
│  └─ Have outliers or need robustness?
│     ├─ Many outliers?
│     │  └─ Use MAE
│     │
│     └─ Some outliers?
│        └─ Use Huber or Log-Cosh
│
└─ Categorical (Classification)
   ├─ Binary classification?
   │  └─ Use Binary Cross-Entropy
   │
   └─ Multi-class classification?
      └─ Use Categorical Cross-Entropy
```

---

## Mathematical Properties

### Convexity

A cost function is convex if:

- Any local minimum is a global minimum
- No "valleys and peaks" that trap optimization

**Convex functions (good for optimization):**

- MSE
- RMSE
- MAE
- Huber
- Binary Cross-Entropy

**Non-convex functions (harder):**

- Neural networks with multiple layers
- Some deep learning architectures

---

## Real-World Examples

### Example 1: House Price Prediction

**Context:** Predicting house prices. Occasional luxury properties (outliers).

**Analysis:**

- Outliers? Yes (mansion worth 10M, typical houses 500K)
- Need interpretability? Yes (explain in dollars)
- Distribution? Some high-value outliers

**Best choice:** **Huber Loss**

- MSE would be pulled by mansions
- MAE would ignore price ranges
- Huber balances both

---

### Example 2: Medical Test Results

**Context:** Predicting disease presence (probability from 0 to 1).

**Analysis:**

- Output type? Probability
- Classes? 2 (disease/no disease)
- Error cost? Symmetric

**Best choice:** **Binary Cross-Entropy**

- Designed for probabilities
- Natural interpretation: divergence from true probability

---

### Example 3: Image Classification

**Context:** Classifying images into 1000 categories.

**Analysis:**

- Classes? 1000 (multi-class)
- Output? One-hot encoding
- Probabilities? Yes (softmax)

**Best choice:** **Categorical Cross-Entropy**

- Standard for multi-class
- Works with softmax output

---

## Implementation Notes

### Computing Cost for Multiple Predictions

```python
# BAD: Using loops
total_error = 0
for i in range(m):
    total_error += (y_pred[i] - y_actual[i]) ** 2
cost = total_error / m

# GOOD: Vectorized
cost = np.mean((y_pred - y_actual) ** 2)

# Speed difference: 100x faster for large datasets
```

### Numerical Stability

**Problem in Cross-Entropy:**

```python
# BAD: log(0) = -infinity
loss = -log(predicted_prob)
# If predicted_prob = 0, this crashes

# GOOD: Clip values
predicted_prob = np.clip(predicted_prob, 1e-7, 1 - 1e-7)
loss = -log(predicted_prob)
```

---

## Summary Table: Quick Reference

| Problem                        | Cost Function                | Reason                          |
|--------------------------------|------------------------------|---------------------------------|
| Regression, clean data         | RMSE                         | Standard, interpretable         |
| Regression, with outliers      | Huber/MAE                    | Robust                          |
| Binary classification          | Binary Cross-Entropy         | For probabilities               |
| Multi-class classification     | Categorical Cross-Entropy    | Extends binary version          |
| Financial predictions          | MAE                          | All errors equally important    |
| Image prediction (pixels)      | MSE                          | Smooth pixel-level errors       |

---
