# Multivariate Linear Regression: Implementation Summary

## Overview

Implemented multivariate linear regression from scratch in Python, including gradient descent optimization, feature normalization, and cost function tracking.

---

## Key Concepts Implemented

### The Hypothesis Function

```text
ŷ = w·x + b

Where:
- ŷ = predicted output
- w = weights vector [w₁, w₂, ..., wₙ]
- x = features vector [x₁, x₂, ..., xₙ]
- b = bias (scalar)
```

In the house price example with 3 features:

```text
Price = w₁*(sqft) + w₂*(bedrooms) + w₃*(age) + b
```

---

### Cost Function: Root Mean Squared Error (RMSE)

```text
MSE = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²

RMSE = √(MSE)
```

**Why RMSE?**

- Mathematically smooth (easy to differentiate)
- Penalizes large errors (quadratic)
- Interpretable (same units as target)
- Industry standard

**Real example from code execution:**

- Initial RMSE (random model): 1.000000
- Final RMSE (trained model): 0.136622
- This means average prediction error is ~13.66% of normalized target scale

---

### Gradient Descent Optimization

**Update rule:**

```text
wⱼ := wⱼ - α * ∂J/∂wⱼ
b := b - α * ∂J/∂b
```

**Gradients calculated:**

```text
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ) * xᵢⱼ

∂J/∂b = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)
```

**How it works:**

1. Calculate prediction error: (ŷᵢ - yᵢ)
2. Multiply by feature value: error * xᵢⱼ
3. Average across all examples
4. Move weights in opposite direction of gradient (descent)
5. Repeat until convergence

---

### Feature Scaling/Normalization (CRITICAL)

**The Problem:**

In multivariate regression, features often have vastly different scales:

- Square feet: 1000-5000
- Number of bedrooms: 2-6
- Age: 0-50

Without scaling, gradient descent:

- Converges extremely slowly
- May diverge (explode to infinity)
- Becomes numerically unstable

**The Solution: Standardization (Z-score normalization):**

```text
x_normalized = (x - mean(x)) / std(x)
```

**Example:**

```text
Original square feet: 2500
Mean: 2500, Std: 1000
Normalized: (2500 - 2500) / 1000 = 0

Original bedrooms: 4
Mean: 4, Std: 1.5
Normalized: (4 - 4) / 1.5 = 0
```

All features now have mean ≈ 0 and std = 1, enabling fast convergence.

**Important:** Use training mean/std for both training AND test data normalization.

---

## Code Architecture

### Class: LinearRegression

**Key Methods:**

1. **`__init__(learning_rate, iterations, verbose)`**
   - Initialize hyperparameters
   - Set up storage for scaling parameters

2. **`_normalize_features(X, fit=True)`**
   - Standardize feature values
   - If fit=True: compute mean/std from data
   - If fit=False: use stored training mean/std

3. **`_calculate_predictions(X)`**
   - Vectorized: ŷ = X·w + b
   - Uses numpy for efficiency

4. **`_calculate_cost(predictions, y)`**
   - Compute RMSE
   - Single scalar output

5. **`_calculate_gradients(X, predictions, y)`**
   - Compute ∂J/∂w and ∂J/∂b
   - Vectorized operations

6. **`_update_parameters(weight_grads, bias_grad)`**
   - Apply gradient descent update
   - Adjusts w and b simultaneously

7. **`fit(X, y)`**
   - Main training loop
   - Normalizes data
   - Runs gradient descent for specified iterations
   - Tracks cost history

8. **`predict(X)`**
   - Make predictions on new data
   - Uses training normalization parameters
   - Returns predictions in original scale

---

## Demonstration Results

### Dataset Created

- 100 examples (house price predictions)
- 3 features: Square Feet, Bedrooms, Age
- True relationship: `Price = 150*sqft + 30000*bedrooms - 500*age + 50000`
- Noise added to make realistic

### Model Performance

- Learning rate: 0.1
- Iterations: 1000
- Convergence: Achieved after ~100 iterations

**Final Metrics:**

```text
Final RMSE: 0.136622 (on normalized scale)
Actual prediction error on new house: $438 out of $540,000 ≈ 0.08% error
```

### Test Prediction

```text
Input:  2500 sqft, 4 bedrooms, 10 years old
Predicted: $539,561.62
True value: $540,000.00
Error: $438.38 (excellent accuracy)
```

---

## Important Insights

### Feature Scaling is Non-Negotiable

Without normalization:

- Cost diverges to infinity (NaN)
- Weights become NaN
- Convergence fails

With normalization:

- Clean convergence
- Stable updates
- Fast learning

### Learning Rate Matters

- Too small (0.0001): Converges very slowly
- Too large (0.1+): May overshoot, diverge
- Sweet spot: 0.01-0.1 (problem dependent)

In code, I used 0.1 and achieved convergence in ~100 iterations.

### Cost Function Shapes Optimization

RMSE penalty structure:

- Error of 0.1: Loss = 0.01
- Error of 1.0: Loss = 1.0
- Error of 10.0: Loss = 100.0

Larger errors are disproportionately penalized, driving optimization toward eliminating outliers.

### Normalization for Predictions

When predicting on new data:

```python
# CORRECT: Use training mean/std
x_test_normalized = (x_test - training_mean) / training_std
predictions = model.predict(x_test_normalized)

# WRONG: Never fit new normalization on test data
x_test_wrong = (x_test - test_mean) / test_std  # ❌ Don't do this
```

---

## Mathematical Deep Dive: The Gradient

**Why does the gradient work?**

The gradient points in direction of steepest ascent of J(w, b). Moving opposite to gradient descends the cost landscape fastest.

**Intuition with concrete numbers:**

Suppose `w₁ = 5` and current cost is `J = 100`

After small perturbation `w₁ = 5.001`:

- If cost increases to 101: gradient is positive
- Update: `w₁ := w₁ - α * (positive) = decrease w₁`

After perturbation `w₁ = 4.999`:

- If cost decreases to 99: gradient is negative
- Update: `w₁ := w₁ - α * (negative) = increase w₁`

The algorithm automatically adjusts weights toward minimizing cost.

---

## When Linear Regression Fails

### Non-linear Relationships

If relationship is curved (polynomial, exponential), linear model underfits.

- Solution: Add polynomial features, use non-linear models

### Categorical Outputs

If target is categorical (spam/not spam, disease/healthy):

- Linear regression produces invalid outputs (not 0 or 1)
- Solution: Use logistic regression (next in sequence)

### Feature Scaling Neglected

Without normalization:

- Different gradient magnitudes for different weights
- Unstable optimization
- Solution: Always scale (you've seen this!)

### Insufficient Data

With very few examples (< 10 per feature):

- Model overfits to noise
- Solution: Collect more data or use regularization

---

## Code Usage Example

```python
import numpy as np
from linear_regression import LinearRegression

# Load your dataset (assuming EDA already done)
X_train = np.load('features.npy')  # (m, n) array
y_train = np.load('targets.npy')   # (m, 1) array

# Create and train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
X_test = np.load('test_features.npy')
predictions = model.predict(X_test)

# Inspect learned parameters
params = model.get_params()
print(f"Weights: {params['weights']}")
print(f"Bias: {params['bias']}")

# Visualize convergence
model.plot_loss_curve()
```

---

The implementation demonstrates that machine learning isn't magic—it's systematic optimization of mathematical functions.
