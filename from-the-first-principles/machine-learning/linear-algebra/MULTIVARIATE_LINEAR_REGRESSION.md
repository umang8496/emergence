# Linear Regression

## Multivariate Linear Regression

Multivariate Linear Regression is an algorithm that finds a linear relationship between multiple input features and a single output variable.

**Core Idea:**

```text
Output = w₁·Feature₁ + w₂·Feature₂ + ... + wₙ·Featureₙ + b

Where:
- Output (y) = what I want to predict
- Features (x₁, x₂, ..., xₙ) = inputs that influence the output
- Weights (w₁, w₂, ..., wₙ) = how much each feature matters
- Bias (b) = baseline value when all features are zero
```

In vector notation:

```text
ŷ = w·x + b
```

Where `ŷ` is the predicted output.

---

## When Should We Use Multivariate Linear Regression?

### Data Pattern Requirements

We should use multivariate linear regression when:

1. **Linearity Assumption:** The relationship between inputs and output is approximately linear.
   - If I plot features vs output, they should show a roughly linear trend
   - Not curved, not exponential, not categorical boundaries

2. **Continuous Output:** The target variable is continuous (not discrete).
   - Example: house price, temperature, salary
   - NOT suitable for: spam/not spam, disease/no disease

3. **Multiple Numeric Features:** I have 2 or more numeric input features.
   - Each feature should be numeric (not categorical text)
   - If categorical, I need to encode them first (EDA step)

4. **Independence:** Features should be somewhat independent of each other.
   - If two features are highly correlated, one becomes redundant

5. **Scale:** Dataset size is moderate to large.
   - Works with 100s to millions of samples
   - For very small datasets (< 50 samples), risk of overfitting

### Example Scenarios

**Good fit for linear regression:**

- Predicting house price from: square footage, number of bedrooms, age, location
- Predicting salary from: years of experience, education level, performance score
- Predicting crop yield from: rainfall, temperature, fertilizer amount

**Poor fit for linear regression:**

- Predicting stock price (highly non-linear, affected by external events)
- Medical diagnosis (categorical output: disease/no disease)
- Image recognition (relationship is extremely non-linear)

---

## What Does the Algorithm Do?

The algorithm learns the optimal weights (w) and bias (b) through iterative optimization.

**Step-by-step process:**

1. **Initialize:** Start with random w and b values (typically zeros or small random numbers)

2. **Predict:** For each training example, calculate `ŷ = w·x + b`

3. **Calculate Error:** Measure how far predictions are from actual values using a cost function

4. **Compute Gradients:** Calculate how to adjust w and b to reduce error

5. **Update Parameters:** Move `w` and `b` in the direction that reduces error (Gradient Descent)

6. **Repeat:** Steps 2-5 for multiple iterations until error stabilizes

The algorithm finds the `w` and `b` that minimize the cost function.

---

## Cost Function: Root Mean Squared Error (RMSE)

### Why RMSE?

- Is mathematically smooth (easy to optimize)
- Penalizes larger errors more (quadratic penalty)
- Is interpretable (in same units as output variable)
- Is industry standard for regression

### Mathematical Definition

```text
MSE = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²

RMSE = √(MSE) = √((1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²)

Where:
- m = number of training examples
- ŷᵢ = predicted value for example i
- yᵢ = actual value for example i
```

### Concrete Example

Suppose we have 3 predictions:

```text
Actual values:    y = [100, 200, 300]
Predicted values: ŷ = [95, 210, 290]
Errors:           e = [5, -10, 10]

Squared errors:        [25, 100, 100]
MSE = (1/3) * (25 + 100 + 100) = 75
RMSE = √75 ≈ 8.66
```

The RMSE of 8.66 means on average my predictions are off by 8.66 units.

### Gradient Calculation

To optimize, I need gradients (direction to adjust parameters):

```text
∂(RMSE)/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ) · xᵢⱼ

∂(RMSE)/∂b = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)
```

These tell me: if gradient is positive, decrease the weight; if negative, increase it.

---

## Gradient Descent: The Optimization Algorithm

### The Update Rule

```text
wⱼ := wⱼ - α · ∂J/∂wⱼ
b := b - α · ∂J/∂b
```

Where:

- `:=` means "update to"
- `α` (alpha) = learning rate (how big each step is)
- `∂J/∂wⱼ` = gradient (direction of steepest ascent)

### Learning Rate (α)

- Too small: Optimization is slow, takes many iterations
- Too large: Might overshoot, miss the minimum, or diverge
- Good value: 0.01 to 0.1 (depends on problem)

### Convergence

Stop when:

- Loss stops decreasing significantly
- Reach maximum iterations
- Loss change between iterations < threshold (0.0001)

---

## Python Implementation

See the accompanying `linear_regression.py` file for complete, annotated code.

- `LinearRegression` class with `fit()` and `predict()` methods
- Gradient descent optimization
- RMSE calculation
- Learning curve visualization

---

## Important Notes

### Feature Scaling

In multivariate regression, features often have different scales:

- Square footage: 1000-5000
- Number of bedrooms: 2-10
- Age in years: 0-100

Gradient descent converges slower with unscaled features. Solution: Normalize features.

```text
x_normalized = (x - mean(x)) / std(x)
```

### Assumptions

Linear regression assumes:

1. Linear relationship exists
2. Errors are normally distributed
3. No multicollinearity (features not highly correlated)
4. Homoscedasticity (constant variance of errors)

If assumptions violated, accuracy decreases.

### Overfitting vs Underfitting

- **Underfitting:** Model too simple, high error on training data
- **Overfitting:** Model fits noise, low training error but high test error
- **Goal:** Balanced model that generalizes well

---
