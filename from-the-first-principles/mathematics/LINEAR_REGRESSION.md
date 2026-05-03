# Mathematics of Linear Regression and Gradient Descent

---

## 1. The Linear Model

$$\hat{y}_i = w_1x_{i1} + w_2x_{i2} + \cdots + w_nx_{in} + b$$

Or in vectorized form:

$$\hat{y} = Xw + b$$

Where:

- $X$ = feature matrix (m × n): m samples, n features
- $w$ = weight vector (n × 1)
- $b$ = bias (scalar)
- $\hat{y}$ = predictions (m × 1)

---

## 2. Cost Functions

### Mean Squared Error (MSE)

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

### Root Mean Squared Error (RMSE)

$$J(w, b) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2}$$

### Mean Absolute Error (MAE)

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} |\hat{y}_i - y_i|$$

---

## 3. Gradient Descent: Partial Derivatives

The key insight: compute how the cost changes with respect to each parameter.

### For weights (for each feature $j$)

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_{ij}$$

In vectorized form:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

### For bias

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

---

## 4. Parameter Update Rules

The gradient descent update happens simultaneously for all parameters:

$$w := w - \alpha \cdot \frac{\partial J}{\partial w}$$

$$b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

Where:

- $\alpha$ = learning rate (hyperparameter you choose)
- The $:=$ notation means "update to"

---

## 5. Algorithm in Pseudocode

```text
Initialize w, b (to 0 or small random values)
For iteration t = 1, 2, ..., max_iterations:
    ŷ ← Xw + b
    J ← compute_cost(ŷ, y)

    dw ← (1/m) * X^T * (ŷ - y)
    db ← (1/m) * mean(ŷ - y)

    w ← w - α * dw
    b ← b - α * db

    record J (for monitoring convergence)
```

---

## 6. Why It Works: Intuition

- **Error term** $(\hat{y} - y)$: How much we missed by
- **Multiply by features** $X^T(\hat{y} - y)$: Which features contributed to the error
- **Subtract the gradient**: Move opposite the direction the cost is increasing
- **Learning rate α**: Controls step size; prevents overshooting

---

## Key Formulas at a Glance

| Concept            | Formula                                   |
|--------------------|-------------------------------------------|
| Prediction         | $\hat{y} = Xw + b$                        |
| MSE                | $\frac{1}{m}\sum(\hat{y} - y)^2$          |
| Gradient w.r.t. w  | $\frac{1}{m}X^T(\hat{y} - y)$             |
| Gradient w.r.t. b  | $\frac{1}{m}\sum(\hat{y} - y)$            |
| Update rule        | $w := w - \alpha \nabla J$                |

---
