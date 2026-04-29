# Linear Regression: From First Principles

Welcome to the Linear Regression learning module!  
This directory contains comprehensive guides and implementations for understanding linear regression from the ground up.

## What is Linear Regression?

Linear regression is one of the most fundamental machine learning algorithms.  
It models the relationship between input features and a continuous output by fitting a linear function:

```text
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Where:

- **w** = weights (model parameters)
- **x** = features (inputs)
- **b** = bias (intercept)
- **ŷ** = predicted output

## Learning Path

Follow this sequence to understand linear regression from first principles:

### 1. **Core Concepts**

- Start with [COST_FUNCTION_GUIDE.md](COST_FUNCTION_GUIDE.md) — Understand what a cost function is and why it matters
- Read [GRADIENT_DESCENT_GUIDE.md](GRADIENT_DESCENT_GUIDE.md) — Learn how gradient descent optimizes the model

### 2. **Implementation Details**

- Study [COST_FUNCTION_AND_GRADIENT_DESCENT.md](COST_FUNCTION_AND_GRADIENT_DESCENT.md) — Deep dive into the mathematics and implementation
- Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) — Overview of the complete implementation

### 3. **Advanced Topics**

- Explore [MULTIVARIATE_LINEAR_REGRESSION.md](MULTIVARIATE_LINEAR_REGRESSION.md) — Extend linear regression to multiple features

## Code Files

- **[linear_regression.py](linear_regression.py)** — Clean, well-documented implementation of linear regression with gradient descent
- **[cost_function_and_gradient_descent.py](cost_function_and_gradient_descent.py)** — Detailed implementation of cost function and gradient descent optimization

## Key Topics Covered

### Cost Functions

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Gradient computation and backpropagation

### Gradient Descent

- Algorithm intuition and mathematics
- Feature normalization
- Learning rate tuning
- Convergence detection

### Multivariate Regression

- Extending from univariate to multiple features
- Matrix operations and vectorization
- Real-world applications (house price prediction, etc.)

## Quick Start

To explore the implementations:

```python
# Run the main linear regression implementation
python linear_regression.py

# Or explore the cost function and gradient descent details
python cost_function_and_gradient_descent.py
```

## Example Application

The implementations use house price prediction as an example:

```text
Price = w₁*(square footage) + w₂*(number of bedrooms) + w₃*(age) + b
```

This demonstrates how linear regression learns to weight different features to make accurate predictions.

## Mathematical Foundation

All code is built from the mathematical foundations:

1. **Cost Function:** Measures prediction error
2. **Gradient Descent:** Iteratively minimizes the cost function
3. **Convergence:** Detecting when the model has optimized

---
