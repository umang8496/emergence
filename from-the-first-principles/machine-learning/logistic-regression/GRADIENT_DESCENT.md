# Gradient Descent for Logistic Regression: Complete Guide

## Introduction

Gradient descent for logistic regression is almost identical to linear regression, but with one key difference: the cost function and hence the gradients are different due to the sigmoid function.  
However, the optimization strategy and variants remain the same.

---

## Part 1: The Basics - Why Gradient Descent?

### The Problem

We have a cost function:

```text
J(w, b) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
where h = sigmoid(w·x + b)
```

We want to find `w` and `b` that minimize this cost.

### The Solution: Follow the Gradient Downhill

**Analogy:** Imagine walking down a foggy mountain. You can't see the valley, but you can feel the slope beneath your feet.  
If you always walk down the steepest slope, you'll eventually reach the valley.  

The gradient tells us the slope. Gradient descent follows this slope to minimize cost.  

---

## Part 2: Computing Gradients for Logistic Regression

### The Gradient Formulas

For binary cross-entropy cost with logistic regression:

**Gradient for weights:**

```text
∂J/∂w = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ) * xᵢ
```

**Gradient for bias:**

```text
∂J/∂b = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ)
```

Where:

- m = number of training examples
- hᵢ = sigmoid(wᵀxᵢ + b) = predicted probability
- yᵢ = actual label (0 or 1)
- xᵢ = features for example i

### Why These Formulas?

**Derivation using chain rule:**

```text
J = -(1/m) * Σ[y*log(σ(z)) + (1-y)*log(1-σ(z))]

where z = w·x + b and σ(z) = 1/(1+e^(-z))

Chain rule:
∂J/∂w = ∂J/∂σ * ∂σ/∂z * ∂z/∂w

Computing each part:
∂J/∂σ = -y/σ + (1-y)/(1-σ)
∂σ/∂z = σ(1-σ)
∂z/∂w = x

Multiplying:
∂J/∂w = [-y/σ + (1-y)/(1-σ)] * σ(1-σ) * x
       = [-y(1-σ) + (1-y)σ] * x
       = [σ - y] * x
```

This simplifies beautifully to: `(h - y) * x`  

**Remarkable observation:** The gradient formula for logistic regression is **identical** to linear regression! The only difference is that `h` now includes sigmoid.

### Code Implementation

```python
def compute_gradients(X, h, y, m):
    """
    Compute gradients for logistic regression.
    Args:
        X: Features (m x n)
        h: Predictions/probabilities (m x 1)
        y: Actual labels (m x 1)
        m: Number of examples
    Returns:
        dw: Gradient for weights (n x 1)
        db: Gradient for bias (scalar)
    """
    
    # Compute error
    error = h - y  # (m x 1)
    
    # Gradient for weights
    dw = (1/m) * np.dot(X.T, error)  # (n x 1)
    
    # Gradient for bias
    db = (1/m) * np.sum(error)  # scalar
    
    return dw, db
```

---

## Part 3: The Update Rule

Once we have gradients, we update parameters:

```text
w := w - learning_rate * ∂J/∂w
b := b - learning_rate * ∂J/∂b
```

### The Learning Rate (α)

The learning rate controls how big each step is:

**Too small (α = 0.0001):**

```text
Iteration 1: Cost = 100
Iteration 100: Cost = 99.5
Iteration 1000: Cost = 98.0
...very slow convergence
```

**Just right (α = 0.01):**

```text
Iteration 1: Cost = 100
Iteration 10: Cost = 50
Iteration 20: Cost = 25
Iteration 30: Cost = 12
...smooth convergence
```

**Too large (α = 1.0):**

```text
Iteration 1: Cost = 100
Iteration 2: Cost = 150  (overshot)
Iteration 3: Cost = 200  (diverging)
...divergence to infinity
```

### Code Implementation

```python
def update_parameters(w, b, dw, db, learning_rate):
    """
    Update parameters using gradient descent.
    Args:
        w: Current weights (n x 1)
        b: Current bias (scalar)
        dw: Gradient for weights (n x 1)
        db: Gradient for bias (scalar)
        learning_rate: Step size (alpha)
    Returns:
        w_new: Updated weights
        b_new: Updated bias
    """
    w_new = w - learning_rate * dw
    b_new = b - learning_rate * db
    
    return w_new, b_new
```

---

## Part 4: Batch Gradient Descent (Standard)

### Algorithm

```text
for iteration = 1 to num_iterations:
    z = X·w + b              # All m examples
    h = sigmoid(z)           # All m examples
    J = compute_cost(h, y)
    dw, db = compute_gradients(X, h, y, m)
    w, b = update_parameters(w, b, dw, db, α)
    
    if J stopped decreasing:
        break
```

### Characteristics

**Pros:**

- Uses exact gradient (all data at once)
- Smooth convergence
- Guaranteed to converge (for logistic regression)
- Good for understanding behavior

**Cons:**

- Slow for large datasets
- Requires loading all data into memory
- Each iteration is expensive (m examples)

### Code Implementation

```python
def train_batch_gd(X, y, learning_rate=0.01, iterations=1000):
    """
    Train using Batch Gradient Descent.    
    Args:
        X: Training features (m x n)
        y: Training labels (m x 1)
        learning_rate: Step size
        iterations: Number of iterations
    Returns:
        w: Final weights
        b: Final bias
        cost_history: Cost at each iteration
    """
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    cost_history = []
    
    for iteration in range(iterations):
        # Forward pass
        z = np.dot(X, w) + b
        h = sigmoid(z)
        
        # Compute cost
        cost = compute_cost(h, y, m)
        cost_history.append(cost)
        
        # Compute and update gradients
        dw, db = compute_gradients(X, h, y, m)
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Cost = {cost:.6f}")
    
    return w, b, cost_history
```

**When to use:**

- Small to medium datasets (< 10K examples)
- When you want smooth convergence curves
- When you have plenty of memory

---

## Part 5: Stochastic Gradient Descent (SGD)

### Algorithm

```text
for iteration = 1 to num_iterations:
    for each training example i:
        z_i = w·x_i + b              # Single example
        h_i = sigmoid(z_i)
        J_i = cost for example i
        dw, db = gradient for example i
        w, b = update parameters
```

### Characteristics

**Pros:**

- Very fast per update (single example)
- Memory efficient (load one example at a time)
- Can handle streaming data (online learning)
- Noise can help escape local minima

**Cons:**

- Noisy convergence (cost bounces around)
- Hard to parallelize
- Learning rate must decay to converge
- Needs careful tuning

### Code Implementation

```python
def train_sgd(X, y, learning_rate=0.01, iterations=1000):
    """
    Train using Stochastic Gradient Descent.    
    Args:
        X: Training features (m x n)
        y: Training labels (m x 1)
        learning_rate: Initial step size (should decay)
        iterations: Number of iterations
    Returns:
        w: Final weights
        b: Final bias
        cost_history: Cost at each iteration
    """
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    cost_history = []
    
    for iteration in range(iterations):
        # Shuffle data
        shuffle_idx = np.random.permutation(m)
        X_shuffled = X[shuffle_idx]
        y_shuffled = y[shuffle_idx]
        
        # Process each example
        for i in range(m):
            x_i = X_shuffled[i:i+1]  # Single example
            y_i = y_shuffled[i:i+1]
            
            # Forward pass
            z_i = np.dot(x_i, w) + b
            h_i = sigmoid(z_i)
            
            # Compute gradient (single example)
            dw = (h_i - y_i) * x_i.T
            db = (h_i - y_i)
            
            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db
        
        # Compute full cost for monitoring
        h_full = sigmoid(np.dot(X, w) + b)
        cost = compute_cost(h_full, y, m)
        cost_history.append(cost)
        
        # Decay learning rate
        learning_rate *= 0.9995
        
        if (iteration + 1) % 100 == 0:
            print(f"Epoch {iteration + 1}: Cost = {cost:.6f}")
    
    return w, b, cost_history
```

**When to use:**

- Large datasets (millions of examples)
- Online learning (streaming data)
- When memory is limited
- When you want fast training

---

## Part 6: Mini-Batch Gradient Descent

### Algorithm

```text
batch_size = 32
for iteration = 1 to num_iterations:
    for each batch of 32 examples:
        z = X_batch·w + b              # 32 examples
        h = sigmoid(z)
        J_batch = cost for batch
        dw, db = gradient for batch
        w, b = update parameters
```

### Characteristics

**Pros:**

- Balance between batch GD and SGD
- Less noisy than SGD, faster than batch GD
- Parallelizable (efficient on GPUs)
- Industry standard choice
- Smooth convergence with some noise

**Cons:**

- Need to choose batch size
- More complex than pure GD or SGD

### Code Implementation

```python
def train_minibatch_gd(X, y, learning_rate=0.01, iterations=1000, batch_size=32):
    """
    Train using Mini-Batch Gradient Descent.
    Args:
        X: Training features (m x n)
        y: Training labels (m x 1)
        learning_rate: Step size
        iterations: Number of epochs
        batch_size: Size of each mini-batch
    Returns:
        w: Final weights
        b: Final bias
        cost_history: Cost at each iteration
    """
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    cost_history = []
    
    for epoch in range(iterations):
        # Shuffle data
        shuffle_idx = np.random.permutation(m)
        X_shuffled = X[shuffle_idx]
        y_shuffled = y[shuffle_idx]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = len(X_batch)
            
            # Forward pass
            z_batch = np.dot(X_batch, w) + b
            h_batch = sigmoid(z_batch)
            
            # Compute gradients
            dw = (1/batch_m) * np.dot(X_batch.T, (h_batch - y_batch))
            db = (1/batch_m) * np.sum(h_batch - y_batch)
            
            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db
        
        # Compute full cost for monitoring
        h_full = sigmoid(np.dot(X, w) + b)
        cost = compute_cost(h_full, y, m)
        cost_history.append(cost)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Cost = {cost:.6f}")
    
    return w, b, cost_history
```

**When to use:**

- Most practical scenarios
- Default for modern ML frameworks
- Good balance of speed and stability

---

## Part 7: Advanced Optimizers

### Momentum

**Idea:** Accumulate gradient direction over time

```text
v = 0
for iteration = 1 to num_iterations:
    gradient = compute_gradient()
    v = β*v + (1-β)*gradient          # Accumulate momentum
    w := w - learning_rate * v
```

**Effect:** Smoother convergence, faster than vanilla GD

```python
def train_with_momentum(X, y, learning_rate=0.01, iterations=1000, batch_size=32, beta=0.9):
    """
    Train using mini-batch GD with momentum.
    """
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    v_w = np.zeros((n, 1))
    v_b = 0.0
    cost_history = []
    
    for epoch in range(iterations):
        shuffle_idx = np.random.permutation(m)
        X_shuffled = X[shuffle_idx]
        y_shuffled = y[shuffle_idx]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = len(X_batch)
            
            # Forward pass
            h_batch = sigmoid(np.dot(X_batch, w) + b)
            
            # Gradients
            dw = (1/batch_m) * np.dot(X_batch.T, (h_batch - y_batch))
            db = (1/batch_m) * np.sum(h_batch - y_batch)
            
            # Momentum update
            v_w = beta * v_w + (1 - beta) * dw
            v_b = beta * v_b + (1 - beta) * db
            
            # Parameter update
            w -= learning_rate * v_w
            b -= learning_rate * v_b
        
        # Monitor cost
        h_full = sigmoid(np.dot(X, w) + b)
        cost = compute_cost(h_full, y, m)
        cost_history.append(cost)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Cost = {cost:.6f}")
    
    return w, b, cost_history
```

### Adam (Adaptive Moment Estimation)

**Idea:** Combine momentum with per-parameter adaptive learning rates

```python
def train_with_adam(X, y, learning_rate=0.001, iterations=1000, batch_size=32, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Train using Adam optimizer.
    """
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    
    # First moment (momentum)
    m_w = np.zeros((n, 1))
    m_b = 0.0
    
    # Second moment (RMSprop)
    v_w = np.zeros((n, 1))
    v_b = 0.0
    
    t = 0  # Time step
    cost_history = []
    
    for epoch in range(iterations):
        shuffle_idx = np.random.permutation(m)
        X_shuffled = X[shuffle_idx]
        y_shuffled = y[shuffle_idx]
        
        for i in range(0, m, batch_size):
            t += 1
            
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = len(X_batch)
            
            # Forward pass
            h_batch = sigmoid(np.dot(X_batch, w) + b)
            
            # Gradients
            dw = (1/batch_m) * np.dot(X_batch.T, (h_batch - y_batch))
            db = (1/batch_m) * np.sum(h_batch - y_batch)
            
            # Update biased first moment estimate
            m_w = beta1 * m_w + (1 - beta1) * dw
            m_b = beta1 * m_b + (1 - beta1) * db
            
            # Update biased second raw moment estimate
            v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
            v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
            
            # Compute bias-corrected first moment estimate
            m_w_hat = m_w / (1 - beta1 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            v_w_hat = v_w / (1 - beta2 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            
            # Update parameters
            w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        # Monitor cost
        h_full = sigmoid(np.dot(X, w) + b)
        cost = compute_cost(h_full, y, m)
        cost_history.append(cost)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Cost = {cost:.6f}")
    
    return w, b, cost_history
```

---

## Part 8: Convergence Diagnostics

### How to Know If Your Training Is Working

**Good convergence:**

```text
Iteration 100: Cost = 0.50
Iteration 200: Cost = 0.40
Iteration 300: Cost = 0.35
Iteration 400: Cost = 0.33
Iteration 500: Cost = 0.32
```

Cost decreases smoothly. Model is learning.

**Learning too slowly:**

```text
Iteration 100: Cost = 0.50
Iteration 200: Cost = 0.49
Iteration 300: Cost = 0.48
Iteration 400: Cost = 0.47
Iteration 500: Cost = 0.46
```

Learning rate might be too small. Increase it.

**Diverging:**

```text
Iteration 100: Cost = 0.50
Iteration 200: Cost = 0.75
Iteration 300: Cost = 1.50
Iteration 400: Cost = 3.50
Iteration 500: Cost = inf (NaN)
```

Learning rate too large. Decrease it significantly.

**Oscillating (noisy):**

```text
Iteration 100: Cost = 0.50
Iteration 200: Cost = 0.48
Iteration 300: Cost = 0.55
Iteration 400: Cost = 0.47
Iteration 500: Cost = 0.52
```

Batch size might be too small or learning rate slightly high. Use mini-batch or decay learning rate.

---

## Comparison Table

| Variant    | Speed  | Memory | Stability   | Noise      | Use Case         |
|------------|--------|--------|-------------|------------|------------------|
| Batch GD   | Slow   | High   | Smooth      | Low        | Small datasets   |
| SGD        | Fast   | Low    | Noisy       | High       | Large datasets   |
| Mini-batch | Medium | Medium | Balanced    | Low-Medium | Standard         |
| Momentum   | Fast   | Low    | Smooth      | Medium     | Standard + boost |
| Adam       | Fast   | Medium | Very Smooth | Very Low   | Deep learning    |

---

## Recommendation

**For Logistic Regression:**

1. **Start with:** Mini-batch gradient descent with batch_size=32, learning_rate=0.01
2. **If too slow:** Try Adam optimizer
3. **If diverging:** Reduce learning rate by 10x
4. **If oscillating:** Reduce learning rate by 2-3x or use momentum

---
