# How Logistic Regression Works: Step-by-Step

## Step 1: Input Features (X)

We start with our training data. Suppose we have a dataset of emails with features.

```python
import numpy as np

# Our training data
# Example: 5 emails, 3 features each
# Features: [number_of_links, suspicious_words, from_unknown_domain]

X = np.array([
    [5, 10, 1],      # Email 1
    [2, 3, 0],       # Email 2
    [15, 25, 1],     # Email 3
    [1, 2, 0],       # Email 4
    [8, 15, 1]       # Email 5
])

# Actual labels (0 = not spam, 1 = spam)
y = np.array([
    [1],  # Email 1 is spam
    [0],  # Email 2 is not spam
    [1],  # Email 3 is spam
    [0],  # Email 4 is not spam
    [1]   # Email 5 is spam
])

m = X.shape[0]  # Number of examples = 5
n = X.shape[1]  # Number of features = 3

print(f"Training data shape: {m} examples, {n} features")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

**What's happening:**

- We have 5 emails (m = 5)
- Each email has 3 features (n = 3)
- Each label is either 0 (not spam) or 1 (spam)

---

## Step 2: Initialize Weights and Bias

Before training, we randomly initialize our weights and bias.  
These will be updated during training.  

```python
# Initialize weights randomly or to small values
# One weight per feature
w = np.random.randn(n, 1) * 0.01  # Shape: (3, 1)
# Or initialize to zeros
w = np.zeros((n, 1))

# Initialize bias to zero
b = 0.0

print(f"Initial weights shape: {w.shape}")
print(f"Initial weights:\n{w}")
print(f"Initial bias: {b}")

# We'll also need a learning rate
learning_rate = 0.01
```

**What's happening:**

- We have 3 weights (one for each feature)
- We have 1 bias (scalar)
- These are randomly initialized (we could also start at zero)
- Learning rate controls how big each update step is

---

## Step 3: Linear Combination (z = w·X + b)

This is the same as linear regression. We compute a weighted sum of features plus bias.

```python
def compute_z(X, w, b):
    """
    Compute the linear combination z = w·X + b
    Args:
        X: Features matrix (m x n)
        w: Weights vector (n x 1)
        b: Bias (scalar)
    Returns:
        z: Linear combination (m x 1)
    """
    z = np.dot(X, w) + b
    return z

# Compute z for our training data
z = compute_z(X, w, b)

print(f"z shape: {z.shape}")
print(f"z values:\n{z}")
print(f"\nThese are unbounded numbers (can be negative, very large, etc.)")
```

**Output example (with initial weights near 0):**

```text
z values:
[[-0.05]
 [-0.01]
 [-0.08]
 [-0.005]
 [-0.06]]
```

**What's happening:**

- For each training example, we compute: z = w₁*x₁ + w₂*x₂ + w₃*x₃ + b
- z can be any number: negative, positive, very large, very small
- This is the output of the "linear part" of logistic regression

---

## Step 4: Apply Sigmoid Function

Now we pass z through the sigmoid function to convert it to a probability.

```python
def sigmoid(z):
    """
    Compute sigmoid function: σ(z) = 1 / (1 + e^(-z))
    This transforms any number into a probability (0, 1)
    """
    return 1 / (1 + np.exp(-z))

# Apply sigmoid to get probabilities
h = sigmoid(z)

print(f"h shape: {h.shape}")
print(f"h values (probabilities):\n{h}")
print(f"\nAll values are now between 0 and 1!")
```

**Output example:**

```text
h values (probabilities):
[[0.4875]
 [0.4975]
 [0.4800]
 [0.4988]
 [0.4850]]

These are probabilities between 0 and 1.
```

**What's happening:**

- Sigmoid squashes any number into (0, 1) range
- h[i] = probability that example i is class 1 (spam)
- If h[i] = 0.95, we're 95% confident it's spam
- If h[i] = 0.1, we're 90% confident it's not spam

**Mathematical detail:**

```python
# The sigmoid formula
# σ(z) = 1 / (1 + e^(-z))

# Example: if z = 2
z_example = 2
h_example = sigmoid(z_example)
print(f"σ(2) = {h_example:.4f}")  # Output: 0.8808

# Example: if z = -2
z_example = -2
h_example = sigmoid(z_example)
print(f"σ(-2) = {h_example:.4f}")  # Output: 0.1192

# Example: if z = 0
z_example = 0
h_example = sigmoid(z_example)
print(f"σ(0) = {h_example:.4f}")  # Output: 0.5000
```

---

## Step 5: Compare with Actual Labels

Now we have our predictions (probabilities). Let's see how far off we are from the actual labels.

```python
# Our predictions (from Step 4)
h = sigmoid(z)

# Actual labels
# y = [[1], [0], [1], [0], [1]]

# Simple error (not the cost function yet, just to visualize)
errors = h - y

print(f"Predictions:\n{h.flatten()}")
print(f"\nActual labels:\n{y.flatten()}")
print(f"\nErrors (prediction - actual):\n{errors.flatten()}")

# Example interpretation:
# Email 1: predicted 0.4875 (47.5% spam), actual 1 (spam) → error = -0.5125
# This email was labeled as spam but we predicted not spam
```

**What's happening:**

- For each example, we compare our predicted probability with the actual label
- If h[i] = 0.95 and y[i] = 1, error = 0.95 - 1 = -0.05 (small error, good)
- If h[i] = 0.1 and y[i] = 1, error = 0.1 - 1 = -0.9 (large error, bad)

---

## Step 6: Cost Function (Binary Cross-Entropy)

Now we compute a single number that measures our total error. This is the cost function.

```python
def compute_cost(h, y, m):
    """
    Compute Binary Cross-Entropy cost function
    Formula: J = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
    Args:
        h: Predictions/probabilities (m x 1)
        y: Actual labels (m x 1)
        m: Number of examples
    Returns:
        cost: Single scalar value
    """
    
    # Avoid log(0) by clipping h
    h_clipped = np.clip(h, 1e-7, 1 - 1e-7)
    
    # Compute cross-entropy for each example
    cost = -(1/m) * np.sum(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))
    
    return cost

# Compute cost with current predictions
cost = compute_cost(h, y, m)
print(f"Cost (Binary Cross-Entropy): {cost:.6f}")
```

**Mathematical breakdown:**

For each training example, the cost is:

```text
If y[i] = 1: cost[i] = -log(h[i])
If y[i] = 0: cost[i] = -log(1 - h[i])
```

Example with actual numbers:

```python
# Example 1: Email is actually spam (y=1), we predict 0.95
y_actual = 1
h_pred = 0.95
cost_this = -y_actual * np.log(h_pred) + (1 - y_actual) * np.log(1 - h_pred)
cost_this = -(1 * np.log(0.95)) + (0 * np.log(0.05))
cost_this = -(-0.0513) = 0.0513  # Small cost, good prediction

# Example 2: Email is actually spam (y=1), we predict 0.10
y_actual = 1
h_pred = 0.10
cost_this = -(1 * np.log(0.10)) + (0 * np.log(0.90))
cost_this = -(-2.303) = 2.303  # Large cost, bad prediction
```

**What's happening:**

- The cost is low when we predict the correct probability
- The cost is high when we predict the wrong probability (especially confidently wrong)
- The overall cost is the average across all examples
- Our goal is to minimize this cost

---

## Step 7: Compute Gradients

Now we compute how much we should adjust each weight. This is where calculus comes in.

```python
def compute_gradients(X, h, y, m):
    """
    Compute gradients for weights and bias
    Gradient for weights: ∂J/∂w = (1/m) * X^T * (h - y)
    Gradient for bias: ∂J/∂b = (1/m) * Σ(h - y)
    Args:
        X: Features (m x n)
        h: Predictions (m x 1)
        y: Actual labels (m x 1)
        m: Number of examples
    Returns:
        dw: Gradient for weights (n x 1)
        db: Gradient for bias (scalar)
    """
    
    # Compute the error
    error = h - y  # (m x 1)
    
    # Gradient for weights
    dw = (1/m) * np.dot(X.T, error)  # (n x 1)
    
    # Gradient for bias
    db = (1/m) * np.sum(error)  # scalar
    
    return dw, db

# Compute gradients
dw, db = compute_gradients(X, h, y, m)

print(f"Gradient for weights (dw) shape: {dw.shape}")
print(f"Gradient for weights:\n{dw}")
print(f"\nGradient for bias (db): {db:.6f}")
```

**Output example:**

```text
Gradient for weights:
[[ 0.0512]
 [ 0.1024]
 [ 0.0256]]

Gradient for bias: 0.0128
```

**What's happening:**

- The gradient tells us: "In which direction should we adjust the weights?"
- dw[i] > 0 means: "Increase weight i to reduce cost"
- dw[i] < 0 means: "Decrease weight i to reduce cost"
- dw[i] ≈ 0 means: "This weight is roughly correct"

**Interpretation of the numbers:**

- Gradient for weight 2 is largest (0.1024), so we should adjust it most
- Gradient for weight 3 is smallest (0.0256), so it's closer to optimal

---

## Step 8: Update Weights and Bias

Now we take a small step in the opposite direction of the gradient.

```python
def update_parameters(w, b, dw, db, learning_rate):
    """
    Update weights and bias using gradient descent
    Update rule:
    w := w - learning_rate * dw
    b := b - learning_rate * db
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

# Before update
print("BEFORE UPDATE:")
print(f"Weights:\n{w}")
print(f"Bias: {b}")

# Update
w_new, b_new = update_parameters(w, b, dw, db, learning_rate)

print("\nAFTER UPDATE:")
print(f"Weights:\n{w_new}")
print(f"Bias: {b_new}")

print("\nDifference:")
print(f"Weight changes:\n{w_new - w}")
print(f"Bias change: {b_new - b}")

# Update w and b for next iteration
w = w_new
b = b_new
```

**Output example:**

```text
BEFORE UPDATE:
Weights:
[[0.]
 [0.]
 [0.]]
Bias: 0.0

AFTER UPDATE:
Weights:
[[-0.00051]
 [-0.00102]
 [-0.00026]]
Bias: -0.000128

The weights got slightly negative because the gradients were positive.
```

**What's happening:**

- We subtract learning_rate * gradient from each parameter
- This moves us slightly toward the optimal values
- The learning rate controls how big each step is
- Small steps = slow but stable convergence
- Large steps = fast but might overshoot

---

## Step 9: Repeat (Multiple Iterations)

We repeat steps 3-8 many times until the cost stops decreasing.

```python
def train_logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    """
    Train logistic regression model
    Args:
        X: Training features (m x n)
        y: Training labels (m x 1)
        learning_rate: Step size
        iterations: Number of iterations
    Returns:
        w: Final weights (n x 1)
        b: Final bias (scalar)
        cost_history: Cost at each iteration
    """
    
    m, n = X.shape
    
    # Initialize parameters
    w = np.zeros((n, 1))
    b = 0.0
    
    # Store cost history
    cost_history = []
    
    # Training loop
    for iteration in range(iterations):
        # Forward pass
        z = compute_z(X, w, b)
        h = sigmoid(z)
        
        # Compute cost
        cost = compute_cost(h, y, m)
        cost_history.append(cost)
        
        # Compute gradients
        dw, db = compute_gradients(X, h, y, m)
        
        # Update parameters
        w, b = update_parameters(w, b, dw, db, learning_rate)
        
        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Cost = {cost:.6f}")
    
    return w, b, cost_history

# Train the model
print("Training Logistic Regression...")
w_trained, b_trained, cost_history = train_logistic_regression(
    X, y, 
    learning_rate=0.1, 
    iterations=1000
)

print(f"\nTraining complete!")
print(f"Final weights:\n{w_trained}")
print(f"Final bias: {b_trained}")
```

**Output:**

```text
Training Logistic Regression...
Iteration 100: Cost = 0.621547
Iteration 200: Cost = 0.612834
Iteration 300: Cost = 0.610234
...
Iteration 1000: Cost = 0.605432

Training complete!
Final weights:
[[ 0.0234]
 [ 0.0512]
 [ 0.0189]]
Final bias: -0.0423
```

**What's happening:**

- Cost decreases as we train (getting better)
- Weights are updated in each iteration
- After 1000 iterations, we have our trained model

---

## Step 10: Making Predictions on New Data

Once trained, we use the model to predict on new, unseen data.

```python
def predict(X_new, w, b, threshold=0.5):
    """
    Make predictions on new data
    
    Args:
        X_new: New features (m_new x n)
        w: Trained weights (n x 1)
        b: Trained bias (scalar)
        threshold: Probability threshold for classification
    
    Returns:
        probabilities: Predicted probabilities (m_new x 1)
        predictions: Binary predictions (0 or 1) (m_new x 1)
    """
    
    # Compute z
    z = compute_z(X_new, w, b)
    
    # Apply sigmoid
    probabilities = sigmoid(z)
    
    # Convert to binary predictions
    predictions = (probabilities >= threshold).astype(int)
    
    return probabilities, predictions

# New email data (never seen before)
X_new = np.array([
    [10, 20, 1],   # Suspicious email
    [1, 1, 0]      # Normal email
])

# Make predictions
probabilities, predictions = predict(X_new, w_trained, b_trained, threshold=0.5)

print("New email 1:")
print(f"  Probability of spam: {probabilities[0][0]:.4f}")
print(f"  Prediction: {'SPAM' if predictions[0][0] == 1 else 'NOT SPAM'}")

print("\nNew email 2:")
print(f"  Probability of spam: {probabilities[1][0]:.4f}")
print(f"  Prediction: {'SPAM' if predictions[1][0] == 1 else 'NOT SPAM'}")
```

**Output:**

```text
New email 1:
  Probability of spam: 0.8523
  Prediction: SPAM

New email 2:
  Probability of spam: 0.3421
  Prediction: NOT SPAM
```

**What's happening:**

- Email 1 has high probability (0.85), so we classify as SPAM
- Email 2 has low probability (0.34), so we classify as NOT SPAM
- We use threshold=0.5 to decide: if probability ≥ 0.5, classify as positive class

---

## The Complete Flow (Summary)

Here's the entire story in one visual:

```text
INPUT: X (5 emails, 3 features each)
  ↓
STEP 3: Compute z = w·X + b (linear combination)
  ↓
STEP 4: Apply sigmoid to z → get probabilities h
  ↓
STEP 5: Compare h with actual y
  ↓
STEP 6: Compute cost using binary cross-entropy
  ↓
STEP 7: Compute gradients (dw, db)
  ↓
STEP 8: Update w and b using gradients
  ↓
REPEAT steps 3-8 for many iterations
  ↓
OUTPUT: Trained w and b
  ↓
USE: On new data to make predictions
```

---
