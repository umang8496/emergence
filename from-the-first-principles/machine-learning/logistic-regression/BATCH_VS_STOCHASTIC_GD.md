# Batch Gradient Descent vs Stochastic Gradient Descent

## Part 1: The Mathematics

### Batch Gradient Descent (BGD)

#### Update Rule

For each iteration:

```text
Compute gradient using ALL m training examples:
∂J/∂w = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ) * xᵢ

Update parameters:
w := w - α * ∂J/∂w
b := b - α * ∂J/∂b
```

Where:

- `α` = learning rate
- `m` = number of training examples
- `hᵢ` = prediction for example i
- `yᵢ` = actual label for example i
- `xᵢ` = features for example i

#### Key Property

The gradient is **exact**. We compute it using all the data, so it points directly toward the minimum.

#### Cost Function

```text
J(w, b) = -(1/m) * Σᵢ₌₁ᵐ [yᵢ * log(hᵢ) + (1-yᵢ) * log(1-hᵢ)]
```

Cost is computed over **all m examples**.

---

### Stochastic Gradient Descent (SGD)

#### Update Rule

For each iteration:

```text
Pick ONE random example i (out of m examples):

Compute gradient using ONLY that example:
∂J/∂w ≈ (hᵢ - yᵢ) * xᵢ

Update parameters:
w := w - α * ∂J/∂w
b := b - α * ∂J/∂b
```

Where:

- We compute gradient from only **one example** (not averaged over m)
- We pick a **different random example** each iteration

#### Why "Stochastic"?

"Stochastic" means "random".  
We're randomly selecting examples, so the gradient is noisy (not exact).  

The gradient estimate:

```text
∂J/∂w ≈ (hᵢ - yᵢ) * xᵢ
```

Is an **unbiased estimate** of the true gradient:

```text
E[∂J/∂w] = true_gradient
```

But individual examples create noise.

#### Expected Value Analysis

For SGD, the expected gradient is:

```text
E[(hᵢ - yᵢ) * xᵢ] = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ) * xᵢ = True gradient

But the variance is:
Var[(hᵢ - yᵢ) * xᵢ] = E[(hᵢ - yᵢ)² * xᵢ²] - [E[(hᵢ - yᵢ) * xᵢ]]²
                    ≈ (1/m) * E[error²] * E[x²]
```

This variance causes the noise (bouncing around the minimum).

---

### Mini-Batch Gradient Descent

#### Update Rule

```text
Pick B random examples (batch size, e.g., B=32):

Compute gradient using the batch:
∂J/∂w = (1/B) * Σᵢ₌₁ᴮ (hᵢ - yᵢ) * xᵢ

Update parameters:
w := w - α * ∂J/∂w
b := b - α * ∂J/∂b
```

#### Variance Reduction

Mini-batch reduces variance compared to SGD:

```text
Var[mini-batch gradient] = Var[full SGD] / B
Example:
- SGD variance: 100
- Mini-batch (B=32) variance: 100/32 ≈ 3.1 (more stable)
- Batch GD (B=m) variance: 100/m (most stable)
```

This is why mini-batch is the "Goldilocks" approach.

---

## Part 2: Complete Python Implementation

### Core Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def sigmoid(z):
    """Sigmoid activation function"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost(y_true, y_pred):
    """Binary cross-entropy cost"""
    m = len(y_true)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    return cost


class GradientDescentComparison:
    """Compare different gradient descent variants"""
    def __init__(self, learning_rate=0.01, verbose=True):
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.history = {}
    
    # =====================================================================
    # BATCH GRADIENT DESCENT
    # =====================================================================
    
    def batch_gradient_descent(self, X, y, iterations=1000, verbose_interval=100):
        """
        Batch Gradient Descent: Uses ALL examples per iteration
        Args:
            X: Features (m x n)
            y: Labels (m x 1)
            iterations: Number of iterations
            verbose_interval: Print every N iterations
        Returns:
            w: Final weights
            b: Final bias
            history: Cost history, gradient norms, time per iteration
        """
        m, n = X.shape
        w = np.zeros((n, 1))
        b = 0.0
        
        cost_history = []
        gradient_norms = []
        iteration_times = []
        
        import time
        
        for iteration in range(iterations):
            start_time = time.time()
            
            # Forward pass (ALL examples)
            z = np.dot(X, w) + b
            h = sigmoid(z)
            
            # Compute cost (ALL examples)
            cost = compute_cost(y, h)
            cost_history.append(cost)
            
            # Compute gradient (ALL examples)
            error = h - y
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)
            gradient_norm = np.linalg.norm(dw)
            gradient_norms.append(gradient_norm)
            
            # Update parameters
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            
            iteration_times.append(time.time() - start_time)
            
            # Progress
            if self.verbose and (iteration + 1) % verbose_interval == 0:
                print(f"Iteration {iteration + 1}: Cost = {cost:.6f}, "
                      f"Gradient norm = {gradient_norm:.6f}")
        
        self.history['batch_gd'] = {
            'costs': cost_history,
            'gradient_norms': gradient_norms,
            'times': iteration_times
        }
        
        return w, b, cost_history
    
    # =====================================================================
    # STOCHASTIC GRADIENT DESCENT
    # =====================================================================
    
    def stochastic_gradient_descent(self, X, y, epochs=10,  learning_rate_decay=None, verbose_interval=100):
        """
        Stochastic Gradient Descent: Uses ONE random example per iteration
        Args:
            X: Features (m x n)
            y: Labels (m x 1)
            epochs: Number of passes through entire dataset
            learning_rate_decay: Function to decay learning rate (e.g., lambda t: 0.01/(1+0.01*t))
            verbose_interval: Print every N iterations
        Returns:
            w: Final weights
            b: Final bias
            history: Cost history, gradient norms, time per iteration
        """
        m, n = X.shape
        w = np.zeros((n, 1))
        b = 0.0
        
        cost_history = []
        gradient_norms = []
        iteration_times = []
        
        import time
        
        iteration = 0
        
        for epoch in range(epochs):
            # Shuffle data
            shuffle_idx = np.random.permutation(m)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]
            
            for i in range(m):  # Process each example
                start_time = time.time()
                
                # Single example
                x_i = X_shuffled[i:i+1]
                y_i = y_shuffled[i:i+1]
                
                # Forward pass (ONE example)
                z_i = np.dot(x_i, w) + b
                h_i = sigmoid(z_i)
                
                # Gradient (ONE example)
                error_i = h_i - y_i
                dw = error_i * x_i.T
                db = error_i
                gradient_norm = np.linalg.norm(dw)
                gradient_norms.append(gradient_norm)
                
                # Decay learning rate (optional)
                current_lr = self.learning_rate
                if learning_rate_decay:
                    current_lr = learning_rate_decay(iteration)
                
                # Update parameters
                w -= current_lr * dw
                b -= current_lr * db
                
                iteration_times.append(time.time() - start_time)
                iteration += 1
            
            # Compute full cost for monitoring (expensive, but informative)
            h_full = sigmoid(np.dot(X, w) + b)
            cost = compute_cost(y, h_full)
            cost_history.append(cost)
            
            if self.verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}: Cost = {cost:.6f}")
        
        self.history['sgd'] = {
            'costs': cost_history,
            'gradient_norms': gradient_norms,
            'times': iteration_times
        }
        
        return w, b, cost_history
    
    # =====================================================================
    # MINI-BATCH GRADIENT DESCENT
    # =====================================================================
    
    def minibatch_gradient_descent(self, X, y, epochs=10, batch_size=32, learning_rate_decay=None, verbose_interval=10):
        """
        Mini-Batch Gradient Descent: Uses B examples per iteration
        Args:
            X: Features (m x n)
            y: Labels (m x 1)
            epochs: Number of passes through entire dataset
            batch_size: Number of examples per batch
            learning_rate_decay: Function to decay learning rate
            verbose_interval: Print every N iterations
        Returns:
            w: Final weights
            b: Final bias
            history: Cost history, gradient norms, time per iteration
        """
        m, n = X.shape
        w = np.zeros((n, 1))
        b = 0.0
        
        cost_history = []
        gradient_norms = []
        iteration_times = []
        
        import time
        
        iteration = 0
        
        for epoch in range(epochs):
            # Shuffle data
            shuffle_idx = np.random.permutation(m)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]
            
            # Process mini-batches
            for i in range(0, m, batch_size):
                start_time = time.time()
                
                # Mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                batch_m = len(X_batch)
                
                # Forward pass (BATCH examples)
                z_batch = np.dot(X_batch, w) + b
                h_batch = sigmoid(z_batch)
                
                # Gradient (BATCH examples)
                error_batch = h_batch - y_batch
                dw = (1/batch_m) * np.dot(X_batch.T, error_batch)
                db = (1/batch_m) * np.sum(error_batch)
                gradient_norm = np.linalg.norm(dw)
                gradient_norms.append(gradient_norm)
                
                # Decay learning rate
                current_lr = self.learning_rate
                if learning_rate_decay:
                    current_lr = learning_rate_decay(iteration)
                
                # Update parameters
                w -= current_lr * dw
                b -= current_lr * db
                
                iteration_times.append(time.time() - start_time)
                iteration += 1
            
            # Compute full cost for monitoring
            h_full = sigmoid(np.dot(X, w) + b)
            cost = compute_cost(y, h_full)
            cost_history.append(cost)
            
            if self.verbose and (epoch + 1) % verbose_interval == 0:
                print(f"Epoch {epoch + 1}: Cost = {cost:.6f}")
        
        self.history['minibatch_gd'] = {
            'costs': cost_history,
            'gradient_norms': gradient_norms,
            'times': iteration_times
        }
        
        return w, b, cost_history


class GradientDescentVisualizer:
    """Create visualizations comparing gradient descent methods"""
    
    @staticmethod
    def plot_cost_comparison(histories, title="Cost Function Comparison"):
        """Plot cost curves for different methods"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cost curve
        ax = axes[0]
        for method, history in histories.items():
            costs = history['costs']
            ax.plot(costs, label=method, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration/Epoch', fontsize=12)
        ax.set_ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Zoomed in (last 50%)
        ax = axes[1]
        for method, history in histories.items():
            costs = history['costs']
            start_idx = len(costs) // 2
            ax.plot(range(start_idx, len(costs)), costs[start_idx:], 
                   label=method, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration/Epoch', fontsize=12)
        ax.set_ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
        ax.set_title('Cost Curve (Zoomed - Last 50%)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cost_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_gradient_norm_comparison(histories):
        """Plot gradient norm evolution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method, history in histories.items():
            norms = history['gradient_norms']
            # Plot every 10th value for clarity (too noisy otherwise)
            if len(norms) > 1000:
                ax.plot(range(0, len(norms), 10), norms[::10], 
                       label=method, linewidth=2, alpha=0.8)
            else:
                ax.plot(norms, label=method, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norm Evolution', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gradient_norm_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_time_comparison(histories):
        """Plot cumulative time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method, history in histories.items():
            times = np.array(history['times'])
            cumsum_times = np.cumsum(times)
            
            # Normalize to number of iterations processed
            iterations = np.arange(len(cumsum_times))
            
            ax.plot(iterations, cumsum_times, label=method, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Number of Updates', fontsize=12)
        ax.set_ylabel('Cumulative Time (seconds)', fontsize=12)
        ax.set_title('Time Complexity Comparison', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()


```

---

## Part 3: Complete Demonstration

```python
def main():
    """
    Full demonstration comparing Batch GD, SGD, and Mini-batch GD
    """
    
    print("=" * 80)
    print("GRADIENT DESCENT COMPARISON: BATCH vs STOCHASTIC vs MINI-BATCH")
    print("=" * 80)
    
    # =====================================================================
    # STEP 1: CREATE SYNTHETIC DATASET
    # =====================================================================
    print("\n1. CREATING SYNTHETIC BINARY CLASSIFICATION DATASET")
    print("-" * 80)
    
    # Create a reasonable dataset
    X, y = make_classification(
        n_samples=1000,      # Number of examples
        n_features=20,       # Number of features
        n_informative=10,    # Number of useful features
        n_redundant=5,       # Correlated features
        random_state=42
    )
    y = y.reshape(-1, 1)  # Reshape to column vector
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} examples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} examples")
    print(f"Feature distribution: mean ≈ 0, std ≈ 1 (normalized)")
    
    # =====================================================================
    # STEP 2: TRAIN WITH BATCH GRADIENT DESCENT
    # =====================================================================
    print("\n2. TRAINING WITH BATCH GRADIENT DESCENT")
    print("-" * 80)
    print("Uses ALL 800 examples per iteration")
    print("Expected: Smooth convergence, slow per iteration\n")
    
    gd = GradientDescentComparison(learning_rate=0.1, verbose=True)
    w_batch, b_batch, cost_batch = gd.batch_gradient_descent(
        X_train, y_train, iterations=200, verbose_interval=20
    )
    
    # Evaluate
    h_test = sigmoid(np.dot(X_test, w_batch) + b_batch)
    accuracy_batch = ((h_test >= 0.5) == y_test).mean()
    print(f"\nBatch GD Test Accuracy: {accuracy_batch:.4f}")
    
    # =====================================================================
    # STEP 3: TRAIN WITH STOCHASTIC GRADIENT DESCENT
    # =====================================================================
    print("\n3. TRAINING WITH STOCHASTIC GRADIENT DESCENT")
    print("-" * 80)
    print("Uses 1 random example per iteration")
    print("Expected: Noisy convergence, fast per iteration")
    print("Using learning rate decay: α(t) = 0.1 / (1 + 0.01*t)\n")
    
    # Learning rate decay function
    def lr_decay_sgd(t):
        return 0.1 / (1 + 0.01 * t)
    
    gd_sgd = GradientDescentComparison(learning_rate=0.1, verbose=True)
    w_sgd, b_sgd, cost_sgd = gd_sgd.stochastic_gradient_descent(
        X_train, y_train, epochs=1,  # 1 epoch = 800 updates (one per example)
        learning_rate_decay=lr_decay_sgd,
        verbose_interval=200  # Print every 200 examples
    )
    
    # Evaluate
    h_test = sigmoid(np.dot(X_test, w_sgd) + b_sgd)
    accuracy_sgd = ((h_test >= 0.5) == y_test).mean()
    print(f"\nSGD Test Accuracy: {accuracy_sgd:.4f}")
    
    # =====================================================================
    # STEP 4: TRAIN WITH MINI-BATCH GRADIENT DESCENT
    # =====================================================================
    print("\n4. TRAINING WITH MINI-BATCH GRADIENT DESCENT")
    print("-" * 80)
    print("Uses 32 examples per iteration")
    print("Expected: Balanced convergence, reasonable speed\n")
    
    def lr_decay_minibatch(t):
        return 0.1 / (1 + 0.001 * t)
    
    gd_minibatch = GradientDescentComparison(learning_rate=0.1, verbose=True)
    w_minibatch, b_minibatch, cost_minibatch = gd_minibatch.minibatch_gradient_descent(
        X_train, y_train, epochs=25, batch_size=32,
        learning_rate_decay=lr_decay_minibatch,
        verbose_interval=5
    )
    
    # Evaluate
    h_test = sigmoid(np.dot(X_test, w_minibatch) + b_minibatch)
    accuracy_minibatch = ((h_test >= 0.5) == y_test).mean()
    print(f"\nMini-batch GD Test Accuracy: {accuracy_minibatch:.4f}")
    
    # =====================================================================
    # STEP 5: VISUALIZE RESULTS
    # =====================================================================
    print("\n5. VISUALIZING COMPARISONS")
    print("-" * 80)
    
    # Prepare histories for visualization
    histories = {
        'Batch GD': gd.history['batch_gd'],
        'SGD (with decay)': gd_sgd.history['sgd'],
        'Mini-batch GD': gd_minibatch.history['minibatch_gd']
    }
    
    # Plot comparisons
    print("Generating visualizations...")
    
    GradientDescentVisualizer.plot_cost_comparison(
        histories, 
        title="Cost Function Convergence"
    )
    
    GradientDescentVisualizer.plot_gradient_norm_comparison(histories)
    
    GradientDescentVisualizer.plot_time_comparison(histories)
    
    # =====================================================================
    # STEP 6: SUMMARY
    # =====================================================================
    print("\n6. SUMMARY STATISTICS")
    print("-" * 80)
    
    print("\nBatch Gradient Descent:")
    print(f"  - Iterations: 200")
    print(f"  - Final cost: {gd.history['batch_gd']['costs'][-1]:.6f}")
    print(f"  - Test accuracy: {accuracy_batch:.4f}")
    print(f"  - Total time: {sum(gd.history['batch_gd']['times']):.4f} seconds")
    print(f"  - Convergence: Smooth (cost monotonically decreases)")
    
    print("\nStochastic Gradient Descent:")
    print(f"  - Iterations: {len(gd_sgd.history['sgd']['costs'])}")
    print(f"  - Final cost: {gd_sgd.history['sgd']['costs'][-1]:.6f}")
    print(f"  - Test accuracy: {accuracy_sgd:.4f}")
    print(f"  - Total time: {sum(gd_sgd.history['sgd']['times']):.4f} seconds")
    print(f"  - Convergence: Noisy (zigzags but general downward trend)")
    
    print("\nMini-Batch Gradient Descent:")
    print(f"  - Iterations: {len(gd_minibatch.history['minibatch_gd']['costs'])}")
    print(f"  - Final cost: {gd_minibatch.history['minibatch_gd']['costs'][-1]:.6f}")
    print(f"  - Test accuracy: {accuracy_minibatch:.4f}")
    print(f"  - Total time: {sum(gd_minibatch.history['minibatch_gd']['times']):.4f} seconds")
    print(f"  - Convergence: Balanced (smooth with some noise)")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("""
1. CONVERGENCE BEHAVIOR:
   - Batch GD: Smooth descent to minimum (exact gradients)
   - SGD: Noisy, zigzag path but faster per update
   - Mini-batch: In between (smooth but with some noise)

2. COMPUTATIONAL EFFICIENCY:
   - Batch GD: Slowest per update (processes all examples)
   - SGD: Fastest per update (processes one example)
   - Mini-batch: Medium per update (processes batch_size examples)

3. MEMORY USAGE:
   - Batch GD: High (must load all training data)
   - SGD: Low (load one example at a time)
   - Mini-batch: Low-Medium (load batch_size examples)

4. FINAL ACCURACY:
   - All three methods should reach similar accuracy
   - The path is different, but destination is same for convex problems

5. WHEN TO USE:
   - Batch GD: Small datasets (< 10K examples)
   - SGD: Massive datasets (millions+), online learning
   - Mini-batch: Most practical (10K - 1M examples)
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
```

---

## Part 4: Detailed Comparison Table

| Aspect                         | Batch GD             | SGD                     | Mini-Batch GD       |
|--------------------------------|----------------------|-------------------------|---------------------|
| **Examples per update**        | All (m)              | One (1)                 | Batch (32)          |
| **Gradient exactness**         | Exact                | Approximate             | Approximate         |
| **Convergence path**           | Smooth, direct       | Noisy, zigzag           | Balanced            |
| **Time per iteration**         | Very slow            | Very fast               | Fast                |
| **Memory required**            | High (load all m)    | Low (load 1)            | Low-Med (load B)    |
| **Updates per epoch**          | 1                    | m                       | m/B                 |
| **Variance of gradient**       | 0                    | High                    | Medium              |
| **Guaranteed convergence**     | Yes                  | Yes*                    | Yes                 |
| **Final solution quality**     | Same                 | Same                    | Same                |
| **Best for**                   | Research, small data | Streaming, massive data | Production ML       |
| **Learning rate decay needed** | No                   | Yes                     | Sometimes           |
| **Parallelizable**             | No                   | No                      | Yes (across batch)  |

With learning rate decay.

---

## Part 5: Mathematical Analysis

### Convergence Rate Analysis

#### Batch Gradient Descent

For convex loss functions (like logistic regression):

```text
Cost after k iterations: J(k) - J* ≤ O(1/k)

Where J* is the optimal cost

This means:
- Cost decreases linearly (in terms of iterations)
- After 100 iterations: Cost ≈ initial + constant/100
- After 1000 iterations: Cost ≈ initial + constant/1000
```

#### Stochastic Gradient Descent

```text
Cost after k iterations: E[J(k)] - J* ≤ O(1/√k)

This means:
- Cost decreases more slowly (in terms of iterations)
- After 100 iterations: Cost ≈ initial + constant/√100 = initial + constant/10
- After 1000 iterations: Cost ≈ initial + constant/√1000 ≈ initial + constant/31.6

BUT: Each iteration is m times faster!

So in terms of examples seen:
- Batch GD after k iterations: Processed k*m examples
- SGD after k*m iterations: Processed k*m examples
- SGD converges in roughly same wall-clock time!
```

#### Why SGD Converges Despite Noise?

The key theorem (from stochastic optimization):

```text
If learning rate decay satisfies:
  Σₜ αₜ = ∞  (total learning rate is infinite)
  Σₜ αₜ² < ∞  (total learning rate squared is finite)

Example: αₜ = 1/(1 + t)
  Σₜ 1/(1+t) = ∞ ✓
  Σₜ 1/(1+t)² < ∞ ✓

Then SGD converges to the minimum!
```

This is why SGD needs learning rate decay.

---

## Part 6: When To Use Each Method

### Use Batch Gradient Descent When

```text
1. Dataset is small (< 10,000 examples)
2. You need guaranteed smooth convergence (for visualization)
3. You're doing research/education
4. Computational resources are abundant (GPU, parallel processing)
5. Full gradient information is needed (regularization with full regularizer)

Example: 5,000 customer records
```

### Use Stochastic Gradient Descent When

```text
1. Dataset is massive (millions of examples)
2. Data arrives in streams (online learning)
3. Memory is limited
4. You need to process data from disk (can't load all in memory)
5. Training time is critical (faster per update)

Example: 1 billion user interactions arriving continuously
```

### Use Mini-Batch Gradient Descent When

```text
1. Dataset is medium to large (10,000 - 1,000,000 examples)
2. You have GPU (vectorized operations on batches are efficient)
3. You need balanced convergence (not too noisy, not too slow)
4. This is production ML (standard in industry)

Example: 100,000 customer transactions, training on GPU
```

---

## Part 7: Learning Rate Decay Strategies

### SGD Without Decay

```python
for iteration in range(1000):
    gradient = compute_gradient(single_example)
    w -= 0.1 * gradient  # Constant learning rate
```

**Problem:**

```text
Iteration 1: Large steps toward minimum
Iteration 100: Bouncing around minimum (can't settle)
Iteration 1000: Still bouncing (noise is too large)
```

### SGD With Decay - Strategy 1: 1/t Decay

```python
def lr_decay(t):
    return initial_lr / (1 + decay_rate * t)

# Example
for iteration in range(1000):
    alpha = 0.1 / (1 + 0.01 * iteration)
    gradient = compute_gradient(single_example)
    w -= alpha * gradient
```

**Effect:**

```text
Iteration 1: α = 0.1 / 1.01 ≈ 0.099 (large step)
Iteration 100: α = 0.1 / 2 = 0.05 (smaller step)
Iteration 1000: α = 0.1 / 11 ≈ 0.009 (tiny step)
```

### SGD With Decay - Strategy 2: Exponential Decay

```python
def lr_decay(t):
    return initial_lr * (decay_factor ** t)

# Example
for iteration in range(1000):
    alpha = 0.1 * (0.999 ** iteration)
    gradient = compute_gradient(single_example)
    w -= alpha * gradient
```

**Effect:**

```text
Iteration 0: α = 0.1 (full learning rate)
Iteration 100: α = 0.1 * 0.999^100 ≈ 0.0905
Iteration 1000: α = 0.1 * 0.999^1000 ≈ 0.0368
```

### SGD With Decay - Strategy 3: Step Decay

```python
# Decay every N iterations/epochs
if (iteration + 1) % 100 == 0:
    learning_rate *= 0.1  # Multiply by 0.1 every 100 iterations

for iteration in range(1000):
    gradient = compute_gradient(single_example)
    w -= learning_rate * gradient
```

**Effect:**

```text
Iterations 0-99: α = 0.1
Iterations 100-199: α = 0.01
Iterations 200-299: α = 0.001
Iterations 300+: α = 0.0001
```

---

## Part 8: Expected Output and Graphs

When you run the complete code, you'll see:

### Cost Curves

```text
Batch GD:
- Smooth, monotonic decrease
- Reaches minimum in ~50 iterations
- No bouncing around minimum

SGD:
- Noisy, zigzag pattern
- Roughly decreasing trend
- Some bouncing even with decay
- Needs learning rate decay to settle

Mini-batch:
- Smooth with slight noise
- Reaches minimum in ~15 epochs
- Good balance of both worlds
```

### Gradient Norm

```text
Batch GD: Decreases smoothly to near zero
SGD: Large spikes (noisy updates)
Mini-batch: Moderate spikes (averaged over batch)
```

### Time Comparison

```text
Batch GD: Linear increase (same time per iteration)
SGD: Steeper slope (faster per iteration, more iterations)
Mini-batch: In between
```

---

## Summary

- **Batch GD:** Slow but predictable, smooth convergence
- **SGD:** Fast but noisy, needs learning rate decay
- **Mini-batch:** Best of both worlds (used in practice)

The choice depends on your dataset size and computational resources.

---
