"""
Batch vs Stochastic Gradient Descent: Complete Demonstration
============================================================

This script demonstrates and compares three gradient descent variants:
1. Batch Gradient Descent (BGD)
2. Stochastic Gradient Descent (SGD)
3. Mini-Batch Gradient Descent (Mini-batch GD)

It includes visualizations of convergence behavior, gradient norms, and timing.

Run this file directly: python batch_vs_stochastic_gradient_descent.py
"""

import numpy as np    # type: ignore
import matplotlib.pyplot as plt    # type: ignore
from sklearn.datasets import make_classification    # type: ignore
from sklearn.model_selection import train_test_split    # type: ignore
import time


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sigmoid(z):
    """Sigmoid activation function"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost(y_true, y_pred):
    """Binary cross-entropy cost"""
    m = len(y_true)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    cost = -(1/m) * np.sum(y_true * np.log(y_pred) + 
                           (1-y_true) * np.log(1-y_pred))
    return cost


# ============================================================================
# GRADIENT DESCENT IMPLEMENTATIONS
# ============================================================================

class BatchGradientDescent:
    """Batch Gradient Descent: Uses ALL examples per iteration"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def train(self, X, y, iterations=1000, verbose_interval=100):
        """
        Train using Batch Gradient Descent
        
        Args:
            X: Features (m x n)
            y: Labels (m x 1)
            iterations: Number of iterations
            verbose_interval: Print every N iterations
        
        Returns:
            w: Final weights
            b: Final bias
            history: Dictionary containing costs, gradients, times
        """
        m, n = X.shape
        w = np.zeros((n, 1))
        b = 0.0
        
        history = {
            'costs': [],
            'gradient_norms': [],
            'times': [],
            'iterations': []
        }
        
        print("Training Batch Gradient Descent...")
        print(f"Dataset: {m} examples, {n} features")
        print(f"Updates per iteration: {m} (ALL examples)")
        print()
        
        for iteration in range(iterations):
            start_time = time.time()
            
            # Forward pass (ALL examples)
            z = np.dot(X, w) + b
            h = sigmoid(z)
            
            # Compute cost
            cost = compute_cost(y, h)
            history['costs'].append(cost)
            
            # Compute gradient (ALL examples)
            error = h - y
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)
            
            # Gradient norm
            gradient_norm = np.linalg.norm(dw)
            history['gradient_norms'].append(gradient_norm)
            
            # Update parameters
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            
            # Timing
            elapsed = time.time() - start_time
            history['times'].append(elapsed)
            history['iterations'].append(iteration + 1)
            
            # Progress
            if (iteration + 1) % verbose_interval == 0:
                print(f"Iteration {iteration + 1:4d}: Cost = {cost:.6f}, "
                      f"Gradient norm = {gradient_norm:.6f}, "
                      f"Time = {elapsed:.4f}s")
        
        print()
        return w, b, history


class StochasticGradientDescent:
    """Stochastic Gradient Descent: Uses ONE random example per iteration"""
    
    def __init__(self, learning_rate=0.01, decay_rate=0.01):
        self.initial_learning_rate = learning_rate
        self.decay_rate = decay_rate
    
    def train(self, X, y, epochs=10, verbose_interval=200):
        """
        Train using Stochastic Gradient Descent
        
        Args:
            X: Features (m x n)
            y: Labels (m x 1)
            epochs: Number of passes through entire dataset
            verbose_interval: Print every N examples
        
        Returns:
            w: Final weights
            b: Final bias
            history: Dictionary containing costs, gradients, times
        """
        m, n = X.shape
        w = np.zeros((n, 1))
        b = 0.0
        
        history = {
            'costs': [],
            'gradient_norms': [],
            'times': [],
            'iterations': []
        }
        
        print("Training Stochastic Gradient Descent...")
        print(f"Dataset: {m} examples, {n} features")
        print(f"Updates per iteration: 1 (ONE random example)")
        print(f"Total iterations: {epochs} epochs × {m} examples = {epochs * m} updates")
        print(f"Learning rate decay: α(t) = {self.initial_learning_rate} / (1 + {self.decay_rate}*t)")
        print()
        
        iteration = 0
        
        for epoch in range(epochs):
            # Shuffle data
            shuffle_idx = np.random.permutation(m)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]
            
            for i in range(m):
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
                
                # Gradient norm
                gradient_norm = np.linalg.norm(dw)
                history['gradient_norms'].append(gradient_norm)
                
                # Learning rate with decay
                learning_rate = self.initial_learning_rate / (1 + self.decay_rate * iteration)
                
                # Update parameters
                w -= learning_rate * dw
                b -= learning_rate * db
                
                # Timing
                elapsed = time.time() - start_time
                history['times'].append(elapsed)
                
                iteration += 1
            
            # Compute cost on full dataset for monitoring
            h_full = sigmoid(np.dot(X, w) + b)
            cost = compute_cost(y, h_full)
            history['costs'].append(cost)
            history['iterations'].append(epoch + 1)
            
            print(f"Epoch {epoch + 1}: Cost = {cost:.6f}, "
                  f"Learning rate = {self.initial_learning_rate / (1 + self.decay_rate * iteration):.6f}")
        
        print()
        return w, b, history


class MiniBatchGradientDescent:
    """Mini-Batch Gradient Descent: Uses B examples per iteration"""
    
    def __init__(self, learning_rate=0.01, decay_rate=0.001):
        self.initial_learning_rate = learning_rate
        self.decay_rate = decay_rate
    
    def train(self, X, y, epochs=10, batch_size=32, verbose_interval=5):
        """
        Train using Mini-Batch Gradient Descent
        
        Args:
            X: Features (m x n)
            y: Labels (m x 1)
            epochs: Number of passes through entire dataset
            batch_size: Number of examples per batch
            verbose_interval: Print every N epochs
        
        Returns:
            w: Final weights
            b: Final bias
            history: Dictionary containing costs, gradients, times
        """
        m, n = X.shape
        w = np.zeros((n, 1))
        b = 0.0
        
        history = {
            'costs': [],
            'gradient_norms': [],
            'times': [],
            'iterations': []
        }
        
        print("Training Mini-Batch Gradient Descent...")
        print(f"Dataset: {m} examples, {n} features")
        print(f"Updates per iteration: {batch_size} examples")
        print(f"Batches per epoch: {m // batch_size}")
        print(f"Total epochs: {epochs}")
        print(f"Learning rate decay: α(t) = {self.initial_learning_rate} / (1 + {self.decay_rate}*t)")
        print()
        
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
                
                # Gradient norm
                gradient_norm = np.linalg.norm(dw)
                history['gradient_norms'].append(gradient_norm)
                
                # Learning rate with decay
                learning_rate = self.initial_learning_rate / (1 + self.decay_rate * iteration)
                
                # Update parameters
                w -= learning_rate * dw
                b -= learning_rate * db
                
                # Timing
                elapsed = time.time() - start_time
                history['times'].append(elapsed)
                
                iteration += 1
            
            # Compute cost on full dataset
            h_full = sigmoid(np.dot(X, w) + b)
            cost = compute_cost(y, h_full)
            history['costs'].append(cost)
            history['iterations'].append(epoch + 1)
            
            if (epoch + 1) % verbose_interval == 0:
                print(f"Epoch {epoch + 1}: Cost = {cost:.6f}")
        
        print()
        return w, b, history


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cost_comparison(histories):
    """Plot cost curves for different methods"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Full cost curve
    ax = axes[0]
    for method, history in histories.items():
        if method == 'Batch GD':
            x = history['iterations']
        elif method == 'SGD':
            x = range(1, len(history['costs']) + 1)
        else:  # Mini-batch
            x = history['iterations']
        
        ax.plot(x, history['costs'], label=method, linewidth=2.5, alpha=0.8, marker='o' if len(x) < 50 else '')
    
    ax.set_xlabel('Iteration/Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cost (Binary Cross-Entropy)', fontsize=13, fontweight='bold')
    ax.set_title('Cost Function Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Zoomed in (last 50%)
    ax = axes[1]
    for method, history in histories.items():
        costs = history['costs']
        start_idx = len(costs) // 2
        
        if method == 'Batch GD':
            x = history['iterations'][start_idx:]
        elif method == 'SGD':
            x = range(start_idx + 1, len(costs) + 1)
        else:  # Mini-batch
            x = history['iterations'][start_idx:]
        
        ax.plot(x, costs[start_idx:], label=method, linewidth=2.5, alpha=0.8, marker='o' if len(x) < 30 else '')
    
    ax.set_xlabel('Iteration/Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cost (Binary Cross-Entropy)', fontsize=13, fontweight='bold')
    ax.set_title('Cost Curve (Zoomed - Last 50%)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cost_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: cost_comparison.png")
    plt.show()


def plot_gradient_norm_comparison(histories):
    """Plot gradient norm evolution"""
    fig, ax = plt.subplots(figsize=(13, 6))
    
    for method, history in histories.items():
        norms = history['gradient_norms']
        
        # Sample for plotting clarity (too dense otherwise)
        if len(norms) > 500:
            step = max(1, len(norms) // 500)
            x = range(0, len(norms), step)
            y = norms[::step]
        else:
            x = range(len(norms))
            y = norms
        
        ax.plot(x, y, label=method, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Update Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=13, fontweight='bold')
    ax.set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('gradient_norm_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: gradient_norm_comparison.png")
    plt.show()


def plot_convergence_speed():
    """Create a conceptual plot of convergence rates"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Theoretical convergence rates
    k = np.arange(1, 1001)
    batch_convergence = 1 / k
    sgd_convergence = 1 / np.sqrt(k)
    
    ax.plot(k, batch_convergence, label='Batch GD (O(1/k))', linewidth=2.5, alpha=0.8)
    ax.plot(k, sgd_convergence, label='SGD (O(1/√k))', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Number of Iterations', fontsize=13, fontweight='bold')
    ax.set_ylabel('Distance to Optimum (Theoretical)', fontsize=13, fontweight='bold')
    ax.set_title('Convergence Rate Comparison (Theory)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax.text(0.5, 0.05, 'Note: SGD converges slower per iteration, but faster per example',
            transform=ax.transAxes, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('convergence_rate_theory.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: convergence_rate_theory.png")
    plt.show()


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Complete demonstration comparing gradient descent methods"""
    
    print("=" * 80)
    print("GRADIENT DESCENT COMPARISON: BATCH vs STOCHASTIC vs MINI-BATCH")
    print("=" * 80)
    print()
    
    # =====================================================================
    # STEP 1: CREATE DATASET
    # =====================================================================
    print("STEP 1: Creating Synthetic Binary Classification Dataset")
    print("-" * 80)
    
    np.random.seed(42)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    y = y.reshape(-1, 1)
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} examples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} examples")
    print(f"Features normalized: mean ≈ 0, std ≈ 1")
    print()
    
    histories = {}
    
    # =====================================================================
    # STEP 2: BATCH GRADIENT DESCENT
    # =====================================================================
    print("STEP 2: Training with Batch Gradient Descent")
    print("-" * 80)
    
    bgd = BatchGradientDescent(learning_rate=0.1)
    w_batch, b_batch, hist_batch = bgd.train(X_train, y_train, iterations=200, verbose_interval=20)
    
    h_test = sigmoid(np.dot(X_test, w_batch) + b_batch)
    acc_batch = ((h_test >= 0.5) == y_test).mean()
    print(f"Test Accuracy: {acc_batch:.4f}\n")
    
    histories['Batch GD'] = hist_batch
    
    # =====================================================================
    # STEP 3: STOCHASTIC GRADIENT DESCENT
    # =====================================================================
    print("STEP 3: Training with Stochastic Gradient Descent")
    print("-" * 80)
    
    sgd = StochasticGradientDescent(learning_rate=0.1, decay_rate=0.01)
    w_sgd, b_sgd, hist_sgd = sgd.train(X_train, y_train, epochs=1, verbose_interval=200)
    
    h_test = sigmoid(np.dot(X_test, w_sgd) + b_sgd)
    acc_sgd = ((h_test >= 0.5) == y_test).mean()
    print(f"Test Accuracy: {acc_sgd:.4f}\n")
    
    histories['SGD'] = hist_sgd
    
    # =====================================================================
    # STEP 4: MINI-BATCH GRADIENT DESCENT
    # =====================================================================
    print("STEP 4: Training with Mini-Batch Gradient Descent")
    print("-" * 80)
    
    minibatch = MiniBatchGradientDescent(learning_rate=0.1, decay_rate=0.001)
    w_minibatch, b_minibatch, hist_minibatch = minibatch.train(
        X_train, y_train, epochs=25, batch_size=32, verbose_interval=5
    )
    
    h_test = sigmoid(np.dot(X_test, w_minibatch) + b_minibatch)
    acc_minibatch = ((h_test >= 0.5) == y_test).mean()
    print(f"Test Accuracy: {acc_minibatch:.4f}\n")
    
    histories['Mini-batch GD'] = hist_minibatch
    
    # =====================================================================
    # STEP 5: VISUALIZATIONS
    # =====================================================================
    print("STEP 5: Generating Visualizations")
    print("-" * 80)
    
    plot_cost_comparison(histories)
    plot_gradient_norm_comparison(histories)
    plot_convergence_speed()
    
    # =====================================================================
    # STEP 6: SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\n1. BATCH GRADIENT DESCENT")
    print(f"   Iterations: 200")
    print(f"   Final cost: {hist_batch['costs'][-1]:.6f}")
    print(f"   Test accuracy: {acc_batch:.4f}")
    print(f"   Total time: {sum(hist_batch['times']):.4f} seconds")
    print(f"   Convergence: Smooth monotonic decrease")
    
    print("\n2. STOCHASTIC GRADIENT DESCENT")
    print(f"   Epochs: 1 (= {len(hist_sgd['gradient_norms'])} updates)")
    print(f"   Final cost: {hist_sgd['costs'][-1]:.6f}")
    print(f"   Test accuracy: {acc_sgd:.4f}")
    print(f"   Total time: {sum(hist_sgd['times']):.4f} seconds")
    print(f"   Convergence: Noisy zigzag with learning rate decay")
    
    print("\n3. MINI-BATCH GRADIENT DESCENT")
    print(f"   Epochs: 25 (batch_size=32)")
    print(f"   Final cost: {hist_minibatch['costs'][-1]:.6f}")
    print(f"   Test accuracy: {acc_minibatch:.4f}")
    print(f"   Total time: {sum(hist_minibatch['times']):.4f} seconds")
    print(f"   Convergence: Balanced smooth + noisy")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. CONVERGENCE BEHAVIOR:
   - Batch GD: Smooth, predictable descent (exact gradients)
   - SGD: Noisy, zigzag but faster per update (random examples)
   - Mini-batch: Balanced (smooth with manageable noise)

2. SPEED PER UPDATE:
   - Batch GD: Slowest (processes all 800 examples)
   - SGD: Fastest (processes 1 example)
   - Mini-batch: Medium (processes 32 examples)

3. MEMORY EFFICIENCY:
   - Batch GD: High (must load all data)
   - SGD: Low (load one example at a time)
   - Mini-batch: Low-Medium (load batch_size examples)

4. FINAL ACCURACY:
   - All three reach similar accuracy (for convex problems)
   - The path differs, but destination is the same

5. PRACTICAL USE:
   - Small data (< 10K): Use Batch GD
   - Large data (> 100K): Use SGD or Mini-batch
   - Production ML: Use Mini-batch (GPU-friendly)
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    main()