"""
Cost Functions and Gradient Descent Variants: Complete Implementation

This module implements:
1. Multiple cost functions (MSE, MAE, RMSE, Huber, Log-Cosh)
2. Multiple gradient descent variants (Batch, SGD, Mini-batch, Momentum, Adam)
3. Comparison framework and visualization
"""

import numpy as np    # type: ignore
import matplotlib.pyplot as plt    # type: ignore
from typing import Tuple, List, Dict
from enum import Enum


# ============================================================================
# COST FUNCTIONS
# ============================================================================

class CostFunction:
    """Base class for cost functions."""
    
    def compute(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """Compute cost for a single batch."""
        raise NotImplementedError
    
    def gradient(self, y_actual: np.ndarray, y_predicted: np.ndarray, 
                 X: np.ndarray) -> np.ndarray:
        """Compute gradient of cost with respect to predictions."""
        raise NotImplementedError
    
    def name(self) -> str:
        """Return human-readable name."""
        raise NotImplementedError


class MSE(CostFunction):
    """Mean Squared Error: (1/m) * Σ(ŷ - y)²"""
    
    def compute(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        MSE = (1/m) * Σ(ŷ - y)²
        """
        m = len(y_actual)
        squared_errors = (y_predicted - y_actual) ** 2
        return np.sum(squared_errors) / m
    
    def gradient(self, y_actual: np.ndarray, y_predicted: np.ndarray, 
                 X: np.ndarray) -> np.ndarray:
        """
        ∂MSE/∂w = (1/m) * Σ(ŷ - y) * x
        """
        m = len(y_actual)
        errors = y_predicted - y_actual
        return np.dot(X.T, errors) / m
    
    def name(self) -> str:
        return "MSE"


class RMSE(CostFunction):
    """Root Mean Squared Error: √((1/m) * Σ(ŷ - y)²)"""
    
    def compute(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        RMSE = √(MSE)
        """
        m = len(y_actual)
        squared_errors = (y_predicted - y_actual) ** 2
        mse = np.sum(squared_errors) / m
        return np.sqrt(mse)
    
    def gradient(self, y_actual: np.ndarray, y_predicted: np.ndarray, 
                 X: np.ndarray) -> np.ndarray:
        """
        For optimization, gradient of RMSE ∝ gradient of MSE
        (constant factor doesn't change direction)
        """
        m = len(y_actual)
        errors = y_predicted - y_actual
        return np.dot(X.T, errors) / m
    
    def name(self) -> str:
        return "RMSE"


class MAE(CostFunction):
    """Mean Absolute Error: (1/m) * Σ|ŷ - y|"""
    
    def compute(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        MAE = (1/m) * Σ|ŷ - y|
        """
        m = len(y_actual)
        absolute_errors = np.abs(y_predicted - y_actual)
        return np.sum(absolute_errors) / m
    
    def gradient(self, y_actual: np.ndarray, y_predicted: np.ndarray, 
                 X: np.ndarray) -> np.ndarray:
        """
        ∂MAE/∂w = (1/m) * Σ sign(ŷ - y) * x
        Note: Non-differentiable at zero, using sign function
        """
        m = len(y_actual)
        errors = y_predicted - y_actual
        # Use sign (non-smooth at zero, but works in practice)
        sign_errors = np.sign(errors)
        # Small epsilon to avoid zero gradient
        sign_errors = np.where(np.abs(errors) < 1e-7, 0, sign_errors)
        return np.dot(X.T, sign_errors) / m
    
    def name(self) -> str:
        return "MAE"


class HuberLoss(CostFunction):
    """
    Huber Loss: MSE for |e| ≤ δ, MAE for |e| > δ
    Smooth approximation, robust to outliers.
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Args:
            delta: Threshold parameter (typical: 1.0)
        """
        self.delta = delta
    
    def compute(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        L = sum of:
        - (1/2) * e² if |e| ≤ δ
        - δ * (|e| - δ/2) if |e| > δ
        """
        m = len(y_actual)
        errors = y_predicted - y_actual
        abs_errors = np.abs(errors)
        
        # Quadratic region
        quadratic = (abs_errors <= self.delta)
        loss_quad = 0.5 * (errors[quadratic] ** 2)
        
        # Linear region
        linear = (abs_errors > self.delta)
        loss_linear = self.delta * (abs_errors[linear] - 0.5 * self.delta)
        
        total_loss = np.sum(loss_quad) + np.sum(loss_linear)
        return total_loss / m
    
    def gradient(self, y_actual: np.ndarray, y_predicted: np.ndarray, 
                 X: np.ndarray) -> np.ndarray:
        """
        ∂L/∂w = (1/m) * Σ g(e) * x
        where g(e) = e if |e| ≤ δ, δ*sign(e) if |e| > δ
        """
        m = len(y_actual)
        errors = y_predicted - y_actual
        abs_errors = np.abs(errors)
        
        # Gradient
        grad_errors = np.zeros_like(errors)
        
        # Quadratic region
        quadratic = (abs_errors <= self.delta)
        grad_errors[quadratic] = errors[quadratic]
        
        # Linear region
        linear = (abs_errors > self.delta)
        grad_errors[linear] = self.delta * np.sign(errors[linear])
        
        return np.dot(X.T, grad_errors) / m
    
    def name(self) -> str:
        return f"Huber(δ={self.delta})"


class LogCosh(CostFunction):
    """
    Log-Cosh Loss: log(cosh(e)) ≈ MSE for small e, MAE for large e
    Smooth everywhere, good for optimization.
    """
    
    def compute(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        L = (1/m) * Σ log(cosh(ŷ - y))
        """
        m = len(y_actual)
        errors = y_predicted - y_actual
        # Avoid numerical overflow with safe computation
        loss = errors + np.log(1 + np.exp(-2 * errors)) - np.log(2)
        return np.sum(loss) / m
    
    def gradient(self, y_actual: np.ndarray, y_predicted: np.ndarray, 
                 X: np.ndarray) -> np.ndarray:
        """
        ∂L/∂w = (1/m) * Σ tanh(e) * x
        """
        m = len(y_actual)
        errors = y_predicted - y_actual
        grad_errors = np.tanh(errors)
        return np.dot(X.T, grad_errors) / m
    
    def name(self) -> str:
        return "Log-Cosh"


# ============================================================================
# GRADIENT DESCENT VARIANTS
# ============================================================================

class GradientDescentVariant:
    """Base class for gradient descent variants."""
    
    def __init__(self, learning_rate: float, cost_function: CostFunction):
        """
        Args:
            learning_rate: Step size (α)
            cost_function: Cost function to minimize
        """
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.cost_history = []
    
    def update(self, w: np.ndarray, b: float, gradient_w: np.ndarray, 
               gradient_b: float) -> Tuple[np.ndarray, float]:
        """
        Update parameters. Override in subclasses.
        """
        raise NotImplementedError
    
    def name(self) -> str:
        raise NotImplementedError


class BatchGradientDescent(GradientDescentVariant):
    """
    Standard gradient descent using ALL examples per iteration.
    """
    
    def update(self, w: np.ndarray, b: float, gradient_w: np.ndarray, 
               gradient_b: float) -> Tuple[np.ndarray, float]:
        """
        w := w - α * ∂J/∂w
        b := b - α * ∂J/∂b
        """
        w_new = w - self.learning_rate * gradient_w
        b_new = b - self.learning_rate * gradient_b
        return w_new, b_new
    
    def name(self) -> str:
        return "Batch GD"


class SGD(GradientDescentVariant):
    """
    Stochastic Gradient Descent: one random example per iteration.
    Not explicitly implemented here (handled by mini-batch with size=1).
    """
    
    def name(self) -> str:
        return "SGD"


class MiniBatchGradientDescent(GradientDescentVariant):
    """
    Mini-batch gradient descent: balance between batch and SGD.
    """
    
    def __init__(self, learning_rate: float, cost_function: CostFunction, 
                 batch_size: int = 32):
        super().__init__(learning_rate, cost_function)
        self.batch_size = batch_size
    
    def update(self, w: np.ndarray, b: float, gradient_w: np.ndarray, 
               gradient_b: float) -> Tuple[np.ndarray, float]:
        """
        Same as batch GD, but applied to mini-batch.
        """
        w_new = w - self.learning_rate * gradient_w
        b_new = b - self.learning_rate * gradient_b
        return w_new, b_new
    
    def name(self) -> str:
        return f"Mini-batch (size={self.batch_size})"


class Momentum(GradientDescentVariant):
    """
    Gradient Descent with Momentum.
    Accumulates gradient direction for faster convergence.
    """
    
    def __init__(self, learning_rate: float, cost_function: CostFunction, 
                 beta: float = 0.9):
        """
        Args:
            beta: Momentum coefficient (typical: 0.9)
        """
        super().__init__(learning_rate, cost_function)
        self.beta = beta
        self.velocity_w = None
        self.velocity_b = None
    
    def _initialize_velocity(self, w_shape: Tuple) -> None:
        """Initialize velocity on first call."""
        self.velocity_w = np.zeros(w_shape)
        self.velocity_b = 0
    
    def update(self, w: np.ndarray, b: float, gradient_w: np.ndarray, 
               gradient_b: float) -> Tuple[np.ndarray, float]:
        """
        v = β * v + (1 - β) * gradient
        w := w - α * v
        """
        if self.velocity_w is None:
            self._initialize_velocity(w.shape)
        
        # Update velocity
        self.velocity_w = self.beta * self.velocity_w + (1 - self.beta) * gradient_w
        self.velocity_b = self.beta * self.velocity_b + (1 - self.beta) * gradient_b
        
        # Update parameters
        w_new = w - self.learning_rate * self.velocity_w
        b_new = b - self.learning_rate * self.velocity_b
        
        return w_new, b_new
    
    def name(self) -> str:
        return f"Momentum (β={self.beta})"


class RMSprop(GradientDescentVariant):
    """
    RMSprop: Root Mean Square Propagation.
    Adaptive learning rate per parameter using exponential moving average.
    """
    
    def __init__(self, learning_rate: float, cost_function: CostFunction, 
                 beta: float = 0.9, epsilon: float = 1e-8):
        """
        Args:
            beta: Exponential moving average coefficient
            epsilon: Small value for numerical stability
        """
        super().__init__(learning_rate, cost_function)
        self.beta = beta
        self.epsilon = epsilon
        self.cache_w = None
        self.cache_b = None
    
    def _initialize_cache(self, w_shape: Tuple) -> None:
        """Initialize cache on first call."""
        self.cache_w = np.zeros(w_shape)
        self.cache_b = 0
    
    def update(self, w: np.ndarray, b: float, gradient_w: np.ndarray, 
               gradient_b: float) -> Tuple[np.ndarray, float]:
        """
        cache = β * cache + (1 - β) * gradient²
        w := w - α * gradient / (√cache + ε)
        """
        if self.cache_w is None:
            self._initialize_cache(w.shape)
        
        # Update cache
        self.cache_w = self.beta * self.cache_w + (1 - self.beta) * (gradient_w ** 2)
        self.cache_b = self.beta * self.cache_b + (1 - self.beta) * (gradient_b ** 2)
        
        # Update parameters
        w_new = w - self.learning_rate * gradient_w / (np.sqrt(self.cache_w) + self.epsilon)
        b_new = b - self.learning_rate * gradient_b / (np.sqrt(self.cache_b) + self.epsilon)
        
        return w_new, b_new
    
    def name(self) -> str:
        return f"RMSprop (β={self.beta})"


class Adam(GradientDescentVariant):
    """
    Adam: Adaptive Moment Estimation.
    Combines momentum and RMSprop for adaptive learning.
    """
    
    def __init__(self, learning_rate: float, cost_function: CostFunction, 
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Args:
            beta1: Momentum (first moment) coefficient
            beta2: RMSprop (second moment) coefficient
            epsilon: Numerical stability constant
        """
        super().__init__(learning_rate, cost_function)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_w = None  # First moment (momentum)
        self.v_w = None  # Second moment (RMSprop)
        self.m_b = None
        self.v_b = None
        self.t = 0  # Time step for bias correction
    
    def _initialize_moments(self, w_shape: Tuple) -> None:
        """Initialize moment estimates on first call."""
        self.m_w = np.zeros(w_shape)
        self.v_w = np.zeros(w_shape)
        self.m_b = 0
        self.v_b = 0
    
    def update(self, w: np.ndarray, b: float, gradient_w: np.ndarray, 
               gradient_b: float) -> Tuple[np.ndarray, float]:
        """
        m = β₁ * m + (1 - β₁) * gradient     (momentum)
        v = β₂ * v + (1 - β₂) * gradient²    (RMSprop)
        m_corrected = m / (1 - β₁^t)         (bias correction)
        v_corrected = v / (1 - β₂^t)
        w := w - α * m_corrected / (√v_corrected + ε)
        """
        if self.m_w is None:
            self._initialize_moments(w.shape)
        
        self.t += 1
        
        # Update biased first and second moments
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * gradient_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (gradient_w ** 2)
        
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * gradient_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (gradient_b ** 2)
        
        # Bias correction
        m_w_corrected = self.m_w / (1 - self.beta1 ** self.t)
        v_w_corrected = self.v_w / (1 - self.beta2 ** self.t)
        
        m_b_corrected = self.m_b / (1 - self.beta1 ** self.t)
        v_b_corrected = self.v_b / (1 - self.beta2 ** self.t)
        
        # Update parameters
        w_new = w - self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
        b_new = b - self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
        
        return w_new, b_new
    
    def name(self) -> str:
        return f"Adam (β₁={self.beta1}, β₂={self.beta2})"


# ============================================================================
# TRAINING FRAMEWORK
# ============================================================================

class LinearRegressionOptimizer:
    """
    Framework to test different optimizers on linear regression problem.
    """
    
    def __init__(self, optimizer: GradientDescentVariant, verbose: bool = False):
        """
        Args:
            optimizer: Gradient descent variant to use
            verbose: Print progress during training
        """
        self.optimizer = optimizer
        self.verbose = verbose
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, iterations: int = 1000, 
            batch_size: int = None) -> None:
        """
        Train using specified optimizer.
        
        Args:
            X (np.ndarray): Features (m, n)
            y (np.ndarray): Targets (m, 1)
            iterations: Number of gradient descent iterations
            batch_size: Batch size for mini-batch methods
        """
        m, n = X.shape
        
        if batch_size is None:
            batch_size = m  # Full batch by default
        
        # Initialize parameters
        self.weights = np.zeros((n, 1))
        self.bias = 0.0
        
        # Training loop
        for iteration in range(iterations):
            # Get batch
            if batch_size < m:
                # Mini-batch: random sampling
                indices = np.random.choice(m, batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                # Full batch
                X_batch = X
                y_batch = y
            
            # Predictions
            predictions = np.dot(X_batch, self.weights) + self.bias
            
            # Cost
            cost = self.optimizer.cost_function.compute(y_batch, predictions)
            self.optimizer.cost_history.append(cost)
            
            # Gradients
            gradient_w = self.optimizer.cost_function.gradient(y_batch, predictions, X_batch)
            gradient_b = np.mean(predictions - y_batch)
            
            # Update parameters
            self.weights, self.bias = self.optimizer.update(
                self.weights, self.bias, gradient_w, gradient_b
            )
            
            # Verbose output
            if self.verbose and (iteration % (iterations // 10) == 0 or iteration == iterations - 1):
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias


# ============================================================================
# DEMONSTRATION AND COMPARISON
# ============================================================================

def create_synthetic_dataset(m: int = 100, n: int = 3, noise_std: float = 100) -> Tuple:
    """
    Create synthetic linear regression dataset.
    
    Returns:
        Tuple of (X, y, true_weights)
    """
    np.random.seed(42)
    
    # Features
    X = np.random.randn(m, n) * 100 + 500
    
    # True weights
    true_weights = np.array([[2.5], [1.8], [-0.5]])
    true_bias = 100
    
    # Target with noise
    y = np.dot(X, true_weights) + true_bias + np.random.randn(m, 1) * noise_std
    
    return X, y, true_weights


def compare_cost_functions():
    """Compare different cost functions on same problem."""
    print("\n" + "=" * 80)
    print("COST FUNCTION COMPARISON")
    print("=" * 80)
    
    X, y, true_w = create_synthetic_dataset(m=100, n=3, noise_std=50)
    
    # Generate some predictions with error
    w_test = np.array([[2.0], [1.5], [-0.6]])
    y_pred = np.dot(X, w_test) + 100
    
    cost_functions = [
        MSE(),
        RMSE(),
        MAE(),
        HuberLoss(delta=1.0),
        LogCosh()
    ]
    
    print(f"\nDataset: {X.shape[0]} examples, {X.shape[1]} features")
    print(f"Prediction error stats:")
    print(f"  Mean error: {np.mean(y_pred - y):.2f}")
    print(f"  Std error: {np.std(y_pred - y):.2f}")
    print(f"  Max error: {np.max(np.abs(y_pred - y)):.2f}")
    print(f"  Min error: {np.min(np.abs(y_pred - y)):.2f}")
    
    print(f"\nCost function values:")
    print(f"{'Function':<20} {'Cost':<15} {'Interpretation'}")
    print("-" * 60)
    
    for cf in cost_functions:
        cost = cf.compute(y, y_pred)
        print(f"{cf.name():<20} {cost:<15.4f}")


def compare_optimizers():
    """Compare different optimizers on same dataset."""
    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON")
    print("=" * 80)
    
    # Create dataset
    X, y, true_w = create_synthetic_dataset(m=100, n=3, noise_std=50)
    
    # Normalize
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    X = (X - X_mean) / (X_std + 1e-8)
    
    y_mean, y_std = np.mean(y), np.std(y)
    y = (y - y_mean) / (y_std + 1e-8)
    
    # Optimizers to test
    optimizers = [
        BatchGradientDescent(learning_rate=0.1, cost_function=RMSE()),
        MiniBatchGradientDescent(learning_rate=0.1, cost_function=RMSE(), batch_size=32),
        MiniBatchGradientDescent(learning_rate=0.1, cost_function=RMSE(), batch_size=1),  # SGD
        Momentum(learning_rate=0.1, cost_function=RMSE(), beta=0.9),
        RMSprop(learning_rate=0.1, cost_function=RMSE(), beta=0.9),
        Adam(learning_rate=0.1, cost_function=RMSE(), beta1=0.9, beta2=0.999),
    ]
    
    results = {}
    
    print(f"\nTraining with {X.shape[0]} examples, {X.shape[1]} features")
    print(f"Iterations: 100\n")
    
    for optimizer in optimizers:
        model = LinearRegressionOptimizer(optimizer, verbose=False)
        model.fit(X, y, iterations=100, batch_size=X.shape[0])
        
        results[optimizer.name()] = {
            'cost_history': optimizer.cost_history,
            'final_cost': optimizer.cost_history[-1]
        }
    
    # Print results
    print(f"{'Optimizer':<30} {'Final Cost':<15} {'Convergence'}")
    print("-" * 70)
    
    for name, data in results.items():
        final_cost = data['final_cost']
        initial_cost = data['cost_history'][0]
        convergence = (initial_cost - final_cost) / initial_cost * 100
        print(f"{name:<30} {final_cost:<15.6f} {convergence:.1f}% reduction")
    
    # Plot comparison
    plt.figure(figsize=(14, 6))
    
    # Plot 1: All optimizers
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['cost_history'], label=name, linewidth=2, alpha=0.7)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (RMSE)', fontsize=12)
    plt.title('Optimizer Convergence Comparison', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed in (last 50 iterations)
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['cost_history'][-50:], label=name, linewidth=2, alpha=0.7)
    
    plt.xlabel('Iteration (from 50 onwards)', fontsize=12)
    plt.ylabel('Cost (RMSE)', fontsize=12)
    plt.title('Convergence Detail (Last 50 Iterations)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_cost_functions_on_training():
    """Compare cost functions during training."""
    print("\n" + "=" * 80)
    print("COST FUNCTION BEHAVIOR DURING TRAINING")
    print("=" * 80)
    
    X, y, _ = create_synthetic_dataset(m=100, n=3, noise_std=50)
    
    # Normalize
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    cost_functions = [
        MSE(),
        RMSE(),
        MAE(),
        HuberLoss(delta=1.0),
        LogCosh()
    ]
    
    results = {}
    
    for cf in cost_functions:
        optimizer = Momentum(learning_rate=0.05, cost_function=cf)
        model = LinearRegressionOptimizer(optimizer, verbose=False)
        model.fit(X, y, iterations=100)
        
        results[cf.name()] = optimizer.cost_history
    
    # Plot
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Full training
    plt.subplot(1, 2, 1)
    for name, history in results.items():
        plt.plot(history, label=name, linewidth=2, alpha=0.7)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Function Behavior During Training', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Detail (avoid outlier scale issues)
    plt.subplot(1, 2, 2)
    for name, history in results.items():
        # Normalize for better comparison
        normalized = (np.array(history) - history[0]) / (history[0] - history[-1])
        plt.plot(normalized, label=name, linewidth=2, alpha=0.7)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Normalized Progress (0=start, 1=end)', fontsize=12)
    plt.title('Relative Convergence Speed', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("COST FUNCTIONS AND GRADIENT DESCENT VARIANTS DEMONSTRATION")
    print("=" * 80)
    
    # 1. Compare cost functions
    compare_cost_functions()
    
    # 2. Compare optimizers
    compare_optimizers()
    
    # 3. Compare cost functions during training
    compare_cost_functions_on_training()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
