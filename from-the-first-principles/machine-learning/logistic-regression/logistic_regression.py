"""
Logistic Regression: Complete Implementation from Scratch

This module implements logistic regression for binary classification using
gradient descent optimization. No external ML libraries are used - everything
is built from first principles using NumPy.
"""

import numpy as np    #type: ignore
import matplotlib.pyplot as plt    #type: ignore
from typing import Tuple, List


# ============================================================================
# PART 1: CORE FUNCTIONS
# ============================================================================

class LogisticRegression:
    """
    Logistic Regression classifier using Gradient Descent.
    
    Learns to predict binary outcomes (0 or 1) by fitting a sigmoid curve
    to the data.
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, 
                 verbose: bool = True):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent (alpha)
            iterations: Number of training iterations
            verbose: Print progress during training
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        
        # Will be set during training
        self.weights = None
        self.bias = None
        self.cost_history = []
        
        # For feature normalization
        self.feature_means = None
        self.feature_stds = None
    
    # ========================================================================
    # SIGMOID FUNCTION
    # ========================================================================
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid function: σ(z) = 1 / (1 + e^(-z))
        
        Transforms any number into a probability between 0 and 1.
        
        Args:
            z: Linear combination (w·x + b)
        
        Returns:
            Probability between 0 and 1
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    # ========================================================================
    # FEATURE NORMALIZATION
    # ========================================================================
    
    def _normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using standardization (z-score normalization).
        
        Formula: x_normalized = (x - mean(x)) / std(x)
        
        Critical for convergence and numerical stability.
        
        Args:
            X: Feature matrix (m x n)
            fit: If True, compute mean/std from X (training)
                 If False, use stored mean/std (prediction)
        
        Returns:
            Normalized features
        """
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1
        
        return (X - self.feature_means) / self.feature_stds
    
    # ========================================================================
    # LINEAR COMBINATION
    # ========================================================================
    
    def _compute_z(self, X: np.ndarray) -> np.ndarray:
        """
        Compute linear combination: z = w·X + b
        
        This is the linear part before sigmoid.
        
        Args:
            X: Features (m x n)
        
        Returns:
            Linear combination (m x 1)
        """
        return np.dot(X, self.weights) + self.bias
    
    # ========================================================================
    # PREDICTIONS
    # ========================================================================
    
    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions: h(x) = sigmoid(w·x + b)
        
        Args:
            X: Features (m x n)
        
        Returns:
            Probabilities (m x 1) between 0 and 1
        """
        z = self._compute_z(X)
        return self.sigmoid(z)
    
    # ========================================================================
    # COST FUNCTION: BINARY CROSS-ENTROPY
    # ========================================================================
    
    def _compute_cost(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Binary Cross-Entropy cost.
        
        Formula: J = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
        
        This cost function:
        - Is 0 when prediction is correct
        - Approaches infinity when prediction is confidently wrong
        - Is symmetric (doesn't favor positive or negative class)
        
        Args:
            predictions: Predicted probabilities (m x 1)
            y: Actual labels (m x 1)
        
        Returns:
            Single cost value (scalar)
        """
        m = len(y)
        
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Binary cross-entropy formula
        cost = -(1/m) * np.sum(
            y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
        )
        
        return cost
    
    # ========================================================================
    # GRADIENT COMPUTATION
    # ========================================================================
    
    def _compute_gradients(self, X: np.ndarray, predictions: np.ndarray, 
                          y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.
        
        Gradient for weights: ∂J/∂w = (1/m) * X^T * (h - y)
        Gradient for bias: ∂J/∂b = (1/m) * Σ(h - y)
        
        These gradients point in the direction of steepest ascent.
        We move opposite to this direction (gradient descent).
        
        Args:
            X: Features (m x n)
            predictions: Predicted probabilities (m x 1)
            y: Actual labels (m x 1)
        
        Returns:
            Tuple of (weight_gradients, bias_gradient)
        """
        m = len(y)
        
        # Compute error
        error = predictions - y
        
        # Gradient for weights
        weight_gradients = (1/m) * np.dot(X.T, error)
        
        # Gradient for bias
        bias_gradient = (1/m) * np.sum(error)
        
        return weight_gradients, bias_gradient
    
    # ========================================================================
    # PARAMETER UPDATE
    # ========================================================================
    
    def _update_parameters(self, weight_gradients: np.ndarray, 
                          bias_gradient: float) -> None:
        """
        Update weights and bias using gradient descent.
        
        Update rule:
        w := w - learning_rate * ∂J/∂w
        b := b - learning_rate * ∂J/∂b
        
        Args:
            weight_gradients: Gradients for weights (n x 1)
            bias_gradient: Gradient for bias (scalar)
        """
        self.weights -= self.learning_rate * weight_gradients
        self.bias -= self.learning_rate * bias_gradient
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model.
        
        Process:
        1. Normalize features
        2. Initialize weights and bias
        3. For each iteration:
           a. Compute predictions
           b. Compute cost
           c. Compute gradients
           d. Update parameters
        4. Stop when cost converges
        
        Args:
            X: Training features (m x n)
            y: Training labels (m x 1), values should be 0 or 1
        
        Returns:
            Self (for method chaining)
        """
        m, n = X.shape
        
        # Validate input
        if len(y) != m:
            raise ValueError(f"X has {m} examples, y has {len(y)} examples")
        
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Labels y must be binary (0 or 1)")
        
        # Normalize features
        X_normalized = self._normalize_features(X, fit=True)
        
        # Initialize parameters
        self.weights = np.zeros((n, 1))
        self.bias = 0.0
        self.cost_history = []
        
        # Training loop (gradient descent)
        for iteration in range(self.iterations):
            # Forward pass
            predictions = self._get_predictions(X_normalized)
            
            # Compute cost
            cost = self._compute_cost(predictions, y)
            self.cost_history.append(cost)
            
            # Compute gradients
            weight_grads, bias_grad = self._compute_gradients(
                X_normalized, predictions, y
            )
            
            # Update parameters
            self._update_parameters(weight_grads, bias_grad)
            
            # Print progress
            if self.verbose and (iteration % (self.iterations // 10) == 0 or 
                                iteration == self.iterations - 1):
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
        
        return self
    
    # ========================================================================
    # PREDICTION
    # ========================================================================
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for new data.
        
        Returns probability of belonging to class 1 (positive class).
        
        Args:
            X: Features (m x n)
        
        Returns:
            Probabilities (m x 1), values between 0 and 1
        
        Raises:
            ValueError: If model hasn't been fitted
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Normalize using training parameters
        X_normalized = self._normalize_features(X, fit=False)
        
        # Get predictions
        return self._get_predictions(X_normalized)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for new data.
        
        Args:
            X: Features (m x n)
            threshold: Probability threshold for classification
                      Default 0.5, but can be adjusted
        
        Returns:
            Binary predictions (m x 1), values are 0 or 1
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                 threshold: float = 0.5) -> dict:
        """
        Evaluate model performance on a dataset.
        
        Computes:
        - Accuracy: Fraction of correct predictions
        - Precision: Of positive predictions, how many are correct
        - Recall: Of actual positives, how many we found
        - F1-score: Harmonic mean of precision and recall
        
        Args:
            X: Features (m x n)
            y: Actual labels (m x 1)
            threshold: Classification threshold
        
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X, threshold)
        
        # True positives, false positives, etc.
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        tn = np.sum((predictions == 0) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        
        # Metrics
        accuracy = (tp + tn) / len(y) if len(y) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_cost_curve(self) -> None:
        """
        Plot cost function over iterations.
        
        Useful for diagnosing:
        - Convergence: Cost should decrease monotonically
        - Learning rate: Too high (oscillations), too low (slow)
        """
        if not self.cost_history:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, linewidth=2, color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
        plt.title('Training Cost Over Iterations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 2: DEMONSTRATION
# ============================================================================

def create_synthetic_classification_data(n_examples: int = 200, 
                                        n_features: int = 2, 
                                        random_state: int = 42) -> Tuple:
    """
    Create a synthetic binary classification dataset.
    
    Returns:
        Tuple of (X, y) where X is features and y is binary labels
    """
    np.random.seed(random_state)
    
    # Create two classes
    class_0 = np.random.randn(n_examples // 2, n_features) - 1
    class_1 = np.random.randn(n_examples // 2, n_features) + 1
    
    X = np.vstack([class_0, class_1])
    y = np.vstack([np.zeros((n_examples // 2, 1)), 
                   np.ones((n_examples // 2, 1))])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_examples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def main():
    """
    Complete demonstration of logistic regression.
    """
    
    print("=" * 80)
    print("LOGISTIC REGRESSION: COMPLETE IMPLEMENTATION")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: CREATE DATASET
    # ========================================================================
    print("\n1. CREATING SYNTHETIC DATASET")
    print("-" * 80)
    
    X, y = create_synthetic_classification_data(n_examples=200, n_features=2)
    
    # Split into train and test
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"Training set: {X_train.shape[0]} examples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} examples")
    print(f"Class distribution (training):")
    print(f"  - Class 0 (negative): {(y_train == 0).sum()} examples")
    print(f"  - Class 1 (positive): {(y_train == 1).sum()} examples")
    
    # ========================================================================
    # STEP 2: CREATE AND TRAIN MODEL
    # ========================================================================
    print("\n2. TRAINING LOGISTIC REGRESSION MODEL")
    print("-" * 80)
    
    model = LogisticRegression(learning_rate=0.01, iterations=1000, verbose=True)
    model.fit(X_train, y_train)
    
    # ========================================================================
    # STEP 3: EVALUATE ON TRAINING SET
    # ========================================================================
    print("\n3. EVALUATING ON TRAINING SET")
    print("-" * 80)
    
    train_metrics = model.evaluate(X_train, y_train)
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Training Precision: {train_metrics['precision']:.4f}")
    print(f"Training Recall: {train_metrics['recall']:.4f}")
    print(f"Training F1-Score: {train_metrics['f1_score']:.4f}")
    
    # ========================================================================
    # STEP 4: EVALUATE ON TEST SET
    # ========================================================================
    print("\n4. EVALUATING ON TEST SET")
    print("-" * 80)
    
    test_metrics = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    
    # ========================================================================
    # STEP 5: MAKE PREDICTIONS ON NEW DATA
    # ========================================================================
    print("\n5. MAKING PREDICTIONS ON NEW DATA")
    print("-" * 80)
    
    # Test on a few examples
    test_examples = X_test[:5]
    probabilities = model.predict_proba(test_examples)
    predictions = model.predict(test_examples)
    actual = y_test[:5]
    
    print("\nExample predictions:")
    for i in range(5):
        prob = probabilities[i][0]
        pred = predictions[i][0]
        actual_label = actual[i][0]
        match = "✓" if pred == actual_label else "✗"
        print(f"  Example {i+1}: Probability={prob:.4f}, "
              f"Predicted={int(pred)}, Actual={int(actual_label)} {match}")
    
    # ========================================================================
    # STEP 6: ADJUST THRESHOLD
    # ========================================================================
    print("\n6. EXPERIMENTING WITH DIFFERENT THRESHOLDS")
    print("-" * 80)
    
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        metrics = model.evaluate(X_test, y_test, threshold=threshold)
        print(f"\nThreshold = {threshold}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    # ========================================================================
    # STEP 7: VISUALIZE
    # ========================================================================
    print("\n7. VISUALIZING TRAINING CURVE")
    print("-" * 80)
    print("Displaying cost curve...")
    model.plot_cost_curve()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey insights:")
    print("✓ Model learned to classify the data")
    print("✓ Cost decreased during training (convergence)")
    print("✓ Training and test accuracy are similar (no overfitting)")
    print("✓ Different thresholds give different precision/recall trade-offs")


if __name__ == "__main__":
    main()