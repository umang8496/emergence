import numpy as np    #type: ignore
import matplotlib.pyplot as plt    #type: ignore
from typing import Tuple

class LinearRegression:
    """
    Multivariate Linear Regression using Gradient Descent.
    
    Learns a linear relationship: y = w·x + b
    Optimizes using gradient descent with RMSE cost function.
    
    Includes automatic feature scaling (normalization) for better convergence.
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, verbose: bool = True):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate (float): Step size for gradient descent (alpha)
            iterations (int): Number of gradient descent iterations
            verbose (bool): Whether to print progress during training
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        
        # Will be set during fit()
        self.weights = None
        self.bias = None
        self.cost_history = []
        
        # For feature scaling
        self.feature_means = None
        self.feature_stds = None
        self.target_mean = None
        self.target_std = None
        
    def _calculate_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate predictions: ŷ = X·w + b
        
        Args:
            X (np.ndarray): Input features (m x n matrix)
                           m = number of examples
                           n = number of features
        
        Returns:
            np.ndarray: Predicted values (m x 1 vector)
        """
        return np.dot(X, self.weights) + self.bias
    
    def _calculate_cost(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        Formula:
            RMSE = √((1/m) * Σ(ŷ - y)²)
        
        Args:
            predictions (np.ndarray): Predicted values
            y (np.ndarray): Actual values
        
        Returns:
            float: RMSE cost
        """
        m = len(y)
        squared_errors = (predictions - y) ** 2
        mse = np.sum(squared_errors) / m
        rmse = np.sqrt(mse)
        return rmse
    
    def _calculate_gradients(self, X: np.ndarray, predictions: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate gradients for weights and bias.
        
        Formulas:
            ∂J/∂wⱼ = (1/m) * Σ(ŷ - y) * xⱼ
            ∂J/∂b = (1/m) * Σ(ŷ - y)
        
        Where:
            J = cost function
            m = number of examples
            ŷ = predictions
            y = actual values
        
        Args:
            X (np.ndarray): Input features (m x n)
            predictions (np.ndarray): Predicted values (m x 1)
            y (np.ndarray): Actual values (m x 1)
        
        Returns:
            Tuple[np.ndarray, float]: Gradients for weights and bias
        """
        m = len(y)
        errors = predictions - y
        
        # Gradient for weights: (1/m) * X^T * (ŷ - y)
        weight_gradients = np.dot(X.T, errors) / m
        
        # Gradient for bias: (1/m) * Σ(ŷ - y)
        bias_gradient = np.sum(errors) / m
        
        return weight_gradients, bias_gradient
    
    def _update_parameters(self, weight_gradients: np.ndarray, bias_gradient: float) -> None:
        """
        Update weights and bias using gradient descent.
        
        Update rule:
            wⱼ := wⱼ - α * ∂J/∂wⱼ
            b := b - α * ∂J/∂b
        
        Where α is learning rate.
        
        Args:
            weight_gradients (np.ndarray): Gradients for weights
            bias_gradient (float): Gradient for bias
        """
        self.weights -= self.learning_rate * weight_gradients
        self.bias -= self.learning_rate * bias_gradient
    
    def _normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using standardization (z-score normalization).
        
        Formula:
            x_normalized = (x - mean(x)) / std(x)
        
        This is critical for gradient descent convergence when features have
        different scales (e.g., square feet vs. number of bedrooms).
        
        Args:
            X (np.ndarray): Feature matrix (m x n)
            fit (bool): If True, compute mean/std from X (training).
                       If False, use stored mean/std (prediction).
        
        Returns:
            np.ndarray: Normalized features
        """
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
        
        # Avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1
        
        return (X - self.feature_means) / self.feature_stds
    
    def _normalize_target(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize target variable.
        
        Args:
            y (np.ndarray): Target values (m x 1)
            fit (bool): If True, compute mean/std from y.
                       If False, use stored mean/std.
        
        Returns:
            np.ndarray: Normalized target
        """
        if fit:
            self.target_mean = np.mean(y)
            self.target_std = np.std(y)
        
        if self.target_std == 0:
            self.target_std = 1
        
        return (y - self.target_mean) / self.target_std
    
    def _denormalize_predictions(self, normalized_predictions: np.ndarray) -> np.ndarray:
        """
        Convert predictions back to original scale.
        
        Args:
            normalized_predictions (np.ndarray): Predictions in normalized scale
        
        Returns:
            np.ndarray: Predictions in original scale
        """
        return (normalized_predictions * self.target_std) + self.target_mean
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Train the linear regression model.
        
        Process:
        1. Normalize features and target (for better convergence)
        2. Initialize weights and bias
        3. For each iteration:
            - Calculate predictions
            - Calculate cost (RMSE)
            - Calculate gradients
            - Update parameters using gradient descent
        4. Store cost history for convergence analysis
        
        Args:
            X (np.ndarray): Training features (m x n)
                           m = number of examples
                           n = number of features
            y (np.ndarray): Training target values (m x 1)
        
        Returns:
            LinearRegression: Returns self for method chaining
        
        Raises:
            ValueError: If X and y have incompatible shapes
        """
        # Validate input
        m, n = X.shape
        if len(y) != m:
            raise ValueError(f"X has {m} examples, y has {len(y)} examples")
        
        # Normalize features and target (CRITICAL for convergence)
        X_normalized = self._normalize_features(X, fit=True)
        y_normalized = self._normalize_target(y, fit=True)
        
        # Initialize parameters
        self.weights = np.zeros((n, 1))
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent iterations
        for iteration in range(self.iterations):
            # Step 1: Calculate predictions
            predictions = self._calculate_predictions(X_normalized)
            
            # Step 2: Calculate cost (RMSE on normalized scale)
            cost = self._calculate_cost(predictions, y_normalized)
            self.cost_history.append(cost)
            
            # Step 3: Calculate gradients
            weight_grads, bias_grad = self._calculate_gradients(X_normalized, predictions, y_normalized)
            
            # Step 4: Update parameters
            self._update_parameters(weight_grads, bias_grad)
            
            # Print progress
            if self.verbose and (iteration % (self.iterations // 10) == 0 or iteration == self.iterations - 1):
                print(f"Iteration {iteration}: Cost (RMSE) = {cost:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Important: Uses the same normalization parameters learned during training.
        
        Args:
            X (np.ndarray): Input features (m x n)
        
        Returns:
            np.ndarray: Predicted values in original scale (m x 1)
        
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Normalize using training mean/std (NOT refitting)
        X_normalized = self._normalize_features(X, fit=False)
        
        # Get predictions in normalized scale
        predictions_normalized = self._calculate_predictions(X_normalized)
        
        # Convert back to original scale
        predictions_original = self._denormalize_predictions(predictions_normalized)
        
        return predictions_original
    
    def get_params(self) -> dict:
        """
        Get learned parameters.
        
        Returns:
            dict: Dictionary containing weights and bias
        """
        if self.weights is None:
            raise ValueError("Model must be fitted first")
        
        return {
            'weights': self.weights.flatten(),
            'bias': self.bias,
            'feature_count': len(self.weights)
        }
    
    def plot_loss_curve(self) -> None:
        """
        Plot the cost function over iterations.
        
        Useful for diagnosing:
        - Convergence: Should decrease over time
        - Learning rate: Too high (spikes), too low (slow)
        """
        if not self.cost_history:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (RMSE)', fontsize=12)
        plt.title('Loss Curve: Cost Function Over Iterations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("MULTIVARIATE LINEAR REGRESSION: DEMONSTRATION")
    print("=" * 80)

    # Create synthetic dataset
    print("\n1. CREATING SYNTHETIC DATASET")
    print("=" * 80)

    np.random.seed(42)

    # Generate features
    m = 100  # number of examples
    n = 3    # number of features

    # Feature 1: Square Feet (1000-4000)
    sq_feet = np.random.uniform(1000, 4000, m).reshape(-1, 1)

    # Feature 2: Number of Bedrooms (2-6)
    bedrooms = np.random.uniform(2, 6, m).reshape(-1, 1)

    # Feature 3: Age in Years (0-50)
    age = np.random.uniform(0, 50, m).reshape(-1, 1)

    # Combine features
    X = np.hstack([sq_feet, bedrooms, age])

    # True relationship: Price = 150*sqft + 30000*bedrooms - 500*age + noise
    true_weights = np.array([150, 30000, -500])
    true_bias = 50000
    noise = np.random.normal(0, 20000, m)
    y = np.dot(X, true_weights.reshape(-1, 1)) + true_bias + noise.reshape(-1, 1)

    print(f"Dataset shape: X = {X.shape}, y = {y.shape}")
    print(f"Features: Square Feet, Bedrooms, Age")
    print(f"Target: House Price")
    print(f"\nTrue underlying relationship (unknown to model):")
    print(f"Price = 150*sqft + 30000*bedrooms - 500*age + 50000")

    # ========================================================================
    # TRAIN THE MODEL
    # ========================================================================
    print("\n2. TRAINING THE MODEL")
    print("=" * 80)

    model = LinearRegression(learning_rate=0.1, iterations=1000, verbose=True)
    model.fit(X, y)

    # ========================================================================
    # EVALUATE THE MODEL
    # ========================================================================
    print("\n3. MODEL EVALUATION")
    print("=" * 80)

    params = model.get_params()
    print(f"\nLearned Parameters:")
    print(f"Weight 1 (sqft coefficient):     {params['weights'][0]:.2f}")
    print(f"  True value was:                 150.00")
    print(f"\nWeight 2 (bedrooms coefficient): {params['weights'][1]:.2f}")
    print(f"  True value was:                 30000.00")
    print(f"\nWeight 3 (age coefficient):     {params['weights'][2]:.2f}")
    print(f"  True value was:                 -500.00")
    print(f"\nBias (intercept):               {params['bias']:.2f}")
    print(f"  True value was:                 50000.00")

    final_cost = model.cost_history[-1]
    print(f"\nFinal RMSE Cost: {final_cost:.2f}")
    print(f"(Average prediction error: ${final_cost:.0f})")

    # ========================================================================
    # MAKE PREDICTIONS ON NEW DATA
    # ========================================================================
    print("\n4. PREDICTIONS ON NEW DATA")
    print("=" * 80)

    # New house: 2500 sqft, 4 bedrooms, 10 years old
    new_house = np.array([[2500, 4, 10]])
    predicted_price = model.predict(new_house)[0][0]

    print(f"\nNew House Features:")
    print(f"  Square Feet: 2500")
    print(f"  Bedrooms: 4")
    print(f"  Age: 10 years")
    print(f"\nPredicted Price: ${predicted_price:,.2f}")

    # Compare with true formula
    true_price = np.dot(new_house, true_weights.reshape(-1, 1)) + true_bias
    print(f"True formula price: ${true_price[0][0]:,.2f}")
    print(f"Difference: ${abs(predicted_price - true_price[0][0]):,.2f}")

    # ========================================================================
    # VISUALIZE LOSS CURVE
    # ========================================================================
    print("\n5. VISUALIZING LOSS CURVE")
    print("=" * 80)
    print("Displaying plot...")
    model.plot_loss_curve()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
