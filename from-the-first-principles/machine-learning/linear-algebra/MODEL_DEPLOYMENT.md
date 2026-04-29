# Model Deployment Using JSON: A Complete Guide

## Introduction

Once the linear regression model is trained, it exists only in the Python process memory.  
To use it in production, we must persist it to disk and serve it to end users.  
This guide explains the complete process using JSON format, which is portable, language-agnostic, and production-ready.  

---

## Part 1: Understanding Model Attributes

When we train a linear regression model, four key attributes are learned. These attributes together define our entire trained model.  

### 1. `model.weights`

**What it is:** The slope or coefficient for each feature.
During training, the algorithm learns how much each feature contributes to the prediction.  
For a house price model with 3 features (square feet, bedrooms, age), we might have weights like `[150, 1000, -500]`.  

**Relevance:** Without weights, we cannot make any prediction. These are the core learnings.
**For N features there will be exactly 256 weights**.  

```python
model.weights = np.array([150.5, 1000.2, -500.1, 0.5, ...])  # 256 values
print(f"Number of weights: {len(model.weights)}")  # Output: 256
```

### 2. `model.bias`

**What it is:** The baseline prediction when all features are zero.
Think of it as: "If everything is zero, the default value is $50,000." It's a single number that shifts all predictions up or down.  

**Relevance:** It's the intercept of the linear equation.  

**Equation:** `prediction = (weights · features) + bias`  

```python
model.bias = 50000.0
print(f"Bias (baseline): ${model.bias:,.2f}")  # Output: Bias (baseline): $50,000.00
```

### 3. `model.feature_means`

**What it is:** The average value of each feature computed during training.
During training, we normalized our input features by subtracting their mean. These means must be saved.  

**Relevance:** When new data arrives for prediction, we must normalize it using the exact same means.  
If we don't, the input will be in a different scale than the training data, and predictions will be wrong.  

**For N features there will be exactly 256 means**.  

```python
model.feature_means = np.array([2000, 4, 20, 100, ...])  # 256 values
print(f"Feature 0 mean: {model.feature_means[0]}")  # Output: Feature 0 mean: 2000
```

### 4. `model.feature_stds`

**What it is:** The standard deviation of each feature computed during training.  
During training, we normalized features by dividing by their standard deviation. This ensures all features have similar ranges.  

**Relevance:** Just like `means`, these must be used consistently.  
New predictions must be normalized using these exact standard deviations.  

**For N features there will be exactly 256 standard-deviations**.  

```python
model.feature_stds = np.array([800, 1.5, 10, 50, ...])  # 256 values
print(f"Feature 0 std: {model.feature_stds[0]}")  # Output: Feature 0 std: 800
```

### Why All Four Matter Together

The prediction formula is:

```text
1. Normalize input: x_normalized = (x - feature_means) / feature_stds
2. Predict: prediction = (weights · x_normalized) + bias
```

Without all four attributes, this formula breaks:

- No weights/bias → Can't compute prediction
- No feature_means/stds → Input is wrong scale → Prediction is garbage

---

## Part 2: Saving the Model to JSON

### Step 1: Extract the Attributes

After training the model, extract the four key attributes:

```python
import json
import numpy as np

# Assuming model is your trained LinearRegression object
weights = model.weights
bias = model.bias
feature_means = model.feature_means
feature_stds = model.feature_stds
```

### Step 2: Create the Model Package

Create a dictionary containing not just the model, but metadata about it:

```python
model_package = {
    # Core model parameters
    "model": {
        "weights": weights.flatten().tolist(),  # Convert numpy to list
        "bias": float(bias)
    },
    
    # Preprocessing parameters (CRITICAL)
    "preprocessing": {
        "feature_means": feature_means.flatten().tolist(),
        "feature_stds": feature_stds.flatten().tolist(),
        "feature_names": [
            "square_feet", "bedrooms", "age", "bathrooms", 
            # ... 252 more feature names
        ]
    },
    
    # Metadata
    "metadata": {
        "model_version": "1.0",
        "model_type": "linear_regression",
        "created_date": "2026-04-29",
        "description": "House price prediction model"
    },
    
    # Training information
    "training": {
        "training_examples": 1000,
        "feature_count": 256,
        "cost_function": "MSE",
        "learning_rate": 0.01,
        "iterations": 1000
    },
    
    # Performance metrics
    "performance": {
        "training_rmse": 15000,
        "validation_rmse": 16500,
        "training_mae": 12000,
        "validation_mae": 13200
    }
}
```

### Step 3: Save to JSON File

```python
with open('house_price_model_v1.0.json', 'w') as f:
    json.dump(model_package, f, indent=2)
print("Model saved to house_price_model_v1.0.json")
```

### File Size

With 256 features, your JSON file will be approximately **20-30 KB**:

- 256 weights × 15 bytes ≈ 3.8 KB
- 256 means × 15 bytes ≈ 3.8 KB
- 256 stds × 15 bytes ≈ 3.8 KB
- Metadata and formatting ≈ 8-15 KB
- **Total: ~20-30 KB**

Compare this to:

- A single image: 100 KB to 5 MB
- A single document: 50 KB to 500 KB
- The model: 20-30 KB

The model file is tiny and easily transferable.

---

## Part 3: Loading the Model from JSON

### Step 1: Load the JSON File

```python
import json
import numpy as np

def load_model(filepath):
    """Load a trained model from JSON file."""
    with open(filepath, 'r') as f:
        model_package = json.load(f)
    
    return model_package

# Usage
model_package = load_model('house_price_model_v1.0.json')
```

### Step 2: Extract Components

```python
# Extract model parameters
weights = np.array(model_package['model']['weights']).reshape(-1, 1)
bias = model_package['model']['bias']

# Extract preprocessing parameters (CRITICAL)
feature_means = np.array(model_package['preprocessing']['feature_means'])
feature_stds = np.array(model_package['preprocessing']['feature_stds'])
feature_names = model_package['preprocessing']['feature_names']

# Get metadata
model_version = model_package['metadata']['model_version']
print(f"Loaded model version: {model_version}")
print(f"Features: {len(feature_names)}")
print(f"Training RMSE: {model_package['performance']['training_rmse']}")
```

### Step 3: Make Predictions

```python
def normalize_input(x, feature_means, feature_stds):
    """Normalize input using training parameters."""
    return (x - feature_means) / feature_stds

def predict(features_array, weights, bias, feature_means, feature_stds):
    """
    Make a prediction.
    
    Args:
        features_array: np.array of shape (n_samples, n_features)
        weights: np.array of shape (n_features, 1)
        bias: float
        feature_means: np.array of shape (n_features,)
        feature_stds: np.array of shape (n_features,)
    
    Returns:
        predictions: np.array of shape (n_samples, 1)
    """
    # Normalize
    x_normalized = normalize_input(features_array, feature_means, feature_stds)
    
    # Predict
    predictions = np.dot(x_normalized, weights) + bias
    
    return predictions

# Usage
new_house_features = np.array([[2200, 4, 10, 2, ...]])  # 256 features
prediction = predict(new_house_features, weights, bias, feature_means, feature_stds)
print(f"Predicted price: ${prediction[0][0]:,.2f}")
```

---

## Part 4: Input Validation

Before making predictions, always validate that the input is in the expected range.

```python
def validate_input(features_dict, feature_names, feature_means, feature_stds):
    """
    Validate input before prediction.
    
    Args:
        features_dict: Dictionary with feature names as keys
        feature_names: List of expected feature names
        feature_means: Means from training (for range calculation)
        feature_stds: Stds from training (for range calculation)
    
    Returns:
        (is_valid, error_message)
    """
    
    # Check all features present
    missing = [f for f in feature_names if f not in features_dict]
    if missing:
        return False, f"Missing features: {missing}"
    
    # Check each feature is in reasonable range
    for i, feature_name in enumerate(feature_names):
        value = features_dict[feature_name]
        
        # Type check
        try:
            value = float(value)
        except (ValueError, TypeError):
            return False, f"Feature '{feature_name}' must be numeric"
        
        # Range check (within 3 std from mean)
        min_val = feature_means[i] - 3 * feature_stds[i]
        max_val = feature_means[i] + 3 * feature_stds[i]
        
        if not (min_val <= value <= max_val):
            return False, f"Feature '{feature_name}' = {value} outside expected range [{min_val:.2f}, {max_val:.2f}]"
    
    return True, None

# Usage
user_input = {
    'square_feet': 2200,
    'bedrooms': 4,
    'age': 10,
    # ... 253 more features
}

is_valid, error = validate_input(user_input, feature_names, feature_means, feature_stds)
if not is_valid:
    print(f"Invalid input: {error}")
else:
    print("Input validated successfully")
```

---

## Part 5: Flask API for Serving Predictions

This is how we serve the model to end users over HTTP.  

### Basic API Setup

```python
from flask import Flask, request, jsonify
import json
import numpy as np

app = Flask(__name__)

# Load model once at startup (not on every request)
def load_model_at_startup():
    with open('house_price_model_v1.0.json', 'r') as f:
        return json.load(f)

model_package = load_model_at_startup()

# Extract parameters
weights = np.array(model_package['model']['weights']).reshape(-1, 1)
bias = model_package['model']['bias']
feature_means = np.array(model_package['preprocessing']['feature_means'])
feature_stds = np.array(model_package['preprocessing']['feature_stds'])
feature_names = model_package['preprocessing']['feature_names']
```

### Prediction Endpoint

```python
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    API endpoint for making predictions.
    
    Expected request:
    {
        "square_feet": 2200,
        "bedrooms": 4,
        "age": 10,
        ... (250 more features)
    }
    
    Response:
    {
        "predicted_price": 380000.50,
        "model_version": "1.0"
    }
    """
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate input
        is_valid, error_msg = validate_input(data, feature_names, feature_means, feature_stds)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Prepare features in correct order
        features_array = np.array([[data[fname] for fname in feature_names]])
        
        # Normalize
        x_normalized = (features_array - feature_means) / feature_stds
        
        # Predict
        prediction = np.dot(x_normalized, weights)[0][0] + bias
        
        # Return response
        return jsonify({
            'predicted_price': float(prediction),
            'model_version': model_package['metadata']['model_version'],
            'status': 'success'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Health Check Endpoint

```python
@app.route('/health', methods=['GET'])
def health_check():
    """Check if the API is running and model is loaded."""
    return jsonify({
        'status': 'healthy',
        'model_version': model_package['metadata']['model_version'],
        'model_created': model_package['metadata']['created_date'],
        'features': len(feature_names)
    }), 200
```

### Model Info Endpoint

```python
@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    return jsonify({
        'version': model_package['metadata']['model_version'],
        'model_type': model_package['metadata']['model_type'],
        'features': {
            'count': model_package['preprocessing']['feature_count'],
            'names': feature_names[:10] + ['...'] if len(feature_names) > 10 else feature_names
        },
        'performance': model_package['performance'],
        'training': {
            'examples': model_package['training']['training_examples'],
            'iterations': model_package['training']['iterations']
        }
    }), 200
```

### Batch Prediction Endpoint

```python
@app.route('/predict-batch', methods=['POST'])
def predict_batch_endpoint():
    """
    Make multiple predictions at once.
    
    Expected request:
    {
        "predictions": [
            {"square_feet": 2200, "bedrooms": 4, "age": 10, ...},
            {"square_feet": 3000, "bedrooms": 5, "age": 5, ...},
            {"square_feet": 1500, "bedrooms": 3, "age": 15, ...}
        ]
    }
    """
    
    try:
        data = request.get_json()
        predictions_list = data.get('predictions', [])
        
        if not predictions_list:
            return jsonify({'error': 'No predictions provided'}), 400
        
        results = []
        
        for pred_input in predictions_list:
            # Validate
            is_valid, error_msg = validate_input(pred_input, feature_names, feature_means, feature_stds)
            if not is_valid:
                results.append({'error': error_msg})
                continue
            
            # Predict
            features_array = np.array([[pred_input[fname] for fname in feature_names]])
            x_normalized = (features_array - feature_means) / feature_stds
            prediction = np.dot(x_normalized, weights)[0][0] + bias
            
            results.append({'predicted_price': float(prediction)})
        
        return jsonify({
            'results': results,
            'count': len(results)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Running the API

```python
if __name__ == '__main__':
    # In production, use a production server like Gunicorn
    # For development:
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## Part 6: How to Call the API

### Using curl (command line)

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "square_feet": 2200,
    "bedrooms": 4,
    "age": 10,
    "bathrooms": 2
  }'

# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/model-info

# Batch predictions
curl -X POST http://localhost:5000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"square_feet": 2200, "bedrooms": 4, "age": 10},
      {"square_feet": 3000, "bedrooms": 5, "age": 5}
    ]
  }'
```

### Using Python requests

```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', 
    json={
        'square_feet': 2200,
        'bedrooms': 4,
        'age': 10
    }
)

result = response.json()
print(f"Predicted price: ${result['predicted_price']:,.2f}")

# Batch predictions
response = requests.post('http://localhost:5000/predict-batch',
    json={
        'predictions': [
            {'square_feet': 2200, 'bedrooms': 4, 'age': 10},
            {'square_feet': 3000, 'bedrooms': 5, 'age': 5},
            {'square_feet': 1500, 'bedrooms': 3, 'age': 15}
        ]
    }
)

results = response.json()
for i, pred in enumerate(results['results']):
    print(f"House {i+1}: ${pred['predicted_price']:,.2f}")
```

### Using JavaScript

```javascript
// Fetch API (browser)
async function predictPrice(features) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(features)
    });
    
    const result = await response.json();
    console.log(`Predicted price: $${result.predicted_price.toLocaleString()}`);
}

// Usage
predictPrice({
    square_feet: 2200,
    bedrooms: 4,
    age: 10
});
```

---

## Part 7: Complete End-to-End Example

```python
# save_model.py - Training and saving
import json
import numpy as np

# Assume model is your trained LinearRegression
model_package = {
    "model": {
        "weights": model.weights.flatten().tolist(),
        "bias": float(model.bias)
    },
    "preprocessing": {
        "feature_means": model.feature_means.flatten().tolist(),
        "feature_stds": model.feature_stds.flatten().tolist(),
        "feature_names": ['sqft', 'beds', 'age', ...]  # 256 features
    },
    "metadata": {
        "model_version": "1.0",
        "model_type": "linear_regression",
        "created_date": "2026-04-29"
    },
    "performance": {
        "training_rmse": 15000,
        "validation_rmse": 16500
    }
}

with open('model.json', 'w') as f:
    json.dump(model_package, f)
```

```python
# serve_model.py - Loading and serving
from flask import Flask, request, jsonify
import json
import numpy as np

app = Flask(__name__)

# Load model
with open('model.json', 'r') as f:
    pkg = json.load(f)

w = np.array(pkg['model']['weights']).reshape(-1, 1)
b = pkg['model']['bias']
means = np.array(pkg['preprocessing']['feature_means'])
stds = np.array(pkg['preprocessing']['feature_stds'])
names = pkg['preprocessing']['feature_names']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    x = np.array([[data[n] for n in names]])
    x_norm = (x - means) / stds
    pred = np.dot(x_norm, w)[0][0] + b
    return jsonify({'predicted_price': float(pred)})

if __name__ == '__main__':
    app.run()
```

---

## Summary

| Step | Stage           | Action                | Purpose                                                                |
|------|-----------------|-----------------------|------------------------------------------------------------------------|
| 1    | Training Phase  | Extract attributes    | Capture weights, bias, means, stds that define the model               |
| 2    | Persistence     | Save to JSON          | Make model persistent, portable, and language-agnostic                 |
| 3    | Initialization  | Load from JSON        | Reconstruct model parameters into memory                               |
| 4    | Input Processing| Normalize input       | Apply training statistics to match the data scale                      |
| 5    | Validation      | Validate input        | Ensure data is in expected range (prevent garbage predictions)         |
| 6    | Inference       | Predict               | Compute: `(w·x_norm) + b` and return prediction                        |
| 7    | Serving         | Serve via API         | Expose model through HTTP endpoints for users                          |

---
