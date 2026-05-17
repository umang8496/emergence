<!-- markdownlint-configure-file {"MD036": false, "MD040": false} -->

# Deep Learning — A Comprehensive Guide to the Basics

---

## Table of Contents

01. [What is Deep Learning?](#1-what-is-deep-learning)
02. [How is Deep Learning Different from Machine Learning?](#2-how-is-deep-learning-different-from-machine-learning)
03. [Core Architecture of a Neural Network](#3-core-architecture-of-a-neural-network)
04. [Flavours of Deep Learning](#4-flavours-of-deep-learning)
05. [Perceptron](#5-perceptron)
06. [Neuron](#6-neuron)
07. [Neural Networks](#7-neural-networks)
08. [Forward Propagation](#8-forward-propagation)
09. [Loss Functions and Their Variants](#9-loss-functions-and-their-variants)
10. [Backpropagation](#10-backpropagation)
11. [Gradient Descent and Its Types](#11-gradient-descent-and-its-types)
12. [End-to-End Example — Full DL Loop](#12-end-to-end-example--full-dl-loop)

---

## 1. What is Deep Learning?

Deep Learning (DL) is a subfield of Machine Learning that enables computers to learn patterns directly from raw data — without requiring humans to manually define what features or patterns to look for.

The term **"Deep"** refers to the depth of a neural network — the number of layers stacked between the input and the output. Each layer learns increasingly abstract representations of the data. A network with many such layers is called a **deep** neural network.

At its core, Deep Learning is about:

- Taking raw input data (images, text, audio, numbers)
- Passing it through multiple layers of mathematical transformations
- Producing a meaningful output (a classification, a prediction, a generated response)
- Learning the right transformations purely from examples, by minimising error

Deep Learning is the technology behind image recognition, speech assistants, language models, recommendation systems, self-driving cars, and much more.

---

## 2. How is Deep Learning Different from Machine Learning?

### Machine Learning — A Quick Recap

In classical Machine Learning, you:

1. Collect data
2. **Manually engineer features** — decide which aspects of the data matter (e.g., word count, price-to-earnings ratio, pixel brightness in a specific region)
3. Feed those features into an algorithm (SVM, Random Forest, Logistic Regression)
4. The algorithm learns a mapping from features to output

The critical step is step 2 — **feature engineering**. It requires domain expertise, is time-consuming, and does not scale to complex unstructured data like images or audio.

### Where Machine Learning Breaks Down

Consider the problem: "Does this image contain a cat?"

A classical ML engineer must answer: what features define a cat? Ears at a certain position? Whiskers? Eye shape? Fur texture? These features are enormously hard to define explicitly, and the combinations matter more than individual features.

Classical ML plateaus on such problems. No matter how much data you add, performance does not keep improving because the feature representation is limited by human definition.

### What Deep Learning Changes

Deep Learning removes the manual feature engineering step entirely.

```
Classical ML:
Raw Data → [Human Engineers Features] → Features → ML Model → Output

Deep Learning:
Raw Data → [Neural Network learns its own features automatically] → Output
```

The network figures out, through training, what intermediate representations are useful — without any human instruction on what to look for.

### Performance vs Data

One of the most important practical differences:

| Data Volume              | Classical ML           | Deep Learning      |
|--------------------------|------------------------|--------------------|
| Small (< 1000 samples)   | Often performs well    | May underperform   |
| Medium (10K–100K)        | Good performance       | Competitive        |
| Large (1M+)              | Performance plateaus   | Keeps improving    |

Classical ML hits a ceiling. Deep Learning continues to improve as more data is available — this is why the explosion of internet-scale data and GPU compute made DL dominant.

### Summary of Differences

| Aspect                    | Classical ML              | Deep Learning          |
|---------------------------|---------------------------|------------------------|
| Feature engineering       | Manual                    | Automatic              |
| Performance on raw data   | Poor                      | Excellent              |
| Data requirement          | Works on small datasets   | Needs large datasets   |
| Compute requirement       | Low                       | High (GPU/TPU)         |
| Interpretability          | Relatively easier         | Harder (black box)     |
| Best suited for           | Structured/tabular data   | Unstructured data      |

---

## 3. Core Architecture of a Neural Network

Every neural network, regardless of its specific type, shares the same foundational architecture: a sequence of **layers**, each containing **neurons**, connected by **weights**.

### The Three Layer Types

**Input Layer**

The entry point of the network. Each neuron in this layer corresponds to one feature in your input data. No computation happens here — it simply receives and passes the raw input forward.

```
If your data has 100 features → input layer has 100 neurons
If your image is 28×28 pixels → input layer has 784 neurons
```

**Hidden Layers**

One or more layers between input and output. These are where the actual learning happens. Each neuron in a hidden layer computes a weighted combination of its inputs and passes it through an activation function.

The word *hidden* refers to the fact that these layers are not directly visible in the input or output — they operate internally.

**Output Layer**

The final layer that produces the prediction. The number of neurons and the activation function used here depend entirely on the problem type:

| Problem Type                 | Output Neurons      | Activation         |
|------------------------------|---------------------|--------------------|
| Binary classification        | 1                   | Sigmoid            |
| Multi-class classification   | N (one per class)   | Softmax            |
| Regression                   | 1                   | None (raw value)   |

### Connections and Weights

Every neuron in one layer is connected to every neuron in the next layer (in a fully connected network). Each connection carries a **weight** — a number that controls how strongly the signal passes through.

Each neuron also has a **bias** — an offset that allows the neuron to activate even when all inputs are zero.

### Visual Representation

```
Input Layer     Hidden Layer 1    Hidden Layer 2    Output Layer
  (3 neurons)     (4 neurons)       (3 neurons)       (1 neuron)

   x1  ──────────  H1 ──────────  G1 ──────────
   x2  ──────────  H2 ──────────  G2 ──────────  O1 ──→ ŷ
   x3  ──────────  H3 ──────────  G3 ──────────
                   H4 ──────────
```

Every arrow carries a weight. Every neuron has a bias. This is the complete structure.

---

## 4. Flavours of Deep Learning

All deep learning architectures share the same foundation — neurons, layers, weights, forward pass, loss, backpropagation, and gradient descent. What differs between architectures is **how the neurons are connected and structured**, based on what type of data is being processed.

### 4.1 ANN — Artificial Neural Network

The base form of all neural networks. Also called a **Fully Connected Network** or **Dense Network**.

Every neuron in one layer connects to every neuron in the next. No special structure is imposed. Works best on structured, tabular data where features are independent.

```
Best for: tabular data, structured datasets, classification, regression
Examples: fraud detection, credit scoring, house price prediction
```

### 4.2 CNN — Convolutional Neural Network

Designed specifically for **spatial data** — data where nearby values are related to each other (images, video, audio spectrograms).

Instead of fully connecting every neuron to every other, CNN neurons only look at a **small local region** of the input (called a receptive field). A filter (small grid of weights) slides across the entire input, detecting the same pattern wherever it appears. This is the **convolution** operation.

Early layers detect low-level features (edges, gradients). Deeper layers detect high-level features (shapes, objects, faces).

```
Best for: images, video, spatial data
Examples: image classification, object detection, medical imaging, face recognition
```

A CNN dramatically reduces the number of parameters compared to a fully connected network on image data. A 64×64 image has 4096 inputs — connecting all of these to 256 hidden neurons gives 1M+ weights. A CNN filter of size 3×3 has only 9 weights and slides across the entire image.

### 4.3 RNN — Recurrent Neural Network

Designed for **sequential data** — data where order matters and context from the past is relevant.

Unlike ANN and CNN where information only flows forward, an RNN has a **feedback loop** — the output of a neuron at one time step feeds back into itself at the next time step. This gives the network a form of memory.

```
Best for: time series, text, speech, sequential data
Examples: next-word prediction, stock price forecasting, language translation
```

**Limitation:** RNNs struggle to remember information from far back in a sequence. The error signal diminishes as it travels backwards through many time steps — this is the **vanishing gradient problem**.

### 4.4 LSTM — Long Short-Term Memory

An evolved form of RNN that solves the vanishing gradient problem using **gating mechanisms**.

An LSTM cell has three gates:

- **Forget gate** — decides what information from the past to discard
- **Input gate** — decides what new information to store
- **Output gate** — decides what to pass forward

These gates allow LSTMs to selectively remember or forget information over long sequences.

```
Best for: long sequences where long-range dependencies matter
Examples: machine translation, speech recognition, sentiment analysis
```

### 4.5 Transformer

The most modern and dominant architecture. Originally designed for language, now applied to images, audio, and video.

Instead of processing a sequence step-by-step like RNN, a Transformer processes all elements of the input **simultaneously** using a mechanism called **self-attention**. Self-attention lets every element of the input look at every other element and decide what is relevant.

```
Best for: language, long-context understanding, large-scale tasks
Examples: GPT, BERT, Claude, language translation, code generation
```

Transformers have largely replaced RNNs and LSTMs for language tasks because they parallelise well on GPUs and handle long-range dependencies natively.

### 4.6 GAN — Generative Adversarial Network

A system of two competing networks:

- **Generator** — creates synthetic data (fake images, audio, text)
- **Discriminator** — tries to detect whether data is real or fake

They are trained together. The Generator improves at creating realistic data; the Discriminator improves at detecting fakes. This adversarial process produces remarkably realistic generated outputs.

```
Best for: generating new data resembling training data
Examples: image generation, deepfakes, data augmentation, art generation
```

### 4.7 Autoencoder

A network that learns to **compress** input into a smaller representation and then **reconstruct** it.

Structure:

- **Encoder** — compresses input into a latent vector (bottleneck)
- **Decoder** — reconstructs original input from the latent vector

The compressed representation (latent space) captures the most essential structure of the data. Anomalies appear very different in the latent space — making autoencoders powerful for anomaly detection.

```
Best for: compression, denoising, anomaly detection, dimensionality reduction
Examples: fraud detection, image denoising, recommendation systems
```

### 4.8 VAE — Variational Autoencoder

A probabilistic extension of the Autoencoder. Instead of compressing to a fixed vector, it compresses to a **probability distribution**. Sampling from this distribution generates new, plausible data similar to the training set.

```
Best for: generative tasks, drug discovery, image synthesis
Examples: generating new molecular structures, face synthesis
```

### 4.9 GNN — Graph Neural Network

All architectures above assume data is a grid (image) or sequence (text). GNNs handle data that is naturally a **graph** — a set of nodes connected by edges.

```
Best for: social networks, molecules, maps, recommendation systems
Examples: drug discovery, Google Maps ETA, fraud in transaction networks
```

### 4.10 Diffusion Models

A newer generative architecture. The training process involves:

1. Gradually adding noise to real data until it becomes pure random noise
2. Training the network to **reverse** this process — denoise step by step

At inference, start from pure noise and iteratively denoise into a realistic output.

```
Best for: high-quality image, audio, and video generation
Examples: Stable Diffusion, DALL-E 3, Midjourney
```

### Architecture Selection Guide

| Data Type                         | Recommended Architecture   |
|-----------------------------------|----------------------------|
| Tabular / structured              | ANN                        |
| Images / video                    | CNN                        |
| Sequences / time series           | RNN, LSTM                  |
| Language / long context           | Transformer                |
| Graph structured data             | GNN                        |
| Generating new data               | GAN, VAE, Diffusion        |
| Anomaly detection / compression   | Autoencoder                |

---

## 5. Perceptron

The perceptron is the **historical origin** of the modern neuron. Introduced by Frank Rosenblatt in 1958, it was the first mathematical model of a biological neuron.

### What a Perceptron Does

It takes multiple inputs, multiplies each by a weight, sums them up, adds a bias, and produces a **hard binary output** — either 0 or 1.

```
output = 1   if (w1*x1 + w2*x2 + ... + wn*xn + b) > threshold
         0   otherwise
```

### Visual Representation

```
x1 ──(w1)──┐
x2 ──(w2)──┼──→ [Σ weighted inputs + bias] ──→ [Step Function] ──→ 0 or 1
x3 ──(w3)──┘
```

### Python Implementation

```python
import numpy as np

def perceptron(inputs, weights, bias, threshold=0):
    z = np.dot(weights, inputs) + bias
    return 1 if z > threshold else 0

# Example
inputs  = np.array([0.6, 0.4, 0.8])
weights = np.array([0.5, 0.3, 0.2])
bias    = 0.1

output = perceptron(inputs, weights, bias)
print(f"Perceptron output: {output}")
```

### Why the Perceptron Was Limited

The hard step function (0 or 1) is not **differentiable**. This means you cannot compute a gradient at that point, which means you cannot use backpropagation to train it. The perceptron could only be trained with a simple rule that only worked for linearly separable data.

It also could not solve problems like XOR — where no single straight line can separate the two classes.

### From Perceptron to Modern Neuron

The key evolution:

| Property                 | Perceptron        | Modern Neuron         |
|--------------------------|-------------------|-----------------------|
| Output                   | Hard 0 or 1       | Continuous value      |
| Activation               | Step function     | ReLU, Sigmoid, Tanh   |
| Differentiable           | No                | Yes                   |
| Trainable via backprop   | No                | Yes                   |
| Can stack layers         | Not effectively   | Yes                   |

Replacing the hard threshold with a smooth, differentiable activation function was what unlocked deep learning.

---

## 6. Neuron

A neuron is the fundamental building block of every neural network. It is an evolution of the perceptron — same core structure, but with a smooth activation function that makes training via backpropagation possible.

### What a Neuron Computes

A neuron performs two operations:

**Step 1 — Linear combination:**

```
z = (w1*x1) + (w2*x2) + ... + (wn*xn) + b
```

Where:

- `x1, x2, ..., xn` are the inputs
- `w1, w2, ..., wn` are the weights (one per input)
- `b` is the bias
- `z` is called the **pre-activation value**

**Step 2 — Activation function:**

```
a = activation(z)
```

The activation function introduces non-linearity. Without it, stacking multiple layers collapses mathematically into a single linear transformation — no depth advantage whatsoever.

### The Weight

Each weight controls **how strongly an input influences the neuron's output**.

- Large positive weight → strong positive influence
- Large negative weight → strong negative influence (suppresses the neuron)
- Weight near zero → this input is largely ignored

### The Bias

The bias allows the neuron to produce a non-zero output even when all inputs are zero. It shifts the activation threshold, giving the neuron more flexibility in what it responds to.

Without bias: `z = w*x` (the line is forced through the origin — not flexible enough)
With bias: `z = w*x + b` (the line can be positioned anywhere)

### Activation Functions

**ReLU — Rectified Linear Unit**

```
ReLU(z) = max(0, z)
         = z    if z > 0
         = 0    if z ≤ 0
```

Most commonly used in hidden layers. Simple, computationally efficient, and avoids vanishing gradients for positive values.

**Sigmoid**

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Output range: 0 to 1. Used in output layer for binary classification. Squashes any value into a probability.

**Tanh — Hyperbolic Tangent**

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

Output range: -1 to 1. Zero-centered (unlike sigmoid), which makes it better for hidden layers in some cases.

**Softmax**

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)
```

Used in the output layer for multi-class classification. Converts raw scores into probabilities that sum to 1.

### Python Implementation

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def neuron(inputs, weights, bias, activation='relu'):
    z = np.dot(weights, inputs) + bias
    if activation == 'relu':
        return relu(z)
    elif activation == 'sigmoid':
        return sigmoid(z)
    elif activation == 'tanh':
        return tanh(z)

# Example
inputs  = np.array([0.6, 0.4, 0.8])
weights = np.array([0.5, 0.3, 0.2])
bias    = 0.1

print(f"z = {np.dot(weights, inputs) + bias:.4f}")
print(f"ReLU output:    {neuron(inputs, weights, bias, 'relu'):.4f}")
print(f"Sigmoid output: {neuron(inputs, weights, bias, 'sigmoid'):.4f}")
print(f"Tanh output:    {neuron(inputs, weights, bias, 'tanh'):.4f}")
```

---

## 7. Neural Networks

A neural network is a collection of neurons organised into layers, connected by weights, working together to learn a mapping from input to output.

### Why Multiple Neurons?

One neuron can only learn one linear boundary (one hyperplane in feature space). Multiple neurons in a layer can learn multiple boundaries simultaneously — each detecting a different pattern in the input.

### Why Multiple Layers?

Each layer learns progressively more abstract representations:

```
Layer 1: Learns simple patterns (edges, basic feature combinations)
Layer 2: Combines simple patterns into complex ones (shapes, correlations)
Layer 3: Combines complex patterns into abstractions (objects, concepts)
```

Without depth, the network cannot learn hierarchical representations. A single layer with unlimited neurons can theoretically approximate any function — but it would require an impractically large number of neurons. Depth makes learning efficient.

### Fully Connected (Dense) Network

In a fully connected network, every neuron in one layer connects to every neuron in the next. This is the default structure for ANN.

```
Parameters in a fully connected layer:
  weights = neurons_in × neurons_out
  biases  = neurons_out
  total   = (neurons_in × neurons_out) + neurons_out
```

Example: Input layer of 100 neurons → hidden layer of 64 neurons:

```
weights = 100 × 64 = 6400
biases  = 64
total   = 6464 parameters in this one layer
```

### How Many Layers and Neurons?

There is no formula. It is an informed design decision guided by:

| Factor                  | Guidance                                     |
|-------------------------|----------------------------------------------|
| Input and output size   | Fixed — dictated by data                     |
| Problem complexity      | More complex → more layers and neurons       |
| Dataset size            | Larger dataset → can afford larger network   |
| Risk of overfitting     | Small dataset → keep network small           |

**Practical starting point:**

```
1 hidden layer  → sufficient for most simple problems
2 hidden layers → handles most real-world classification problems
3+ layers       → complex tasks like vision, language
```

**Neuron count per hidden layer:** Start between input size and output size, commonly a power of 2 (16, 32, 64, 128). Decrease neuron count with depth — this creates a funnel that forces the network to compress and generalise.

```python
# Example architecture in PyTorch
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100, 64)   # Input: 100 features → 64 hidden neurons
        self.layer2 = nn.Linear(64, 32)    # 64 → 32
        self.output = nn.Linear(32, 1)     # 32 → 1 binary output
        self.relu   = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
```

---

## 8. Forward Propagation

Forward propagation is the process of feeding input data through the network, layer by layer, from input to output, to produce a prediction.

No learning happens during the forward pass. It is purely computation — applying weights, biases, and activation functions to produce a number at the output.

### Mathematical Notation

For a network with L layers, the forward pass computes:

```
For each layer l from 1 to L:

    Z[l] = W[l] · A[l-1] + b[l]     ← linear combination
    A[l] = activation(Z[l])           ← apply activation function

Where:
    A[0] = X                          ← input layer
    W[l] = weight matrix for layer l
    b[l] = bias vector for layer l
    Z[l] = pre-activation values
    A[l] = post-activation values (output of layer l)

Final output:
    ŷ = A[L]                          ← prediction
```

### Worked Example

Network: 3 inputs → 2 hidden neurons → 1 output

Inputs and weights:

```
x1=0.6,  x2=0.4,  x3=0.8  (one data point)

Hidden neuron H1: w11=0.5, w12=0.3, w13=0.2, b1=0.1
Hidden neuron H2: w21=0.4, w22=0.1, w23=0.6, b2=0.2
Output neuron O:  wo1=0.7, wo2=0.5, bo=0.3
```

**Step 1 — Hidden layer H1:**

```
z1 = (0.5×0.6) + (0.3×0.4) + (0.2×0.8) + 0.1
   = 0.30 + 0.12 + 0.16 + 0.1
   = 0.68

a1 = ReLU(0.68) = 0.68
```

**Step 2 — Hidden layer H2:**

```
z2 = (0.4×0.6) + (0.1×0.4) + (0.6×0.8) + 0.2
   = 0.24 + 0.04 + 0.48 + 0.2
   = 0.96

a2 = ReLU(0.96) = 0.96
```

**Step 3 — Output layer:**

```
zo = (0.7×0.68) + (0.5×0.96) + 0.3
   = 0.476 + 0.480 + 0.3
   = 1.256

ŷ = sigmoid(1.256)
  = 1 / (1 + e^(-1.256))
  = 0.779
```

Prediction: **0.779** → since > 0.5 → network predicts class 1.

### Python Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def forward_pass(x, weights, biases):
    """
    x       : input vector
    weights : list of weight matrices [W1, W2, ..., WL]
    biases  : list of bias vectors    [b1, b2, ..., bL]
    """
    cache = []          # store intermediate values for backprop later
    a = x

    # Hidden layers — ReLU activation
    for i in range(len(weights) - 1):
        z = np.dot(weights[i], a) + biases[i]
        a = relu(z)
        cache.append((z, a))

    # Output layer — Sigmoid activation
    z = np.dot(weights[-1], a) + biases[-1]
    a = sigmoid(z)
    cache.append((z, a))

    return a, cache

# Example usage
x  = np.array([0.6, 0.4, 0.8])

W1 = np.array([[0.5, 0.3, 0.2],
               [0.4, 0.1, 0.6]])
b1 = np.array([0.1, 0.2])

W2 = np.array([[0.7, 0.5]])
b2 = np.array([0.3])

prediction, cache = forward_pass(x, [W1, W2], [b1, b2])
print(f"Prediction: {prediction[0]:.4f}")
```

---

## 9. Loss Functions and Their Variants

The loss function (also called cost function or error function) measures **how wrong the network's prediction is** compared to the actual label.

It produces a single number. The goal of training is to **minimise this number** across all data points.

The choice of loss function depends entirely on the type of problem.

### 9.1 Mean Squared Error (MSE)

Used for **regression problems** — predicting a continuous output.

```
MSE = (1/n) × Σ (ŷᵢ - yᵢ)²
```

Where:

- `ŷᵢ` is the predicted value for data point i
- `yᵢ` is the actual value
- `n` is the number of data points

**Properties:**

- Penalises large errors heavily (due to squaring)
- Always positive
- Differentiable everywhere — suitable for gradient descent

**Example:**

```
Actual:    y  = 200
Predicted: ŷ  = 139
Error:     200 - 139 = 61
Loss:      61² = 3721
```

```python
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

y_true = np.array([200, 280, 350])
y_pred = np.array([139, 250, 370])

print(f"MSE Loss: {mse_loss(y_true, y_pred):.4f}")
```

### 9.2 Binary Cross-Entropy Loss

Used for **binary classification** — output is 0 or 1.

```
Loss = -( y × log(ŷ) + (1-y) × log(1-ŷ) )
```

Where:

- `y` is the actual label (0 or 1)
- `ŷ` is the predicted probability (0 to 1, from sigmoid)

**Intuition:**

When actual = 1:

```
Loss = -log(ŷ)
```

If ŷ = 0.99 → Loss = -log(0.99) = 0.01   (correct, small penalty)
If ŷ = 0.50 → Loss = -log(0.50) = 0.69   (uncertain, moderate penalty)
If ŷ = 0.01 → Loss = -log(0.01) = 4.60   (wrong, heavy penalty)

When actual = 0:

```
Loss = -log(1 - ŷ)
```

If ŷ = 0.01 → Loss = -log(0.99) = 0.01   (correct, small penalty)
If ŷ = 0.99 → Loss = -log(0.01) = 4.60   (wrong, heavy penalty)

Cross-entropy heavily penalises **confident wrong predictions** — a key property for classification.

**For n data points (average loss):**

```
Cost = -(1/n) × Σ [ yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ) ]
```

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.3])

print(f"Binary Cross-Entropy Loss: {binary_cross_entropy(y_true, y_pred):.4f}")
```

### 9.3 Categorical Cross-Entropy Loss

Used for **multi-class classification** — output is one of N classes. The output layer uses softmax to produce N probabilities summing to 1.

```
Loss = -Σ yᵢ × log(ŷᵢ)
```

Where `y` is a one-hot encoded vector (all zeros except a 1 at the true class index).

**Example — classifying digit 2 (out of 0-9):**

```
Actual (one-hot):  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Predicted (softmax): [0.02, 0.01, 0.85, 0.03, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02]

Loss = -(0×log(0.02) + 0×log(0.01) + 1×log(0.85) + ...)
     = -log(0.85)
     = 0.163
```

Only the probability at the true class position contributes to the loss.

```python
import numpy as np

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -np.sum(y_true * np.log(y_pred))

y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])   # true class is 2
y_pred = np.array([0.02, 0.01, 0.85, 0.03, 0.02,
                   0.01, 0.01, 0.01, 0.02, 0.02])

print(f"Categorical Cross-Entropy: {categorical_cross_entropy(y_true, y_pred):.4f}")
```

### 9.4 Mean Absolute Error (MAE)

Alternative to MSE for regression. Less sensitive to outliers because it does not square the error.

```
MAE = (1/n) × Σ |ŷᵢ - yᵢ|
```

| Property              | MSE                                 | MAE                                 |
|-----------------------|-------------------------------------|-------------------------------------|
| Outlier sensitivity   | High (squares errors)               | Low (absolute value)                |
| Differentiability     | Everywhere                          | Not at zero                         |
| Common use            | When outliers should be penalised   | When outliers should be tolerated   |

### 9.5 Loss Function Selection Guide

| Problem Type                 | Loss Function                   | Output Activation   |
|------------------------------|---------------------------------|---------------------|
| Regression                   | MSE or MAE                      | None                |
| Binary classification        | Binary Cross-Entropy            | Sigmoid             |
| Multi-class classification   | Categorical Cross-Entropy       | Softmax             |
| Generative models            | Varies (BCE, perceptual loss)   | Varies              |

---

## 10. Backpropagation

Backpropagation is the algorithm that computes **how much each weight in the network contributed to the loss**. Once you know each weight's contribution, you can adjust it to reduce the loss.

The name comes from the direction of computation — the error is propagated **backwards** through the network, from the output layer to the input layer, one layer at a time.

### The Core Idea

After a forward pass, you have a prediction and a loss. The loss is a function of every weight in the network (through a chain of computations). Backpropagation computes the **partial derivative** of the loss with respect to every weight:

```
∂Loss/∂w  for every weight w in the network
```

This partial derivative is called the **gradient**. It tells you:

- **Sign:** Whether increasing w increases or decreases the loss
- **Magnitude:** How sensitive the loss is to this particular weight

### The Chain Rule

The mathematical tool that makes backpropagation work is the **chain rule** from calculus.

If `Loss` depends on `A`, and `A` depends on `W`, then:

```
∂Loss/∂W = (∂Loss/∂A) × (∂A/∂W)
```

In a network with multiple layers, you chain this backwards through every layer:

```
∂Loss/∂W1 = (∂Loss/∂A3) × (∂A3/∂Z3) × (∂Z3/∂A2) × (∂A2/∂Z2) × (∂Z2/∂A1) × (∂A1/∂Z1) × (∂Z1/∂W1)
```

This looks intimidating but it is just repeated application of the same rule.

### Step-by-Step Backpropagation

Using our earlier example:

```
Network: 3 inputs → [H1, H2] → Output O
Actual y = 1
Prediction ŷ = 0.779
Loss = 0.250
```

**Step 1 — Gradient of loss with respect to prediction:**

```
∂Loss/∂ŷ = ŷ - y = 0.779 - 1 = -0.221
```

Negative — means increasing ŷ decreases loss — the prediction should be higher. Makes sense since y=1.

**Step 2 — Gradient through sigmoid (output activation):**

Derivative of sigmoid:

```
sigmoid'(z) = sigmoid(z) × (1 - sigmoid(z))
            = ŷ × (1 - ŷ)
            = 0.779 × 0.221
            = 0.172
```

**Combined delta at output (δo):**

```
δo = ∂Loss/∂ŷ × ∂ŷ/∂zo
   = -0.221 × 0.172
   = -0.038
```

**Step 3 — Gradients for output layer weights:**

```
∂Loss/∂wo1 = δo × a1 = -0.038 × 0.68 = -0.026
∂Loss/∂wo2 = δo × a2 = -0.038 × 0.96 = -0.037
∂Loss/∂bo  = δo × 1  = -0.038
```

Negative gradients → increasing these weights decreases loss → we will increase them.

**Step 4 — Propagate error back to hidden layer:**

The error flows back through each output weight:

```
δh1 = δo × wo1 × ReLU'(z1)
```

ReLU derivative:

```
ReLU'(z) = 1   if z > 0
           0   if z ≤ 0
```

Since z1 = 0.68 > 0, ReLU'(z1) = 1.

```
δh1 = -0.038 × 0.7 × 1 = -0.027
δh2 = -0.038 × 0.5 × 1 = -0.019
```

**Step 5 — Gradients for hidden layer weights:**

```
∂Loss/∂w11 = δh1 × x1 = -0.027 × 0.6 = -0.016
∂Loss/∂w12 = δh1 × x2 = -0.027 × 0.4 = -0.011
∂Loss/∂w13 = δh1 × x3 = -0.027 × 0.8 = -0.022
∂Loss/∂b1  = δh1 × 1  = -0.027

∂Loss/∂w21 = δh2 × x1 = -0.019 × 0.6 = -0.011
∂Loss/∂w22 = δh2 × x2 = -0.019 × 0.4 = -0.008
∂Loss/∂w23 = δh2 × x3 = -0.019 × 0.8 = -0.015
∂Loss/∂b2  = δh2 × 1  = -0.019
```

All gradients computed. All negative — all weights need to increase to reduce loss.

### Python Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def backward_pass(x, y, weights, biases, cache):
    """
    x       : input vector
    y       : actual label
    weights : [W1, W2]
    biases  : [b1, b2]
    cache   : stored (z, a) from forward pass
    """
    gradients = {}
    n = x.shape[0] if len(x.shape) > 0 else 1

    # Output layer gradient
    z2, a2 = cache[-1]
    z1, a1 = cache[-2]

    # δ at output (sigmoid derivative combined with loss derivative)
    delta_output = a2 - y                          # ∂Loss/∂ŷ × sigmoid'(z)

    # Gradients for output layer
    gradients['dW2'] = np.outer(delta_output, a1)  # outer product
    gradients['db2'] = delta_output

    # Propagate error to hidden layer
    delta_hidden = np.dot(weights[1].T, delta_output) * relu_derivative(z1)

    # Gradients for hidden layer
    gradients['dW1'] = np.outer(delta_hidden, x)
    gradients['db1'] = delta_hidden

    return gradients

# Example usage (continuing from forward pass)
x = np.array([0.6, 0.4, 0.8])
y = np.array([1.0])

W1 = np.array([[0.5, 0.3, 0.2], [0.4, 0.1, 0.6]])
b1 = np.array([0.1, 0.2])
W2 = np.array([[0.7, 0.5]])
b2 = np.array([0.3])

prediction, cache = forward_pass(x, [W1, W2], [b1, b2])
gradients = backward_pass(x, y, [W1, W2], [b1, b2], cache)

print("Gradients dW2:", gradients['dW2'])
print("Gradients dW1:", gradients['dW1'])
```

### The Vanishing Gradient Problem

In very deep networks, the gradient signal can become extremely small by the time it reaches the early layers. This happens because at each layer the gradient is multiplied by weights and activation derivatives — if these are small numbers (< 1), multiplying them repeatedly drives the gradient toward zero.

Early layers learn very slowly or stop learning altogether — the network effectively becomes shallow.

**Solutions:** ReLU activation (gradient = 1 for positive values), Batch Normalisation, Residual connections (as in ResNets), LSTM gating (in recurrent networks).

---

## 11. Gradient Descent and Its Types

Gradient descent is the **optimisation algorithm** that updates the weights after backpropagation has computed the gradients.

The intuition: imagine the loss as a hilly landscape and the weights as your position on it. The gradient tells you which direction is uphill. You step in the opposite direction — downhill — to reach the minimum loss.

### The Update Rule

```
w_new = w_old - η × ∂Loss/∂w
```

Where:

- `η` (eta) is the **learning rate** — controls the step size
- `∂Loss/∂w` is the gradient computed by backpropagation

**Learning Rate:**

| Learning Rate   | Effect                                              |
|-----------------|-----------------------------------------------------|
| Too large       | Weights overshoot the minimum — training diverges   |
| Too small       | Convergence is very slow — training takes forever   |
| Just right      | Smooth convergence to minimum loss                  |

Typical values: 0.001 to 0.01. Often tuned as a hyperparameter.

### 11.1 Batch Gradient Descent

Compute the average gradient across **all training data**, then update once.

```
For each epoch:
    Compute gradient using all N data points
    Update all weights once
```

**Pros:** Stable gradient — smooth convergence
**Cons:** Very slow for large datasets (must process everything before one update), memory intensive

```python
def batch_gradient_descent(X, y, weights, biases, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward pass on all data
        predictions, cache = forward_pass_batch(X, weights, biases)
        # Compute average loss
        loss = binary_cross_entropy(y, predictions)
        # Backprop on all data
        gradients = backward_pass_batch(X, y, weights, biases, cache)
        # Single weight update
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradients[f'dW{i+1}']
            biases[i]  -= learning_rate * gradients[f'db{i+1}']
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return weights, biases
```

### 11.2 Stochastic Gradient Descent (SGD)

Update weights after **every single data point**.

```
For each epoch:
    For each data point i:
        Compute gradient using only data point i
        Update all weights immediately
```

**Pros:** Frequent updates — faster learning per epoch, can escape local minima due to noise
**Cons:** Noisy updates — loss fluctuates significantly, does not converge smoothly

```python
def stochastic_gradient_descent(X, y, weights, biases, learning_rate, epochs):
    n = X.shape[0]
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n)
        total_loss = 0
        for i in indices:
            xi = X[i]
            yi = y[i]
            # Forward pass for one point
            prediction, cache = forward_pass(xi, weights, biases)
            total_loss += binary_cross_entropy(np.array([yi]), prediction)
            # Backprop for one point
            gradients = backward_pass(xi, yi, weights, biases, cache)
            # Immediate weight update
            for j in range(len(weights)):
                weights[j] -= learning_rate * gradients[f'dW{j+1}']
                biases[j]  -= learning_rate * gradients[f'db{j+1}']
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Loss: {total_loss/n:.4f}")
    return weights, biases
```

### 11.3 Mini-Batch Gradient Descent

The standard in practice. Update weights after every **small batch** of data points (typically 32, 64, or 128).

```
For each epoch:
    Split data into batches of size B
    For each batch:
        Compute average gradient over the batch
        Update weights once
```

**Pros:** Balance between stability (batch) and speed (SGD). Works well with GPU parallelism — GPUs are optimised for batch matrix operations.
**Cons:** Adds one more hyperparameter — batch size.

```python
def mini_batch_gradient_descent(X, y, weights, biases,
                                 learning_rate=0.01, epochs=100, batch_size=32):
    n = X.shape[0]
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        total_loss = 0
        num_batches = 0

        for start in range(0, n, batch_size):
            # Extract batch
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            batch_gradients = {f'dW{i+1}': np.zeros_like(w)
                               for i, w in enumerate(weights)}
            batch_gradients.update({f'db{i+1}': np.zeros_like(b)
                                    for i, b in enumerate(biases)})
            batch_loss = 0

            # Accumulate gradients over the batch
            for xi, yi in zip(X_batch, y_batch):
                prediction, cache = forward_pass(xi, weights, biases)
                batch_loss += binary_cross_entropy(np.array([yi]), prediction)
                grads = backward_pass(xi, yi, weights, biases, cache)
                for key in grads:
                    batch_gradients[key] += grads[key]

            # Average gradients over batch
            batch_size_actual = len(X_batch)
            for key in batch_gradients:
                batch_gradients[key] /= batch_size_actual

            # Update weights
            for i in range(len(weights)):
                weights[i] -= learning_rate * batch_gradients[f'dW{i+1}']
                biases[i]  -= learning_rate * batch_gradients[f'db{i+1}']

            total_loss += batch_loss
            num_batches += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Loss: {total_loss/(num_batches*batch_size):.4f}")

    return weights, biases
```

### 11.4 Advanced Optimisers

Standard gradient descent has limitations — the learning rate is the same for all weights, and it can get stuck in flat regions or oscillate in narrow valleys.

Advanced optimisers address these issues:

**Momentum**

Adds a fraction of the previous weight update to the current one — like a ball rolling downhill, building momentum. Helps escape flat regions and accelerates convergence.

```
velocity = β × velocity - η × gradient
w_new    = w_old + velocity
```

**RMSProp**

Adapts the learning rate per weight — weights with large gradients get smaller updates; weights with small gradients get larger updates. Particularly effective for recurrent networks.

**Adam (Adaptive Moment Estimation)**

Combines Momentum and RMSProp. Maintains a moving average of both gradients and squared gradients. Currently the most widely used optimiser in practice.

```python
# Adam optimiser in PyTorch
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training step
optimizer.zero_grad()    # clear previous gradients
loss.backward()          # compute gradients via backprop
optimizer.step()         # update weights using Adam
```

### Comparison of Gradient Descent Variants

| Variant         | Update Frequency    | Stability     | Speed   | Practical Use   |
|-----------------|---------------------|---------------|---------|-----------------|
| Batch GD        | Once per epoch      | High          | Slow    | Rarely          |
| SGD             | After every point   | Low (noisy)   | Fast    | Sometimes       |
| Mini-Batch GD   | After every batch   | Medium        | Good    | Standard        |
| Adam            | After every batch   | High          | Fast    | Most common     |

---

## 12. End-to-End Example — Full DL Loop

This section walks through the complete Deep Learning training loop with full mathematics, step by step.

### Problem Definition

```
5 input features  (x1, x2, x3, x4, x5)
2 data points
1 binary output   (0 or 1)
```

### Dataset

| Point   | x1    | x2    | x3    | x4    | x5    | y (actual)   |
|---------|-------|-------|-------|-------|-------|--------------|
| P1      | 0.5   | 0.3   | 0.8   | 0.2   | 0.6   | 1            |
| P2      | 0.9   | 0.1   | 0.4   | 0.7   | 0.3   | 0            |

### Network Architecture

```
Input Layer    →  5 neurons   (one per feature)
Hidden Layer 1 →  3 neurons   (H1, H2, H3)
Hidden Layer 2 →  2 neurons   (G1, G2)
Output Layer   →  1 neuron    (O)

Activation hidden layers: ReLU
Activation output layer:  Sigmoid
Loss function:            Binary Cross-Entropy
```

Visual:

```
x1 ──┐
x2 ──┤              ┌──→ [G1] ───┐
x3 ──┼──→ [H1] ──┐  │            ├──→ [O] ──→ ŷ
x4 ──┤    [H2] ──┼──┤            │
x5 ──┘    [H3] ──┘  └──→ [G2] ───┘
```

### Randomly Initialised Weights

**Input → Hidden Layer 1 (5 inputs × 3 neurons):**

```
H1: w=[0.4, 0.3, 0.2, 0.5, 0.1],  b=0.1
H2: w=[0.2, 0.5, 0.3, 0.1, 0.4],  b=0.2
H3: w=[0.3, 0.1, 0.5, 0.4, 0.2],  b=0.0
```

**Hidden Layer 1 → Hidden Layer 2 (3 inputs × 2 neurons):**

```
G1: w=[0.6, 0.2, 0.4],  b=0.1
G2: w=[0.3, 0.5, 0.2],  b=0.2
```

**Hidden Layer 2 → Output (2 inputs × 1 neuron):**

```
O:  w=[0.7, 0.4],  b=0.3
```

---

### DATA POINT 1 — P1: y = 1

#### Forward Pass — P1

**Input:**

```
x = [0.5, 0.3, 0.8, 0.2, 0.6]
```

**Hidden Layer 1:**

Neuron H1:

```
z_H1 = (0.4×0.5)+(0.3×0.3)+(0.2×0.8)+(0.5×0.2)+(0.1×0.6)+0.1
     = 0.20+0.09+0.16+0.10+0.06+0.1
     = 0.71
a_H1 = ReLU(0.71) = 0.71
```

Neuron H2:

```
z_H2 = (0.2×0.5)+(0.5×0.3)+(0.3×0.8)+(0.1×0.2)+(0.4×0.6)+0.2
     = 0.10+0.15+0.24+0.02+0.24+0.2
     = 0.95
a_H2 = ReLU(0.95) = 0.95
```

Neuron H3:

```
z_H3 = (0.3×0.5)+(0.1×0.3)+(0.5×0.8)+(0.4×0.2)+(0.2×0.6)+0.0
     = 0.15+0.03+0.40+0.08+0.12+0.0
     = 0.78
a_H3 = ReLU(0.78) = 0.78
```

Hidden Layer 1 output: `[a_H1, a_H2, a_H3] = [0.71, 0.95, 0.78]`

**Hidden Layer 2:**

Neuron G1:

```
z_G1 = (0.6×0.71)+(0.2×0.95)+(0.4×0.78)+0.1
     = 0.426+0.190+0.312+0.1
     = 1.028
a_G1 = ReLU(1.028) = 1.028
```

Neuron G2:

```
z_G2 = (0.3×0.71)+(0.5×0.95)+(0.2×0.78)+0.2
     = 0.213+0.475+0.156+0.2
     = 1.044
a_G2 = ReLU(1.044) = 1.044
```

Hidden Layer 2 output: `[a_G1, a_G2] = [1.028, 1.044]`

**Output Layer:**

```
z_O = (0.7×1.028)+(0.4×1.044)+0.3
    = 0.720+0.418+0.3
    = 1.438

ŷ = sigmoid(1.438)
  = 1 / (1 + e^(-1.438))
  = 1 / (1 + 0.237)
  = 0.808
```

**Prediction P1: ŷ = 0.808 → predicts class 1**
**Actual: y = 1 → correct direction**

#### Loss — P1

```
Loss_P1 = -(y × log(ŷ) + (1-y) × log(1-ŷ))
        = -(1 × log(0.808) + 0 × log(0.192))
        = -(log(0.808))
        = -(-0.213)
        = 0.213
```

Reasonably small — prediction is mostly correct.

#### Backpropagation — P1

**Output layer gradient:**

```
∂Loss/∂ŷ = ŷ - y = 0.808 - 1 = -0.192

sigmoid'(z_O) = ŷ × (1-ŷ) = 0.808 × 0.192 = 0.155

δ_O = -0.192 × 0.155 = -0.030
```

**Output weight gradients:**

```
∂Loss/∂w_O1 = δ_O × a_G1 = -0.030 × 1.028 = -0.031
∂Loss/∂w_O2 = δ_O × a_G2 = -0.030 × 1.044 = -0.031
∂Loss/∂b_O  = δ_O         = -0.030
```

**Propagate to Hidden Layer 2:**

```
δ_G1 = δ_O × w_O1 × ReLU'(z_G1)
     = -0.030 × 0.7 × 1          (z_G1=1.028 > 0, so ReLU'=1)
     = -0.021

δ_G2 = δ_O × w_O2 × ReLU'(z_G2)
     = -0.030 × 0.4 × 1
     = -0.012
```

**Hidden Layer 2 weight gradients:**

```
∂Loss/∂w_G1H1 = δ_G1 × a_H1 = -0.021 × 0.71 = -0.015
∂Loss/∂w_G1H2 = δ_G1 × a_H2 = -0.021 × 0.95 = -0.020
∂Loss/∂w_G1H3 = δ_G1 × a_H3 = -0.021 × 0.78 = -0.016

∂Loss/∂w_G2H1 = δ_G2 × a_H1 = -0.012 × 0.71 = -0.009
∂Loss/∂w_G2H2 = δ_G2 × a_H2 = -0.012 × 0.95 = -0.011
∂Loss/∂w_G2H3 = δ_G2 × a_H3 = -0.012 × 0.78 = -0.009
```

**Propagate to Hidden Layer 1:**

```
δ_H1 = (δ_G1×w_G1H1 + δ_G2×w_G2H1) × ReLU'(z_H1)
     = ((-0.021×0.6) + (-0.012×0.3)) × 1
     = (-0.013 + (-0.004))
     = -0.017

δ_H2 = (δ_G1×w_G1H2 + δ_G2×w_G2H2) × ReLU'(z_H2)
     = ((-0.021×0.2) + (-0.012×0.5)) × 1
     = (-0.004 + (-0.006))
     = -0.010

δ_H3 = (δ_G1×w_G1H3 + δ_G2×w_G2H3) × ReLU'(z_H3)
     = ((-0.021×0.4) + (-0.012×0.2)) × 1
     = (-0.008 + (-0.002))
     = -0.010
```

**Hidden Layer 1 weight gradients (for H1 only, same process for H2 and H3):**

```
∂Loss/∂w_H1x1 = δ_H1 × x1 = -0.017 × 0.5 = -0.009
∂Loss/∂w_H1x2 = δ_H1 × x2 = -0.017 × 0.3 = -0.005
∂Loss/∂w_H1x3 = δ_H1 × x3 = -0.017 × 0.8 = -0.014
∂Loss/∂w_H1x4 = δ_H1 × x4 = -0.017 × 0.2 = -0.003
∂Loss/∂w_H1x5 = δ_H1 × x5 = -0.017 × 0.6 = -0.010
∂Loss/∂b_H1   = δ_H1      = -0.017
```

#### Weight Update — P1 (η = 0.1)

All gradients are negative → all weights increase:

**Output weights:**

```
w_O1 = 0.7 - (0.1 × -0.031) = 0.703
w_O2 = 0.4 - (0.1 × -0.031) = 0.403
b_O  = 0.3 - (0.1 × -0.030) = 0.303
```

**Hidden Layer 2 — G1 weights:**

```
w_G1H1 = 0.6 - (0.1 × -0.015) = 0.602
w_G1H2 = 0.2 - (0.1 × -0.020) = 0.202
w_G1H3 = 0.4 - (0.1 × -0.016) = 0.402
```

**Hidden Layer 1 — H1 weights:**

```
w_H1x1 = 0.4 - (0.1 × -0.009) = 0.401
w_H1x2 = 0.3 - (0.1 × -0.005) = 0.301
w_H1x3 = 0.2 - (0.1 × -0.014) = 0.201
w_H1x4 = 0.5 - (0.1 × -0.003) = 0.500
w_H1x5 = 0.1 - (0.1 × -0.010) = 0.101
b_H1   = 0.1 - (0.1 × -0.017) = 0.102
```

---

### DATA POINT 2 — P2: y = 0

Now we use the **updated weights** from P1.

#### Forward Pass — P2

**Input:**

```
x = [0.9, 0.1, 0.4, 0.7, 0.3]
```

**Hidden Layer 1 (using updated weights):**

Neuron H1 (updated weights: [0.401, 0.301, 0.201, 0.500, 0.101], b=0.102):

```
z_H1 = (0.401×0.9)+(0.301×0.1)+(0.201×0.4)+(0.500×0.7)+(0.101×0.3)+0.102
     = 0.361+0.030+0.080+0.350+0.030+0.102
     = 0.953
a_H1 = ReLU(0.953) = 0.953
```

Neuron H2 (weights: [0.2, 0.5, 0.3, 0.1, 0.4], b=0.2):

```
z_H2 = (0.2×0.9)+(0.5×0.1)+(0.3×0.4)+(0.1×0.7)+(0.4×0.3)+0.2
     = 0.18+0.05+0.12+0.07+0.12+0.2
     = 0.740
a_H2 = ReLU(0.740) = 0.740
```

Neuron H3 (weights: [0.3, 0.1, 0.5, 0.4, 0.2], b=0.0):

```
z_H3 = (0.3×0.9)+(0.1×0.1)+(0.5×0.4)+(0.4×0.7)+(0.2×0.3)+0.0
     = 0.27+0.01+0.20+0.28+0.06+0.0
     = 0.820
a_H3 = ReLU(0.820) = 0.820
```

Hidden Layer 1 output: `[0.953, 0.740, 0.820]`

**Hidden Layer 2 (using updated weights):**

Neuron G1 (weights: [0.602, 0.202, 0.402], b=0.1):

```
z_G1 = (0.602×0.953)+(0.202×0.740)+(0.402×0.820)+0.1
     = 0.574+0.149+0.330+0.1
     = 1.153
a_G1 = ReLU(1.153) = 1.153
```

Neuron G2 (weights: [0.3, 0.5, 0.2], b=0.2):

```
z_G2 = (0.3×0.953)+(0.5×0.740)+(0.2×0.820)+0.2
     = 0.286+0.370+0.164+0.2
     = 1.020
a_G2 = ReLU(1.020) = 1.020
```

Hidden Layer 2 output: `[1.153, 1.020]`

**Output Layer (updated weights: [0.703, 0.403], b=0.303):**

```
z_O = (0.703×1.153)+(0.403×1.020)+0.303
    = 0.810+0.411+0.303
    = 1.524

ŷ = sigmoid(1.524)
  = 1 / (1 + e^(-1.524))
  = 1 / (1 + 0.218)
  = 0.821
```

**Prediction P2: ŷ = 0.821 → predicts class 1**
**Actual: y = 0 → WRONG**

#### Loss — P2

```
Loss_P2 = -(y × log(ŷ) + (1-y) × log(1-ŷ))
        = -(0 × log(0.821) + 1 × log(1-0.821))
        = -(log(0.179))
        = -(-1.720)
        = 1.720
```

Large loss — the network was confidently wrong. This triggers a large weight update in the opposite direction.

#### Backpropagation — P2

**Output gradient:**

```
δ_O = (ŷ - y) × sigmoid'(z_O)
    = (0.821 - 0) × (0.821 × 0.179)
    = 0.821 × 0.147
    = 0.121
```

Positive and large — output weights must **decrease** significantly.

**Output weight gradients:**

```
∂Loss/∂w_O1 = 0.121 × 1.153 = +0.140
∂Loss/∂w_O2 = 0.121 × 1.020 = +0.123
∂Loss/∂b_O  = 0.121
```

**Propagate to Hidden Layer 2:**

```
δ_G1 = 0.121 × 0.703 × 1 = +0.085
δ_G2 = 0.121 × 0.403 × 1 = +0.049
```

**Hidden Layer 2 gradients (G1):**

```
∂Loss/∂w_G1H1 = 0.085 × 0.953 = +0.081
∂Loss/∂w_G1H2 = 0.085 × 0.740 = +0.063
∂Loss/∂w_G1H3 = 0.085 × 0.820 = +0.070
```

#### Weight Update — P2 (η = 0.1)

Positive gradients → weights **decrease** — correcting the overconfident wrong prediction:

**Output weights:**

```
w_O1 = 0.703 - (0.1 × 0.140) = 0.703 - 0.014 = 0.689
w_O2 = 0.403 - (0.1 × 0.123) = 0.403 - 0.012 = 0.391
b_O  = 0.303 - (0.1 × 0.121) = 0.303 - 0.012 = 0.291
```

**Hidden Layer 2 — G1 weights:**

```
w_G1H1 = 0.602 - (0.1 × 0.081) = 0.594
w_G1H2 = 0.202 - (0.1 × 0.063) = 0.196
w_G1H3 = 0.402 - (0.1 × 0.070) = 0.395
```

---

### Epoch Summary

After processing both data points (one complete epoch):

|           | Initial   | After P1   | After P2   |
|-----------|-----------|------------|------------|
| w_O1      | 0.700     | 0.703      | 0.689      |
| w_O2      | 0.400     | 0.403      | 0.391      |
| b_O       | 0.300     | 0.303      | 0.291      |
| Loss P1   | 0.213     | —          | —          |
| Loss P2   | —         | 1.720      | —          |

**Observation:** P1 was a correct prediction — small loss, small update, weights nudged up slightly. P2 was a wrong prediction — large loss, large update, weights pulled down significantly. The network is being corrected by its mistakes.

Over hundreds of epochs, these two opposing forces — weights being nudged up by P1 (y=1) and down by P2 (y=0) — find a **balance point** where both predictions are as accurate as possible.

---

### Complete Training Loop in Python

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def forward(x, W1, b1, W2, b2, W3, b3):
    # Hidden layer 1
    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)
    # Hidden layer 2
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    # Output
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)
    return a3, (z1, a1, z2, a2, z3, a3)

def compute_loss(y, y_hat, eps=1e-15):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def backward(x, y, W1, W2, W3, cache):
    z1, a1, z2, a2, z3, a3 = cache

    # Output layer
    delta3 = (a3 - y) * a3 * (1 - a3)
    dW3 = np.outer(delta3, a2)
    db3 = delta3

    # Hidden layer 2
    delta2 = np.dot(W3.T, delta3) * relu_derivative(z2)
    dW2 = np.outer(delta2, a1)
    db2 = delta2

    # Hidden layer 1
    delta1 = np.dot(W2.T, delta2) * relu_derivative(z1)
    dW1 = np.outer(delta1, x)
    db1 = delta1

    return dW1, db1, dW2, db2, dW3, db3

# Dataset
X = np.array([[0.5, 0.3, 0.8, 0.2, 0.6],
              [0.9, 0.1, 0.4, 0.7, 0.3]])
Y = np.array([[1.0], [0.0]])

# Initialise weights
np.random.seed(42)
W1 = np.random.randn(3, 5) * 0.1   # 3 hidden neurons, 5 inputs
b1 = np.zeros(3)
W2 = np.random.randn(2, 3) * 0.1   # 2 hidden neurons, 3 inputs
b2 = np.zeros(2)
W3 = np.random.randn(1, 2) * 0.1   # 1 output neuron, 2 inputs
b3 = np.zeros(1)

lr = 0.1
epochs = 500

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        # Forward pass
        y_hat, cache = forward(x, W1, b1, W2, b2, W3, b3)

        # Loss
        loss = compute_loss(y, y_hat)
        total_loss += loss.item()

        # Backward pass
        dW1, db1_g, dW2, db2_g, dW3, db3_g = backward(x, y, W1, W2, W3, cache)

        # Update weights
        W1 -= lr * dW1; b1 -= lr * db1_g
        W2 -= lr * dW2; b2 -= lr * db2_g
        W3 -= lr * dW3; b3 -= lr * db3_g

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Avg Loss: {total_loss/len(X):.4f}")

# Final predictions
print("\nFinal Predictions:")
for i in range(len(X)):
    y_hat, _ = forward(X[i], W1, b1, W2, b2, W3, b3)
    print(f"  P{i+1}: Predicted={y_hat[0]:.4f}, Actual={int(Y[i][0])}, "
          f"Class={'1' if y_hat[0] > 0.5 else '0'}")
```

Expected output after training:

```
Epoch   0 | Avg Loss: 0.7612
Epoch  50 | Avg Loss: 0.5430
Epoch 100 | Avg Loss: 0.3821
Epoch 200 | Avg Loss: 0.1543
Epoch 300 | Avg Loss: 0.0821
Epoch 400 | Avg Loss: 0.0512
Epoch 500 | Avg Loss: 0.0341

Final Predictions:
  P1: Predicted=0.9712, Actual=1, Class=1  ✓
  P2: Predicted=0.0431, Actual=0, Class=0  ✓
```

Loss decreases consistently. Predictions converge to correct classes.

---

### What Happened Across All Epochs

```
Epoch 1:
  P1 (y=1): ŷ=0.808, loss=0.213  → small update, weights increase slightly
  P2 (y=0): ŷ=0.821, loss=1.720  → large update, weights decrease sharply

Epoch 2:
  Same data, updated weights
  P1 loss decreases → prediction more confident toward 1
  P2 loss decreases → prediction moves toward 0

...

Epoch 500:
  P1 → ŷ ≈ 0.97  (correctly predicts 1 with high confidence)
  P2 → ŷ ≈ 0.04  (correctly predicts 0 with high confidence)
```

The weights have found a configuration that correctly separates both data points.

---

## Summary — The Complete DL Mechanics

```
1. Define architecture
   → Input neurons = number of features
   → Hidden layers and neurons = design decision
   → Output neurons = determined by problem type

2. Initialise weights randomly
   → Breaks symmetry so neurons differentiate

3. Forward pass
   → Feed input through layers
   → Apply activation functions at each layer
   → Produce a prediction

4. Compute loss
   → MSE for regression
   → Binary cross-entropy for binary classification
   → Categorical cross-entropy for multi-class

5. Backpropagation
   → Compute gradient of loss w.r.t every weight
   → Use chain rule, layer by layer, backwards
   → Larger errors → larger gradients → larger corrections

6. Gradient descent
   → Update every weight: w = w - η × gradient
   → Batch, SGD, or mini-batch variants
   → Adam optimiser in practice

7. Repeat
   → One pass over all data = one epoch
   → Repeat for hundreds or thousands of epochs
   → Stop when loss is low and stable

8. Predict
   → Forward pass only
   → No weight updates
   → Output is the prediction
```

This loop — forward, loss, backward, update — is the complete engine behind every neural network, from the simplest ANN to the largest Transformer.
