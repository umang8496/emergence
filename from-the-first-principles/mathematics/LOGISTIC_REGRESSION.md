# Logistic Regression: The Mathematics

## Part 1: The Sigmoid Function - Our Foundation

The sigmoid function is the mathematical heart of logistic regression. Let us understand it completely.

### Definition

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where `e` is Euler's number (approximately 2.718).

We can rewrite this equivalently as:

$$\sigma(z) = \frac{e^z}{1 + e^z}$$

Both forms are identical mathematically.

In terms of hyperbolic tangent:

$$\sigma(z) = \frac{1 + \tanh(z/2)}{2}$$

Or  

$$\sigma(z) = \frac{\tanh(z/2)}{2} + \frac{1}{2}$$

### Why This Function?

The sigmoid function has properties we need:

#### Property 1: Range is (0, 1)

For any value of `z` (from -∞ to +∞), the sigmoid output is always between 0 and 1.

Proof:

- When z → -∞: e^(-z) → ∞, so σ(z) → 1/(1+∞) = 0
- When z → +∞: e^(-z) → 0, so σ(z) → 1/(1+0) = 1

This is exactly what we need for probabilities.

#### Property 2: Smooth and Differentiable

The sigmoid function is smooth (no sharp corners). Its derivative is simple:

**Starting with the sigmoid function:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Taking the derivative with respect to z:**

Using the chain rule, we treat this as a composition: $(1 + e^{-z})^{-1}$

$$\frac{d\sigma}{dz} = \frac{d}{dz}(1 + e^{-z})^{-1}$$

Or

$$\frac{d\sigma}{dz} = \frac{d(1 + e^{-z})^{-1}}{d(1 + e^{-z})}.\frac{d(1 + e^{-z})}{dz}$$

**Applying the chain rule:**

$$\frac{d\sigma}{dz} = -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z})$$

**Computing the inner derivative:**

$$\frac{d}{dz}(1 + e^{-z}) = -e^{-z}$$

**Substituting back:**

$$\frac{d\sigma}{dz} = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z})$$

$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1 + e^{-z})^2}$$

**Now, expressing this in terms of σ(z):**

We can rewrite this fraction cleverly. Notice that:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Rearranging:
$$1 + e^{-z} = \frac{1}{\sigma(z)}$$

$$e^{-z} = \frac{1}{\sigma(z)} - 1 = \frac{1 - \sigma(z)}{\sigma(z)}$$

**Substituting into our derivative:**

$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1 + e^{-z})^2}$$

$$= \frac{\frac{1 - \sigma(z)}{\sigma(z)}}{\left(\frac{1}{\sigma(z)}\right)^2}$$

$$= \frac{1 - \sigma(z)}{\sigma(z)} \cdot \sigma(z)^2$$

$$= (1 - \sigma(z)) \cdot \sigma(z)$$

**Final result:**

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

**Why this is beautiful:**

1. The derivative is expressed in terms of the function itself (no exponentials!)
2. The derivative is always positive: $0 < \sigma(z) < 1$ and $0 < (1-\sigma(z)) < 1$
3. Maximum derivative occurs at z = 0 where $\sigma'(0) = 0.5 \times 0.5 = 0.25$
4. The derivative approaches 0 as z → ±∞ (the curve flattens)

This smoothness makes gradient descent work beautifully.  
We'll use this derivative when computing gradients in the backpropagation process.

#### Property 3: Symmetric Around 0.5

$$σ(-z) = 1 - σ(z)$$

This symmetry is mathematically elegant and useful in our calculations.

### Visualizing the Sigmoid

Let us compute values at key points:

```text
z = -5:  σ(-5) = 1 / (1 + e^5) ≈ 1 / (1 + 148) ≈ 0.007
z = -2:  σ(-2) = 1 / (1 + e^2) ≈ 1 / (1 + 7.39) ≈ 0.119
z = -1:  σ(-1) = 1 / (1 + e^1) ≈ 1 / (1 + 2.72) ≈ 0.269
z =  0:  σ(0)  = 1 / (1 + e^0) = 1 / (1 + 1) = 0.5
z =  1:  σ(1)  = 1 / (1 + e^(-1)) ≈ 1 / (1 + 0.368) ≈ 0.731
z =  2:  σ(2)  = 1 / (1 + e^(-2)) ≈ 1 / (1 + 0.135) ≈ 0.881
z =  5:  σ(5)  = 1 / (1 + e^(-5)) ≈ 1 / (1 + 0.007) ≈ 0.993
```

The curve is S-shaped. Around z = 0, the change is steepest.  
Far from zero (very positive or very negative), the function flattens.  

---

## Part 2: Our Hypothesis Function

In linear regression, our hypothesis was:

```text
h(x) = w·x + b
```

In logistic regression, we wrap this in our sigmoid:

```text
h(x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))
```

Let's denote `z = w·x + b` for clarity:

```text
h(x) = σ(z) where z = w·x + b
```

### What This Means

Our hypothesis function outputs a probability:

```text
h(x) = P(y = 1 | x)
```

This reads as: "The probability that y equals 1 (positive class), given input x."

By complement:

```text
P(y = 0 | x) = 1 - h(x)
```

For example, if h(x) = 0.8, we interpret this as:

- 80% probability the example belongs to class 1
- 20% probability the example belongs to class 0

---

## Part 3: The Cost Function - Binary Cross-Entropy

In linear regression, we minimized Mean Squared Error:

```text
J = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ)²
```

For logistic regression, we use a different cost function designed for probabilities: **Binary Cross-Entropy** (Log Loss).

### The Formula

```text
J(w, b) = -(1/m) * Σᵢ₌₁ᵐ [yᵢ * log(hᵢ) + (1 - yᵢ) * log(1 - hᵢ)]
```

Where:

- `m` = number of training examples
- `yᵢ` = actual label (0 or 1)
- `hᵢ` = predicted probability from our sigmoid
- `log` = natural logarithm (ln)

### Breaking It Down

Let's understand what happens in each case:

#### Case 1: When yᵢ = 1 (actual positive class)

The second term becomes zero: `(1-1) * log(1-hᵢ) = 0`

We're left with:

```text
Cost = -log(hᵢ)
```

Now:

- If hᵢ = 0.9 (we predict high probability): Cost = -log(0.9) ≈ 0.105 (small, good)
- If hᵢ = 0.5 (we're uncertain): Cost = -log(0.5) ≈ 0.693 (medium)
- If hᵢ = 0.1 (we predict low probability): Cost = -log(0.1) ≈ 2.303 (large, bad)

So when the actual class is 1, the cost is low when we predict high probability and high when we predict low probability. This makes sense.

#### Case 2: When yᵢ = 0 (actual negative class)

The first term becomes zero: `0 * log(hᵢ) = 0`

We're left with:

```text
Cost = -log(1 - hᵢ)
```

Now:

- If hᵢ = 0.9 (we predict high probability): Cost = -log(0.1) ≈ 2.303 (large, bad)
- If hᵢ = 0.5 (we're uncertain): Cost = -log(0.5) ≈ 0.693 (medium)
- If hᵢ = 0.1 (we predict low probability): Cost = -log(0.9) ≈ 0.105 (small, good)

When the actual class is 0, the cost is low when we predict low probability and high when we predict high probability. Again, this makes intuitive sense.

### Why This Cost Function?

We could ask: why not use Mean Squared Error?

The answer lies in **likelihood maximization**.  
Cross-entropy is derived from maximum likelihood estimation.  
It's the natural cost function for classification because:

1. **It's convex** — has a single global minimum
2. **It's differentiable** — we can use gradient descent
3. **It penalizes confidently wrong predictions heavily** — if we're sure but wrong, we pay a steep price
4. **It's calibrated for probabilities** — it naturally handles the 0-1 range

---

## Part 4: Computing Gradients

Now we need to find how to update our weights and bias to minimize our cost function.  
This requires computing gradients.

### The Gradient for Weights

Let us compute the derivative of our cost function with respect to weight w:

```text
∂J/∂w = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ) * xᵢ
```

This is remarkably similar to linear regression!  
The only difference is that `hᵢ` now includes the sigmoid.

### The Gradient for Bias

```text
∂J/∂b = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ)
```

Again, similar to linear regression.

### Derivation (Optional Deep Dive)

For those interested in how we derive these gradients:

The cost for one example when y=1:

```text
J₁ = -log(h) = -log(σ(z))
```

Where `z = w·x + b` and `h = σ(z) = 1 / (1 + e^(-z))`

Using the chain rule:

```text
∂J₁/∂w = ∂J₁/∂h * ∂h/∂z * ∂z/∂w
```

Computing each part:

```text
∂J₁/∂h = -1/h

∂h/∂z = σ(z) * (1 - σ(z)) = h * (1 - h)    [sigmoid derivative]

∂z/∂w = x
```

Multiplying them:

```text
∂J₁/∂w = (-1/h) * (h * (1-h)) * x
       = -(1-h) * x
       = (h - 1) * x    [when y = 1]
```

Similarly, when y=0, we get:

```text
∂J₀/∂w = h * x    [when y = 0]
```

Combining both cases (which is what the cross-entropy formula naturally does):

```text
∂J/∂w = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ) * xᵢ
```

This is remarkable: **the gradient formula is identical to linear regression**, even though our hypothesis function and cost function are different!

---

## Part 5: The Update Rule - Gradient Descent

With our gradients computed, we update our parameters:

```text
w := w - α * ∂J/∂w = w - α * (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ) * xᵢ

b := b - α * ∂J/∂b = b - α * (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ)
```

Where `α` is our learning rate.

This is exactly the same update rule as linear regression. Our intuition fully transfers:

- Learning rate too high → divergence
- Learning rate too low → slow convergence
- We repeat until convergence

---

## Part 6: The Training Algorithm - Complete

Let us write out our full training algorithm:

```text
Initialize w and b to small random values or zeros

For iteration = 1 to num_iterations:
    
    For each training example i:
        Compute zᵢ = w·xᵢ + b
        Compute hᵢ = σ(zᵢ) = 1 / (1 + e^(-zᵢ))
    
    Compute cost:
        J = -(1/m) * Σᵢ₌₁ᵐ [yᵢ * log(hᵢ) + (1-yᵢ) * log(1-hᵢ)]
    
    Compute gradients:
        ∂J/∂w = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ) * xᵢ
        ∂J/∂b = (1/m) * Σᵢ₌₁ᵐ (hᵢ - yᵢ)
    
    Update parameters:
        w := w - α * ∂J/∂w
        b := b - α * ∂J/∂b
    
    If J is not decreasing or change is very small:
        Break (converged)

Return learned w and b
```

---

## Part 7: Prediction and Classification

Once we've trained our model (found optimal w and b), we make predictions:

### Probability Prediction

```text
h(x_new) = 1 / (1 + e^(-(w·x_new + b)))
```

This gives us the probability that the example belongs to class 1.

### Classification

To make a binary decision, we compare against a threshold (usually 0.5):

```text
If h(x) ≥ 0.5: predict class 1
If h(x) < 0.5:  predict class 0
```

But we could choose different thresholds:

- Conservative: threshold = 0.7 (only classify as 1 when very confident)
- Aggressive: threshold = 0.3 (classify as 1 when moderately confident)

---

## Part 8: Log-Odds and Weight Interpretation

There's a beautiful mathematical relationship in logistic regression that helps us interpret weights.

### The Log-Odds Formula

If we rearrange our sigmoid function:

Starting with:

```text
h = σ(z) = 1 / (1 + e^(-z))
```

We can derive:

```text
log(h / (1-h)) = z = w·x + b
```

The left side is called the **log-odds** or **logit** of h.

### Interpreting Weights

This means:

```text
log(P(y=1) / P(y=0)) = w·x + b
```

So weight `wⱼ` represents: "How much does feature j increase the log-odds of being in class 1?"

If `w₁ = 0.5`:

- Increasing feature 1 by 1 unit increases the log-odds by 0.5
- Which increases the odds by a factor of e^0.5 ≈ 1.65

This interpretation is more complex than linear regression (where a weight is a direct contribution), but it's mathematically meaningful.

---

## Part 9: Convergence Properties

**Question:** Does logistic regression always converge?

**Answer:** Not always, but usually.

### When It Converges

The cost function (binary cross-entropy with sigmoid) is **convex** when we use all the data (batch gradient descent). This means:

- There's a single global minimum
- Gradient descent will reach it (eventually)
- The cost always decreases (or stays same)

### When It Struggles

Convergence can be slow if:

1. **Learning rate is too small** — takes forever
2. **Features are unscaled** — gradients have very different magnitudes
3. **Classes are imbalanced** — one class dominates
4. **Linearly separable data** — weights can grow unbounded while loss stays near zero

---

## Part 10: Comparison with Linear Regression

Let us create a mathematical comparison:

| Aspect         | Linear Regression       | Logistic Regression                    |
|----------------|-------------------------|----------------------------------------|
| Hypothesis     | h = w·x + b             | h = σ(w·x + b)                         |
| Output Range   | (-∞, ∞)                 | (0, 1)                                 |
| Cost Function  | `(1/2m) * Σ(h - y)²`    | `-(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]`|
| Gradient for w | `(1/m) * Σ(h - y) * x`  | `(1/m) * Σ(h - y) * x`                 |
| Update Rule    | w := w - α * gradient   | w := w - α * gradient                  |
| Convergence    | Always (convex)         | Mostly (convex)                        |
| Interpretation | w = direct contribution | w = log-odds contribution              |

Notice: **The gradient and update rule are identical!** The only differences are the hypothesis function and cost function.

---

## Part 11: The Decision Boundary

One concept unique to logistic regression is the **decision boundary**.

### The Boundary Equation

We classify as class 1 when:

```text
h(x) ≥ 0.5
```

Which means:

```text
σ(w·x + b) ≥ 0.5
```

This happens when:

```text
w·x + b ≥ 0
```

(Because σ(0) = 0.5, and σ is monotonically increasing)

So our decision boundary is:

```text
w·x + b = 0
```

This is a hyperplane in our feature space!

### Example with 2 Features

If we have 2 features and learn `w = [2, 3]` and `b = -1`, our decision boundary is:

```text
2*x₁ + 3*x₂ - 1 = 0
```

Or rearranged:

```text
x₂ = (1 - 2*x₁) / 3
```

This is a straight line in 2D space. Examples on one side are class 0, the other side are class 1.

---

## Part 12: Why Cross-Entropy Makes Sense

Let us derive cross-entropy from first principles using **maximum likelihood**.

Assuming each example is independent and follows a Bernoulli distribution:

```text
P(y | x) = h(x)^y * (1 - h(x))^(1-y)
```

The likelihood of our entire dataset:

```text
L = ∏ᵢ₌₁ᵐ P(yᵢ | xᵢ) = ∏ᵢ₌₁ᵐ hᵢ^yᵢ * (1-hᵢ)^(1-yᵢ)
```

Taking the log (for computational convenience):

```text
log(L) = Σᵢ₌₁ᵐ [yᵢ * log(hᵢ) + (1-yᵢ) * log(1-hᵢ)]
```

We want to **maximize** this log-likelihood, which is equivalent to **minimizing** its negative:

```text
J = -log(L) = -(1/m) * Σᵢ₌₁ᵐ [yᵢ * log(hᵢ) + (1-yᵢ) * log(1-hᵢ)]
```

This is exactly our cross-entropy cost function!

So cross-entropy isn't arbitrary—it emerges naturally from the principle of maximum likelihood estimation.  
We're finding the weights that make our observed data most likely.  

---

## Part 13: Numerical Stability Consideration

One practical concern: what happens when h approaches 0 or 1?

```text
When h = 0.9999: -log(h) ≈ 0.00005 (very small, fine)
When h = 0.0001: -log(h) ≈ 9.21 (large, fine)
When h = 1: -log(h) = 0 (fine)
When h = 0: -log(h) = ∞ (problem!)
```

In practice, the sigmoid never reaches exactly 0 or 1, but with extreme weights, it can get very close. To avoid numerical issues, we clip probabilities:

```text
h = max(min(h, 1 - 1e-7), 1e-7)
```

This ensures we never take log(0).

---

## Summary of Mathematical Concepts

Let us review what we've covered:

✓ **Sigmoid function:** Transforms unbounded numbers into probabilities (0, 1)

✓ **Hypothesis function:** h(x) = σ(w·x + b) outputs probability of class 1

✓ **Binary cross-entropy:** Cost function designed for classification

✓ **Gradients:** Computed using chain rule, remarkably similar to linear regression

✓ **Gradient descent:** Same optimization algorithm, same update rules

✓ **Decision boundary:** The hyperplane where h(x) = 0.5

✓ **Log-odds interpretation:** Weights affect log-odds, not direct values

✓ **Maximum likelihood:** Cross-entropy emerges from likelihood maximization

✓ **Numerical stability:** Must clip probabilities to avoid log(0)

---
