# ML Fundamentals: Filling the Knowledge Gaps

These concepts come up repeatedly in machine learning and are worth understanding deeply before moving on.

---

## 1. What is Validation Error?

When you train a model, you have two sets of data:

- **Training set** — the data the model learns from
- **Validation set** — a held-out set the model never sees during training

**Training error** measures how well the model fits the data it was trained on.  
**Validation error** measures how well the model performs on new, unseen data.

The goal is not to minimize training error — it is to minimize validation error, because that is what tells you how the model will behave in the real world.

### Why they can differ

If your model memorized the training data (instead of learning the pattern), it will have low training error but high validation error.  
If your model is too simple to capture the pattern, both errors will be high.

### A concrete example

```text
Training set:   1000 house prices → model sees these
Validation set:  200 house prices → model is tested on these (never seen before)

Training error:  $8,000  (model fits training data well)
Validation error: $25,000 (model generalizes poorly)
```

The gap between the two is the key diagnostic signal.

---

## 2. Overfitting and Underfitting

These are the two failure modes of any machine learning model.

### Overfitting

The model learns the training data **too well** — including its noise and random fluctuations.  
It mistakes noise for signal, so it performs poorly on new data.

**Signs:**

- Very low training error
- Much higher validation error
- The model is too complex relative to the data

**Analogy:** A student who memorizes every past exam question word-for-word but can't solve a new question phrased differently.

### Underfitting

The model is **too simple** to capture the underlying pattern.  
It fails on both training and validation data.

**Signs:**

- High training error
- High validation error (similar to training error)
- The model lacks the capacity to learn the relationship

**Analogy:** A student who only learned "price goes up with size" but ignored all other factors.

### The balance

| Property         | Underfitting                          | Sweet Spot                                 | Overfitting                        |
|------------------|---------------------------------------|--------------------------------------------|------------------------------------|
| Model complexity | Low                                   | Balanced                                   | High                               |
| Training error   | High                                  | Low                                        | Low                                |
| Validation error | High                                  | Low                                        | High (but for a different reason)  |
| Goal             | —                                     | Training ≈ Validation, both reasonably low |                                    |

---

## 3. High-Bias and Low-Variance

These terms describe the source of a model's errors and its behaviour across different datasets.

### Bias

**Bias** is the error introduced by a model that is too simple to capture the true pattern.

A high-bias model makes strong, rigid assumptions.  
It consistently misses the target in the same direction — it's systematically wrong.  

**High-bias example:**  
A linear model trying to fit a curved (quadratic) relationship.  
No matter how much data you give it, it will always underfit because it cannot represent curves.

### Variance

**Variance** is how much the model's predictions change when trained on different subsets of data.

A high-variance model is very sensitive to the specific training data it saw.  
It fits training data well but changes dramatically with new data — it's unstable.

**High-variance example:**  
A very deep decision tree that perfectly fits 1000 training examples.  
Retrain it on a slightly different 1000 examples and the predictions shift significantly.

### The trade-off

| Model Type     | Bias   | Variance | Typical Problem    |
|----------------|--------|----------|--------------------|
| Too simple     | High   | Low      | Underfitting       |
| Too complex    | Low    | High     | Overfitting        |
| Well-balanced  | Low    | Low      | Good generalisation|

### Where does linear regression sit?

Linear regression is a **high-bias, low-variance** model by nature.

- It makes a strong assumption: the relationship is linear.
- But its predictions are stable — train it on different subsets of similar data and the weights won't jump around wildly.

This is why linear regression rarely overfits dramatically, but often underfits when the true relationship is complex.

---

## 4. What is Standard Deviation?

Standard deviation (σ) measures **how spread out values are around the mean**.

A small standard deviation means values are clustered tightly around the mean.  
A large standard deviation means values are spread far apart.

### Formula

$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}$$

Where:

- $x_i$ = each individual value
- $\mu$ = mean of all values
- $n$ = number of values

### Step-by-step example

House sizes: `[1000, 1200, 1500, 1800, 2000]`

**Step 1 — Calculate the mean:**

$$\mu = \frac{1000 + 1200 + 1500 + 1800 + 2000}{5} = 1500$$

**Step 2 — Find the difference from mean for each value:**

```text
1000 - 1500 = -500
1200 - 1500 = -300
1500 - 1500 =    0
1800 - 1500 =  300
2000 - 1500 =  500
```

**Step 3 — Square each difference:**

```text
(-500)² = 250,000
(-300)² =  90,000
    0²  =       0
  300²  =  90,000
  500²  = 250,000
```

**Step 4 — Average the squared differences (this is the variance):**

$$\text{Variance} = \frac{250000 + 90000 + 0 + 90000 + 250000}{5} = 136{,}000$$

**Step 5 — Take the square root (this is the standard deviation):**

$$\sigma = \sqrt{136000} \approx 368.78 \text{ sqft}$$

**Interpretation:** On average, house sizes deviate from the mean by about 369 sqft.

---

## 5. Standard Deviation vs Variance

They both measure the same thing — **spread** — but in different units and at different scales.

### Variance

$$\text{Variance} = \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

- Units are **squared** (e.g., sqft² for house sizes, $² for prices)
- Amplifies large deviations because it squares them
- Useful in mathematics and statistics because it has nice algebraic properties
- Hard to interpret intuitively because of the squared units

### Standard Deviation

$$\sigma = \sqrt{\text{Variance}}$$

- Units are the **same as the original data** (sqft, $, etc.)
- Directly interpretable: "the typical deviation from the mean is X"
- More intuitive for human understanding

### Side-by-side comparison

| Property               | Variance                       | Standard Deviation            |
|------------------------|--------------------------------|-------------------------------|
| Formula                | $\frac{1}{n}\sum(x_i - \mu)^2$ | $\sqrt{\text{Variance}}$      |
| Units                  | Squared units                  | Same as original data         |
| Scale                  | Larger number                  | Smaller, interpretable        |
| Used for               | Math/statistics                | Communication/intuition       |
| Sensitive to outliers  | Very (squares them)            | Less so (square root dampens) |

### Which to use?

Use **variance** when doing calculations (it has cleaner math properties).  
Use **standard deviation** when communicating results to humans (it is in the same units as the data).

### Why this matters in machine learning

Feature normalization uses standard deviation:

$$x_{\text{normalized}} = \frac{x - \mu}{\sigma}$$

This scales every feature so that it has mean 0 and standard deviation 1.  
Features with very different ranges (e.g., 0–5000 sqft vs 1–10 rooms) become comparable,  
which is exactly why gradient descent converges much faster after normalization.

---

## 6. What is Normalization?

Normalization rescales feature values into a **fixed range**, typically [0, 1].  
It answers the question: "Where does this value sit within the observed range of the data?"

### Formula (Min-Max Scaling)

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### Step-by-step example

House sizes: `[1000, 1200, 1500, 1800, 2000]`

- $x_{\min} = 1000$, $x_{\max} = 2000$

```text
1000 → (1000 - 1000) / (2000 - 1000) = 0.00
1200 → (1200 - 1000) / (2000 - 1000) = 0.20
1500 → (1500 - 1000) / (2000 - 1000) = 0.50
1800 → (1800 - 1000) / (2000 - 1000) = 0.80
2000 → (2000 - 1000) / (2000 - 1000) = 1.00
```

Every value is now between 0 and 1.

### When to use it

- When the algorithm requires bounded inputs (e.g., neural networks with sigmoid activation)
- When you know and trust the min and max of the data
- When the distribution of values is roughly uniform

### Its weakness

Normalization is **sensitive to outliers**.  
If one house is 10,000 sqft, everything else gets squeezed into a tiny range near 0.

---

## 7. What is Standardization?

Standardization rescales feature values so the data has **mean = 0 and standard deviation = 1**.  
It answers the question: "How many standard deviations away from the mean is this value?"

### Formula (Z-Score Scaling)

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

Where:

- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

### Step-by-step example

House sizes: `[1000, 1200, 1500, 1800, 2000]`

From section 4: $\mu = 1500$, $\sigma \approx 368.78$

```text
1000 → (1000 - 1500) / 368.78 ≈ -1.36
1200 → (1200 - 1500) / 368.78 ≈ -0.81
1500 → (1500 - 1500) / 368.78 =  0.00
1800 → (1800 - 1500) / 368.78 ≈  0.81
2000 → (2000 - 1500) / 368.78 ≈  1.36
```

Negative values are below the mean, positive values are above it.  
The result is centered at 0 and spread symmetrically around it.

### When to use it

- When features have different units or ranges (the most common ML scenario)
- When the algorithm is sensitive to feature magnitude (linear regression, gradient descent, SVMs)
- When the data has outliers — standardization handles them more gracefully than normalization

### Why it handles outliers better

Normalization uses min and max, which are directly pulled to the extremes by outliers.  
Standardization uses mean and standard deviation, which are influenced by outliers but not completely dominated by them.

---

## 8. Normalization vs Standardization

Both techniques serve the same purpose — making features comparable in scale — but they work differently and suit different situations.

### Side-by-side comparison

| Property              | Normalization (Min-Max)                  | Standardization (Z-Score)                   |
|-----------------------|------------------------------------------|---------------------------------------------|
| Formula               | $(x - x_{\min}) / (x_{\max} - x_{\min})$ | $(x - \mu) / \sigma$                        |
| Output range          | Always [0, 1]                            | No fixed range; roughly [-3, 3] in practice |
| Mean of output        | Not guaranteed to be 0                   | Always 0                                    |
| Std dev of output     | Not guaranteed to be 1                   | Always 1                                    |
| Sensitive to outliers | Yes — outliers distort the range         | Less so — outliers affect mean/std mildly   |
| Requires knowing      | Min and max                              | Mean and standard deviation                 |
| Best suited for       | Bounded inputs, uniform distributions    | Gradient descent, most ML algorithms        |

### How they are related

Both are linear transformations of the original data.  
Both preserve the shape of the original distribution — they do not change whether data is skewed or symmetric.  
The difference is purely in what reference points they use: min/max vs mean/std.

### Which one should you use in practice?

For most machine learning tasks, **standardization is the safer default**.

- It works well even when you do not know the true bounds of the data
- New data can fall outside the training min/max (normalization breaks down), but it rarely falls more than a few standard deviations from the training mean
- Gradient descent converges faster with standardized features because all gradients are in comparable units

Use normalization when:

- The algorithm explicitly requires inputs in [0, 1]
- You know the data is bounded and outlier-free
- You are working with image pixel values (0–255 → 0.0–1.0)

### A visual intuition

```text
Original data:    [1000, 1200, 1500, 1800, 2000]   (sqft, large numbers, different scale)

After Normalization:  [0.00, 0.20, 0.50, 0.80, 1.00]   (compressed into [0,1])
After Standardization: [-1.36, -0.81, 0.00, 0.81, 1.36]  (centered at 0, spread by 1σ units)
```

Both versions are now compatible with other features of different scales.

---

## Quick Reference

| Concept            | One-line summary                                                                  |
|--------------------|-----------------------------------------------------------------------------------|
| Validation error   | How well the model performs on data it never saw during training                  |
| Overfitting        | Model memorized training data; fails on new data                                  |
| Underfitting       | Model too simple; fails on both training and new data                             |
| High bias          | Model makes wrong assumptions; consistently misses the pattern                    |
| Low variance       | Model is stable; predictions don't change much across datasets                    |
| Standard deviation | Average distance of values from the mean, in original units                       |
| Variance           | Same as above, but squared — used in calculations, not communication              |
| Normalization      | Rescales values to [0, 1] using min and max; sensitive to outliers                |
| Standardization    | Rescales values to mean=0, std=1 using z-score; robust default for most ML tasks  |

---
