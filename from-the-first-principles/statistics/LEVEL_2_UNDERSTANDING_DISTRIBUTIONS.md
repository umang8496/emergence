# Level 2: Understanding Distributions

## Overview

Level 0 taught you what data is.
Level 1 taught you how to describe individual variables (center, spread, shape).
Level 2 teaches you about **distributions**—the patterns underlying data.

A distribution is a complete description of:

- Which values are common (high probability)
- Which values are rare (low probability)
- What the overall pattern looks like
- Where values tend to cluster

**Why distributions matter:**

- Some ML models assume specific distributions
- Different distributions need different transformations
- Understanding distributions helps you recognize data patterns
- Distributions form the foundation of probability and statistics

---

## 2.1 What is a Distribution?

### Intuitive Understanding

Imagine asking 1000 people their height. You'd get:

```text
Heights:
160cm: 5 people (rare)
165cm: 20 people
170cm: 150 people (common)
175cm: 250 people (very common)
180cm: 250 people (very common)
185cm: 150 people (common)
190cm: 20 people
195cm: 5 people (rare)
```

This list of "how many people at each height" is a **distribution**. It shows:

- Where values cluster (around 177cm)
- Which values are common (175-180cm)
- Which are rare (160cm or 195cm)
- The overall shape (bell curve)

### Formal Definition

**Distribution:** A mathematical function describing the probability of observing different values.

It answers: "If I randomly pick one observation, what's the probability it has value x?"

### Two Ways to Represent Distributions

#### 1. Frequency Distribution (What we see)

```text
Age        Frequency    Percentage
20-25      150          16.8%
25-30      180          20.2%
30-35      160          17.9%
35-40      140          15.7%
40-45      100          11.2%
45-50      80            9.0%
50+        81            9.1%

This shows actual observed frequencies in data.
```

#### 2. Probability Distribution (Theoretical model)

```text
A smooth mathematical curve showing:
"If we sampled infinitely many passengers,
what would the distribution look like?"

Smooth curve (not histogram of actual data)
```

**Key difference:**

- **Frequency distribution:** What we actually observe in our sample
- **Probability distribution:** The theoretical pattern our sample came from

---

## 2.2 The Normal Distribution (Gaussian Distribution)

### Why Study Normal Distribution First?

The normal distribution is the most important distribution because:

1. **Ubiquitous in nature:** Heights, weights, test scores, measurement errors
2. **Central Limit Theorem:** Averages of any distribution tend toward normal
3. **Model assumptions:** Many ML algorithms assume normal distributions
4. **Mathematical convenience:** Easy to work with mathematically
5. **Foundation of statistics:** Most statistical tests assume normality

### Visual and Mathematical Description

#### The Bell Curve Shape

```text
                    ╱╲
                   ╱  ╲
                  ╱    ╲
                 ╱      ╲
                ╱        ╲
               ╱          ╲
              ╱            ╲
    ──────────              ──────────
    -3σ -2σ -1σ 0 +1σ +2σ +3σ
             μ (mean)
```

#### Key Properties

| Property                 | Value                  | Meaning                        |
| ------------------------ | ---------------------- | ------------------------------ |
| **Shape**                | Symmetric bell curve   | Tails are equal on both sides  |
| **Center**               | μ (mean)               | Peak at the middle             |
| **Mean = Median = Mode** | All three equal        | Perfectly balanced             |
| **Spread**               | σ (standard deviation) | Controls width of curve        |
| **Tails**                | Never touch zero       | Extend to ±∞ but very unlikely |

#### Mathematical Formula

```text
f(x) = (1 / (σ√(2π))) × e^(-(x-μ)²/(2σ²))

Where:
  μ = mean (center)
  σ = standard deviation (spread)
  e ≈ 2.718 (mathematical constant)
  π ≈ 3.14159 (mathematical constant)

Don't memorize this! It's just showing that the normal distribution
is completely defined by two parameters: μ and σ
```

### The 68-95-99.7 Rule (Empirical Rule)

For any normal distribution:

#### Rule Statement

```text
Within ±1σ of mean:  ~68% of data
Within ±2σ of mean:  ~95% of data
Within ±3σ of mean:  ~99.7% of data
```

#### Visual Representation

```text
                      68%
              ────────────────────
             ╱                    ╲
            ╱      95%            ╲
           ╱  ──────────────────   ╲
          ╱  ╱    99.7%         ╲  ╲
         ╱  ╱ ─────────────────── ╲  ╲

    ─────────┼─────────┼─────────┼─────
    -3σ     -2σ       -1σ    0   +1σ  +2σ  +3σ
                       μ (mean)

Within ±1σ: 68.27%
Within ±2σ: 95.45%
Within ±3σ: 99.73%
```

#### Practical Example: Titanic Age

Assume Age is normally distributed:

- Mean (μ) = 29.7 years
- Std Dev (σ) = 14.5 years

```text
±1σ: 29.7 ± 14.5 = [15.2 to 44.2] years
     ~68% of passengers between 15-44 years old

±2σ: 29.7 ± 29.0 = [0.7 to 58.7] years
     ~95% of passengers between 1-59 years old

±3σ: 29.7 ± 43.5 = [-13.8 to 73.2] years
     ~99.7% of passengers between 0-73 years old
     (negative ages impossible in reality)
```

#### Practical Application: Finding Probabilities

**Question:** What percentage of passengers were between 15 and 44 years old?

**Answer:** Within ±1σ = ~68%

**Question:** What percentage were older than 44 years old?

**Answer:** Beyond +1σ = (100% - 68%) / 2 = 16% (on right tail)

**Question:** What percentage were between 44 and 59 years old?

**Answer:** Between +1σ and +2σ = (95% - 68%) / 2 = 13.5%

### Testing for Normality

#### Visual Inspection: Histogram

```text
For normal distribution, histogram should:
✓ Be bell-shaped
✓ Be symmetric (left tail mirrors right tail)
✓ Have one peak (mode = mean)
✗ Have long skewed tail (indicates non-normal)
✗ Have multiple peaks (indicates mixture)
```

#### Visual Inspection: Q-Q Plot (Quantile-Quantile Plot)

```text
Q-Q plot compares your data against theoretical normal.

If points fall on diagonal line → Data is normal ✓
If points deviate from line → Data deviates from normal ✗

     Theoretical Normal
               │     ╱
               │   ╱ ← Points should fall here
               │ ╱
      ─────────┼─────── Your Data
             ╱ │
           ╱   │
         ╱     │
```

#### Statistical Test: Shapiro-Wilk Test

```text
Null hypothesis: Data comes from normal distribution
If p-value > 0.05: Fail to reject null → Data is normal
If p-value < 0.05: Reject null → Data is NOT normal

Python: from scipy.stats import shapiro
        shapiro(data)
```

### When Data is NOT Normal: Transformations

If your data isn't normal, you can transform it:

#### Log Transformation (for right-skewed data)

```text
Original: [1, 2, 5, 10, 100]
After log: [0, 0.3, 1.6, 2.3, 4.6]

Effect: Pulls down large values, compresses right tail
Result: More symmetric, closer to normal

When to use: Exponential data (Fare, Income, Page Views)
```

#### Square Root Transformation (mild right skew)

```text
Original: [1, 4, 9, 16, 100]
After sqrt: [1, 2, 3, 4, 10]

Effect: Less aggressive than log, moderate compression
Result: Moderate reduction in right skew
```

#### Box-Cox Transformation (automatic)

```text
Automatically finds best transformation for normality.
Uses lambda parameter to optimize.

When to use: When you're unsure which transformation to try
```

---

## 2.3 Uniform Distribution

### What It Is (Uniform)

A distribution where **all values are equally likely**.

### Visual Representation (Uniform)

```text
Probability
│
│  ════════════════
│  ║            ║
│  ║            ║
│  ║            ║
│  ║            ║
└──╫────────────╫──→ Value
   a            b

All values between a and b equally likely.
All other values have 0 probability.
```

### Mathematical Definition (Uniform)

```text
For a uniform distribution between a and b:

P(x) = 1/(b-a)  for a ≤ x ≤ b
P(x) = 0        otherwise

Example: Uniform between 0 and 1
P(x) = 1/(1-0) = 1 for all x in [0,1]
```

### Key Properties (Uniform)

| Property               | Value                     |
| ---------------------- | ------------------------- |
| **Shape**              | Perfectly flat            |
| **Mean**               | (a + b) / 2               |
| **Std Dev**            | (b - a) / √12             |
| **Symmetry**           | Perfectly symmetric       |
| **Common in reality?** | Rare (mostly theoretical) |

### Real-World Examples (Uniform)

```text
Random number generator: Uniform [0, 1]
Each decimal between 0 and 1 equally likely
```

**Poorly measured data:**

```text
If measurements are rounded to nearest integer:
Value could be anywhere in [x - 0.5, x + 0.5]
Roughly uniform within that range
```

**Titanic context:**

```text
PassengerId: 1, 2, 3, ..., 891
(Almost uniform—each ID equally represented)
Not useful for modeling!
```

### Why Uniform Distribution Matters

- **Reference point:** Most extreme case of "no pattern"
- **Null hypothesis:** If random, would be uniform
- **Model checking:** If model predicts uniform but data isn't → model is wrong
- **Sampling:** Uniform distribution underlies random sampling

---

## 2.4 Exponential Distribution

### What It Is (Exponential)

A distribution where values start high and **decay exponentially**. Most values are low, with rare very high values.

### Visual Representation (Exponential)

```text
Probability
│
│╲
│ ╲
│  ╲
│   ╲___
│       ╲____
│           ╲_____
└───────────────────→ Value

Sharp peak at low values
Long tail toward high values
Right-skewed
```

### Mathematical Definition (Exponential)

```text
f(x) = λ × e^(-λx)  for x ≥ 0

Where:
  λ (lambda) = rate parameter (controls decay speed)
  Large λ = fast decay (peak near 0)
  Small λ = slow decay (spread out)
  e ≈ 2.718
```

### Key Properties (Exponential)

| Property    | Value                       |
| ----------- | --------------------------- |
| **Shape**   | Right-skewed                |
| **Mean**    | 1/λ                         |
| **Std Dev** | 1/λ (always equal to mean!) |
| **Median**  | (ln(2))/λ ≈ 0.693/λ         |
| **Mode**    | 0 (peak at zero)            |
| **Range**   | 0 to +∞                     |

### Real-World Examples (Exponential)

```text
Most passengers: Cheap fares ($1-$20)
Some: Medium fares ($20-$50)
Few: Expensive fares ($100-$512)

This matches exponential pattern!
```

**Other examples:**

- Time between customer arrivals
- Equipment failure times
- Wait times in queues
- Income distribution in lower income brackets
- Page load times
- Size of files downloaded

### Skewness of Exponential Distribution

```text
Skewness = 2 (always positive, always right-skewed)

This is much higher skewness than normal distribution (0).
Indicates serious asymmetry.
```

### When Data is Exponential: Log Transform

If your data is exponential (highly right-skewed):

```text
Original data (Fare):
[1, 2, 3, 5, 10, 50, 100, 200, 512]

After log transform:
[0, 0.69, 1.10, 1.61, 2.30, 3.91, 4.61, 5.30, 6.24]

Result: More symmetric, closer to normal distribution
```

### Practical Implication for Modeling

```text
If feature is exponential:
  ✗ Linear regression may not work well (assumes linear relationships)
  ✓ Log-transform first, then linear regression
  ✓ Or use tree-based models (robust to distributions)

If target is exponential:
  ✗ Linear regression residuals won't be normal
  ✓ Use log-transform of target
  ✓ Or use generalized linear models (Poisson regression)
```

---

## 2.5 Binomial Distribution

### What It Is (Binomial)

Distribution of **counts of successes** in a fixed number of independent trials.

Each trial has two outcomes: Success (1) or Failure (0).

### Visual Representation (Binomial)

Different shapes based on probability of success (p):

#### p = 0.5 (50% success rate): Symmetric

```text
Probability
│
│      ╱╲
│     ╱  ╲
│    ╱    ╲
│   ╱      ╲
└──────────────→ Number of Successes

Symmetric, bell-shaped
Peak in the middle
```

#### p < 0.5 (Low success rate): Right-skewed

```text
Probability
│
│╲
│ ╲
│  ╲___
│      ╲___
└──────────────→ Number of Successes

Peak on left (more failures)
Tail extends right (fewer successes)
```

#### p > 0.5 (High success rate): Left-skewed

```text
Probability
│
│      ╱╲
│     ╱  ╲
│    ╱    ╲___
│   ╱        ╲___
└──────────────→ Number of Successes

Peak on right (more successes)
Tail extends left (fewer failures)
```

### Mathematical Definition (Binomial)

```text
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

Where:
  n = number of trials
  k = number of successes (0 to n)
  p = probability of success on each trial
  C(n,k) = combinations (ways to arrange k successes in n trials)

Don't memorize! Just understand the concept.
```

### Key Properties (Binomial)

| Property                | Value                     |
| ----------------------- | ------------------------- |
| **Number of trials**    | n (fixed in advance)      |
| **Success probability** | p (same for each trial)   |
| **Possible outcomes**   | 0, 1, 2, ..., n successes |
| **Mean**                | n × p                     |
| **Std Dev**             | √(n × p × (1-p))          |
| **Distribution shape**  | Depends on p and n        |

### Real-World Examples (Binomial)

```text
Each passenger: Survives (1) or Dies (0)
Total: n = 891 passengers
Survival rate: p = 0.38 (38% survived)

How many survivors?
Expected = 891 × 0.38 = 338.58 ≈ 339 (close to actual 342)

Distribution: Binomial(n=891, p=0.38)
Shape: Slightly left-skewed (since p < 0.5)
```

**Coin flips:**

```text
Flip coin 10 times
Success = Heads, p = 0.5

How many heads?
Distribution: Binomial(n=10, p=0.5)
Mean = 10 × 0.5 = 5 heads
```

**Election polling:**

```text
Ask 1000 voters: Will you vote for candidate X?
Success = Yes vote
p = estimated probability

How many Yes votes?
Distribution: Binomial(n=1000, p)
```

### Normal Approximation to Binomial

When n is large and p is not extreme (0.1 < p < 0.9):

```text
Binomial(n, p) ≈ Normal(μ = np, σ = √(np(1-p)))

Titanic example:
Binomial(891, 0.38) ≈ Normal(μ = 338.58, σ = √(891 × 0.38 × 0.62))
                     = Normal(μ = 338.58, σ = 14.47)

This approximation is why understanding normal distribution
helps understand binomial for large samples!
```

### When to Use Binomial vs Normal

```text
Use Binomial distribution when:
  ✓ Exactly n trials (fixed sample size)
  ✓ Counting successes/failures
  ✓ p is not extremely small or large
  ✓ Small to moderate n (< 30)

Use Normal approximation when:
  ✓ Very large n (> 30)
  ✓ Computation is easier
  ✓ p not too close to 0 or 1
```

---

## 2.6 Poisson Distribution

### What It Is (Poisson)

Distribution of **count data** describing rare events occurring in a fixed time/space interval.

Count how many times something rare happens.

### When to Use Poisson

```text
✓ Counting occurrences of rare events
✓ Events happen independently
✓ Events occur at constant average rate
✓ Over fixed time or space

Examples:
  - Number of emails received per hour
  - Number of defects in a batch
  - Number of accidents per month
  - Number of calls to customer service per day
  - Number of typos per page
```

### Visual Representation (Poisson)

#### λ = 1 (Low count): Peak at 0-1

```text
Probability
│
│╲
│ ╲
│  ╲
│   ╲____
└──────────→ Number of Events

Peak at 0 or 1
Tail extends right
Right-skewed
```

#### λ = 5 (Medium count): Broader peak

```text
Probability
│
│   ╱╲
│  ╱  ╲
│ ╱    ╲
│╱      ╲___
└──────────────→ Number of Events

Peak around 5
More spread out
Approaches normal as λ increases
```

#### λ = 10 (Higher count): Nearly normal

```text
Probability
│
│      ╱╲
│     ╱  ╲
│    ╱    ╲
│   ╱      ╲___
└──────────────────→ Number of Events

Peak around 10
Even more symmetric
Very close to normal distribution
```

### Mathematical Definition (Poisson)

```text
P(X = k) = (e^(-λ) × λ^k) / k!

Where:
  λ (lambda) = average count (rate)
  k = actual count (0, 1, 2, 3, ...)
  e ≈ 2.718
  k! = k factorial

Don't memorize! Key insight: only parameter is λ
```

### Key Properties (Poisson)

| Property            | Value                                                 |
| ------------------- | ----------------------------------------------------- |
| **Parameter**       | λ (lambda): average count                             |
| **Mean**            | λ                                                     |
| **Variance**        | λ                                                     |
| **Std Dev**         | √λ                                                    |
| **Possible values** | 0, 1, 2, 3, ... (non-negative integers)               |
| **Mode**            | floor(λ)                                              |
| **Shape**           | Right-skewed if λ small, approaches normal if λ large |

**Unique property:** Mean = Variance = λ

### Real-World Examples (Poisson)

```text
Average 50 calls per hour (λ = 50)
What's probability of exactly 45 calls in an hour?
Distribution: Poisson(λ=50)

Expected variance = 50
Std Dev = √50 ≈ 7.07 calls
So getting 45 or 55 calls is within typical variation.
```

**Quality control:**

```text
Average 3 defects per batch (λ = 3)
What's probability of 5 defects?
Distribution: Poisson(λ=3)

Mean = 3, Std Dev = √3 ≈ 1.73
5 defects is (5-3)/1.73 ≈ 1.15 std devs away (reasonable)
```

**Rare disease incidents:**

```text
Average 2 cases per week in hospital (λ = 2)
What's probability of 5 cases in a week?
```

### When to Switch from Poisson to Normal

```text
When λ is large (λ > 30):
Poisson(λ) ≈ Normal(μ=λ, σ=√λ)

Why? As λ increases, Poisson distribution becomes symmetric
(approaches normal distribution)

This is another application of central limit theorem!
```

### Practical Implication: Poisson Regression

```text
If target variable is count data (follows Poisson):
  ✗ Linear regression may not work (assumes continuous target)
  ✓ Use Poisson Regression (generalized linear model)
  ✗ Don't log-transform target (loses count interpretation)

When would you encounter this?
  - Predicting number of customer complaints
  - Predicting number of website visits
  - Predicting number of defects
```

---

## 2.7 Relationship Between Distributions

### Central Limit Theorem (Most Important Concept)

#### Statement

If you take **random samples** from ANY distribution and calculate the **sample mean** of each sample,
those sample means will be approximately **normally distributed**.

#### Why This Matters

```text
Original distribution can be:
  ✓ Exponential (very skewed)
  ✓ Uniform (flat)
  ✓ Binomial (asymmetric)
  ✓ Poisson (right-skewed)
  ✓ Literally anything

Sample means will be:
  ≈ Normal distribution (approximately)

This is profound! It explains why normal distribution is so important.
```

#### Visual Illustration

```text
Original data (exponential, right-skewed):
│╲
│ ╲___
│     ╲___
└────────────→ Values

Sample means (from repeated sampling):
│      ╱╲
│     ╱  ╲
│    ╱    ╲
│   ╱      ╲___
└──────────────→ Sample Means
            (approximately normal!)
```

#### Practical Example: Titanic Ages

```text
Individual ages: Somewhat normal, slight skew
Sample means: Very close to normal

If you took 100 random samples of 30 passengers each:

- Sample 1 mean ≈ 29.2
- Sample 2 mean ≈ 30.1
- Sample 3 mean ≈ 29.8
- ...
- Sample 100 mean ≈ 30.3

Distribution of these 100 means: Nearly perfect normal curve!
```

#### Sample Size Effect

```text
Central Limit Theorem works better with larger samples:

Sample size n = 5:  Distribution of means ≈ somewhat normal
Sample size n = 30: Distribution of means ≈ quite normal
Sample size n = 100: Distribution of means ≈ very normal
Sample size n = 1000: Distribution of means ≈ nearly perfect normal
```

#### Practical Implication

```text
This is why:
  ✓ Regression assumes normal residuals (not raw data!)
  ✓ Confidence intervals use normal distribution
  ✓ Hypothesis tests assume normal sampling distribution
  ✓ Even if your data isn't normal, averages are!
```

### Quick Reference: Distribution Relationships

```text
Binomial(n, p) with large n → Normal(μ=np, σ=√(np(1-p)))

Poisson(λ) with large λ → Normal(μ=λ, σ=√λ)

Sample means from ANY distribution → Normal (CLT)

Exponential data → Becomes more normal after log-transform
```

---

## 2.8 Choosing the Right Distribution

### Decision Tree

```text
What are you modeling?

Is it continuous (can be any real value)?
├─ YES → Normal or Exponential
│  ├─ Is it symmetric? → Normal
│  └─ Is it right-skewed? → Exponential (or transform to normal)
│
└─ NO → Is it binary (0 or 1)?
   ├─ YES → Binomial or Bernoulli
   │  └─ Many observations? → Use Normal approximation
   │
   └─ NO → Is it count data (integers 0, 1, 2, ...)?
      ├─ YES, rare events → Poisson
      ├─ YES, many trials → Binomial
      └─ NO → Check what specifically this represents
```

### Common Titanic Variables and Their Distributions

| Variable     | Distribution            | Reason                                           |
| ------------ | ----------------------- | ------------------------------------------------ |
| **Age**      | Normal (approximately)  | Continuous, symmetric, bell-shaped               |
| **Fare**     | Exponential             | Continuous, highly right-skewed, most low values |
| **Sex**      | Categorical             | Binary but qualitative, not probability          |
| **Survived** | Binomial                | Binary (0/1), fixed n=891 passengers             |
| **Pclass**   | Categorical (ordinal)   | Ordered categories, not numeric distribution     |
| **SibSp**    | Poisson (approximately) | Count data, rare events (high values)            |
| **Parch**    | Poisson (approximately) | Count data, rare events (high values)            |

---

## 2.9 Why Distribution Matters for Machine Learning

### Linear Regression Assumptions

```text
Linear Regression assumes:
  1. Linear relationship between X and y
  2. Errors are normally distributed
  3. Constant variance (homoscedasticity)
  4. Independent observations

If data is exponential:
  ✗ Linear regression may violate normality assumption
  ✓ Log-transform to make normal
  ✓ Use robust methods
```

### Logistic Regression and Binomial

```text
Logistic Regression assumes:
  - Target is binary (Bernoulli/Binomial)
  - Probability of success varies with features
  - This perfectly matches binomial distribution

Why it works:
  Logistic regression models probability p
  Binomial distribution describes outcomes from that probability
```

### Poisson Regression

```text
Use when:
  - Target is count data (Poisson-distributed)
  - Examples: predicting number of events

Regular linear regression would:
  ✗ Predict negative counts (impossible!)
  ✗ Underestimate variance for large counts
  ✓ Poisson regression handles this correctly
```

### Tree-based Models (Random Forest, XGBoost)

```text
These are "distribution-agnostic":
  ✓ Don't assume any specific distribution
  ✓ Work well with skewed data
  ✓ Don't require transformations

Why?
  Trees make splits based on values, not assumptions
  Distributions don't matter as much
```

### Feature Preprocessing Based on Distribution

```text
Normal distribution:
  → Use as-is
  → Linear regression works well

Exponential (right-skewed):
  → Log-transform
  → Then linear regression or tree models

Binomial/Count:
  → Might not need transformation
  → But log-transform sometimes helps
  → Special models (Poisson regression) available
```

---

## 2.10 Practical: Identifying Distributions in Data

### Visual Inspection Checklist

#### For Each Variable, Ask

1. **Shape question:**
   - Is histogram bell-shaped? (Normal)
   - Is it flat? (Uniform)
   - Does it have sharp peak and long right tail? (Exponential)
   - Is it discrete peaks? (Binomial/Categorical)

2. **Skewness question:**
   - Is it symmetric? (Normal)
   - Is it right-skewed? (Exponential, Poisson with small λ)
   - Is it left-skewed? (Binomial with high p)

3. **Range question:**
   - Is it continuous? (Normal, Exponential)
   - Is it 0 and 1 only? (Bernoulli/Binomial)
   - Is it non-negative integers? (Poisson, Binomial)
   - Is it categorical labels? (Categorical)

4. **Context question:**
   - What is this variable measuring?
   - What would cause these values?
   - Are there physical constraints?

### Python Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = df['variable']

# Visual inspection
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram')

plt.subplot(1, 3, 2)
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot (vs Normal)')

plt.subplot(1, 3, 3)
plt.boxplot(data)
plt.title('Boxplot')

plt.tight_layout()
plt.show()

# Statistical tests
# Shapiro-Wilk test for normality
stat, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk p-value: {p_value:.4f}")
if p_value > 0.05:
    print("Data appears normal")
else:
    print("Data does NOT appear normal")

# Skewness and Kurtosis
skewness = stats.skew(data)
kurt = stats.kurtosis(data)
print(f"Skewness: {skewness:.3f}")
print(f"Kurtosis: {kurt:.3f}")
```

---

## 2.11 Summary and Key Takeaways

### The Five Distributions You Need to Know

| Distribution    | Shape                 | Use Case                  | Key Parameter |
| --------------- | --------------------- | ------------------------- | ------------- |
| **Normal**      | Bell curve, symmetric | Continuous data, defaults | μ, σ          |
| **Exponential** | Right-skewed decay    | Rare events, wait times   | λ             |
| **Uniform**     | Flat line             | Random generation         | a, b          |
| **Binomial**    | Depends on p          | Counting successes        | n, p          |
| **Poisson**     | Right-skewed discrete | Rare counts               | λ             |

### Decision Rules

1. **For normal distribution:**
   - Use as-is for most analyses
   - Linear regression applies
   - Can use 68-95-99.7 rule for predictions

2. **For right-skewed (Exponential, etc.):**
   - Apply log-transformation
   - Or use tree-based models
   - Consider Poisson/Exponential regression

3. **For discrete/binary data:**
   - Use logistic regression (binary)
   - Use Poisson regression (counts)
   - Use binomial tests

4. **Central Limit Theorem applies to:**
   - Sample means (always)
   - Confidence intervals (derived from this)
   - Hypothesis tests (rely on this)

### Why Distributions Matter for Data Science

```text
Understanding distributions helps you:

1. DIAGNOSE problems
   "Why doesn't linear regression work?"
   → Check if residuals are normal

2. CHOOSE transformations
   "How should I preprocess this feature?"
   → Identify distribution, apply appropriate transform

3. SELECT models
   "Which model should I use?"
   → Match model to data distribution

4. MAKE predictions
   "How confident in this prediction?"
   → Use distribution properties (68-95-99.7 rule)

5. VALIDATE assumptions
   "Do my model assumptions hold?"
   → Test if residuals follow assumed distribution
```

---

## 2.12 Practice Exercises

### Exercise 1: Distribution Identification

**Dataset:** Download Titanic, examine these variables:

1. Age
2. Fare
3. Survived
4. Sex

For each, answer:

- What distribution might it follow?
- What visual properties support your answer?
- What transformation (if any) would help normality?

<details>
<summary>Click for guidance (not full answers)</summary>

**Age:**

- Plot histogram (should be somewhat bell-shaped)
- Slight left skew (infants vs adults)
- → Approximately normal
- Transformation: None needed

**Fare:**

- Plot histogram (should be right-skewed)
- Many cheap, few expensive
- → Exponential-like
- Transformation: Try log-transform

**Survived:**

- Only values: 0 or 1
- 38% survived, 62% didn't
- → Binomial(n=891, p=0.38)
- Transformation: N/A (categorical target)

**Sex:**

- Categorical, not numeric
- → Not a numeric distribution
- But we can note frequencies

</details>

### Exercise 2: Normal Distribution Applications

Assume Age in Titanic is Normal: μ=29.7, σ=14.5

1. What percentage of passengers were between 15 and 44 years old?
2. What percentage were older than 58.7 years?
3. If someone was 73+ years old, how many standard deviations from mean?

<details>
<summary>Click for answers</summary>

**1. Between 15 and 44 years old:**

- 15 = 29.7 - 14.5 = μ - 1σ
- 44 = 29.7 + 14.5 = μ + 1σ
- Within ±1σ = ~68%

**2. Older than 58.7 years:**

- 58.7 = 29.7 + 2(14.5) = μ + 2σ
- Beyond +2σ = (100% - 95%)/2 = 2.5%

**3. Someone 73 years old:**

- Deviation = 73 - 29.7 = 43.3 years
- Number of σ = 43.3 / 14.5 = 2.99 ≈ 3σ
- This is at the extreme tail (~0.15%)

</details>

### Exercise 3: Transform Choice

You have three features, all continuous:

1. Age: Mean=30, Std=14, Skewness=0.3 (slightly right)
2. Income: Mean=$50k, Std=$75k, Skewness=2.5 (highly right)
3. Test Score: Mean=75, Std=12, Skewness=-0.4 (slightly left)

For each, decide: Keep as-is, log-transform, or sqrt-transform?

<details>
<summary>Click for guidance</summary>

**Age:**

- Skewness 0.3 is minimal
- → Keep as-is
- Already suitable for linear regression

**Income:**

- Skewness 2.5 is high
- → Log-transform
- Log-transform works well for financial data

**Test Score:**

- Skewness -0.4 is small
- → Keep as-is or mild transformation
- Slight left skew is acceptable for most methods

</details>

---
