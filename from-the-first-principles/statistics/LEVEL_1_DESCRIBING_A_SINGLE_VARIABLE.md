# Level 1: Describing a Single Variable

## Overview

Level 0 taught you *what* data is (observations, variables, variability). Level 1 teaches you *how to describe* that data numerically and visually.

When you have a single variable, you need to answer four fundamental questions:

1. **Where is the center?** (Mean, Median, Mode)
2. **How spread out is it?** (Variance, Standard Deviation, Range, IQR)
3. **What's the shape?** (Skewness, Kurtosis)
4. **Are there extreme values?** (Outliers)

These four aspects completely describe any single variable. Together, they form the foundation of Exploratory Data Analysis (EDA).

---

## 1.1 Center (Location): Mean, Median, Mode

### Why Measure Center?

The center represents a **single number that represents the typical observation**.

**Analogy:** If you wanted to describe the typical passenger's age on the Titanic in one number, what would you say?

Two different datasets can have identical centers but very different shapes—this is why we need spread and shape measures too. But first, let's understand the three ways to measure center.

### 1. Mean (Arithmetic Average)

#### Definition

The sum of all values divided by the count.

#### Formula

```text
μ = (x₁ + x₂ + x₃ + ... + xₙ) / n
  = Σx / n

Where:
  μ = population mean (the true average)
  Σx = sum of all values
  n = number of observations
  x̄ = sample mean (average of sample)
```

#### Worked Example: Titanic Ages

```text
Passengers: 22, 38, 26, 35, 35 years old

Step 1: Sum all ages
22 + 38 + 26 + 35 + 35 = 156

Step 2: Divide by count
156 / 5 = 31.2 years

Mean age = 31.2 years
```

#### Interpretation

"On average, a Titanic passenger was 31.2 years old."

#### Characteristics

| Aspect                      | Detail                                            |
| --------------------------- | ------------------------------------------------- |
| **Computational**           | Uses every data point                             |
| **Sensitivity to outliers** | Very sensitive (one extreme value can shift mean) |
| **Best for**                | Numerical data without extreme outliers           |
| **Skewed data**             | Mean gets pulled toward the tail                  |
| **Units**                   | Same as original data                             |

#### Example: Outlier Effect

```text
Dataset A: [20, 25, 30]
Mean = 75 / 3 = 25 ✓ Reasonable

Dataset B: [20, 25, 30, 1000]  (one extreme value)
Mean = 1075 / 4 = 268.75 ✗ Unreasonable! No one was 268.75 years old

The outlier pulls the mean to 268.75, which doesn't represent the typical passenger.
```

### 2. Median

#### Definition

The middle value when all observations are sorted in order.

#### Formula

```text
For n observations sorted: x₁ ≤ x₂ ≤ ... ≤ xₙ

If n is odd:     Median = x₍ₙ₊₁₎/₂ (the middle value)
If n is even:    Median = (xₙ/₂ + xₙ/₂₊₁) / 2 (average of two middle values)
```

#### Worked Example 1: Odd Number of Values

```text
Passengers: 22, 38, 26, 35, 35 years old

Step 1: Sort
22, 26, 35, 35, 38

Step 2: Find middle position
n = 5, so position = (5+1)/2 = 3

Step 3: Get value at position 3
Median = 35 years
```

#### Worked Example 2: Even Number of Values

```text
Passengers: 20, 25, 30, 40 years old

Step 1: Already sorted
20, 25, 30, 40

Step 2: Find two middle positions
n = 4, so positions = 2 and 3

Step 3: Average them
Median = (25 + 30) / 2 = 27.5 years
```

#### Interpretation

"Half the passengers were younger than 35 years, half were older."

#### Characteristics

| Aspect                      | Detail                                        |
| --------------------------- | --------------------------------------------- |
| **Computational**           | Sorting required, doesn't use all values      |
| **Sensitivity to outliers** | Robust (outliers don't affect it much)        |
| **Best for**                | Skewed distributions or when outliers present |
| **Skewed data**             | Stays near the bulk of data                   |
| **Units**                   | Same as original data                         |

#### Example: Robustness to Outliers

```text
Dataset A: [20, 25, 30]
Median = 25 ✓

Dataset B: [20, 25, 30, 1000]  (one extreme value)
Sorted: [20, 25, 30, 1000]
Median = (25 + 30) / 2 = 27.5 ✓ Still reasonable

The outlier doesn't pull the median. Compare to mean = 268.75!
```

### 3. Mode

#### Definition

The value that appears most frequently.

#### Characteristics

| Aspect                      | Detail                                      |
| --------------------------- | ------------------------------------------- |
| **Computational**           | Count frequency of each value               |
| **Sensitivity to outliers** | Not affected (doesn't use value magnitudes) |
| **Best for**                | Categorical data, bimodal distributions     |
| **Data requirement**        | Values must repeat                          |
| **Uniqueness**              | Can be multiple modes, or no mode           |

#### Worked Example 1: Numerical Data

```text
Ages: [20, 25, 25, 25, 30, 35, 40]

Frequency count:
20 appears 1 time
25 appears 3 times ← most frequent
30 appears 1 time
35 appears 1 time
40 appears 1 time

Mode = 25 years
```

Interpretation: "The most common age was 25."

#### Worked Example 2: Categorical Data

```text
Sex: [Male, Female, Male, Male, Female, Male, Male]

Frequency count:
Male appears 5 times ← most frequent
Female appears 2 times

Mode = Male
```

Interpretation: "Most Titanic passengers were Male."

#### Worked Example 3: No Mode

```text
Ages: [20, 25, 30, 35, 40]

Each value appears once.
No value is most frequent.

Mode = None (or "no mode exists")
```

#### Worked Example 4: Bimodal Distribution

```text
Heights: [160, 160, 170, 170, 180]

Frequency count:
160 appears 2 times ← tied for most
170 appears 2 times ← tied for most
180 appears 1 time

Mode = Both 160 and 170 (bimodal)
```

Interpretation: "Heights cluster around two values: 160 and 170."

### Choosing the Right Measure of Center

#### Decision Tree

```text
Is data normally distributed (symmetric bell curve)?
├─ YES → Use Mean
│  └─ Mean = Median = Mode
│  └─ "Passenger age averaged 30 years"
│
└─ NO → Is there skew or outliers?
   ├─ YES → Use Median
   │  └─ "Typical passenger was 28 years old"
   │
   └─ Is data categorical?
      └─ YES → Use Mode
         └─ "Most passengers were Male"
```

#### Quick Comparison

| Measure    | Best For                      | Example Use                           |
| ---------- | ----------------------------- | ------------------------------------- |
| **Mean**   | Normal data, no outliers      | "Average age was 30 years"            |
| **Median** | Skewed data, outliers present | "Typical fare was $14" (not $32 mean) |
| **Mode**   | Categorical, most common      | "Most passengers were in 3rd class"   |

#### Real Titanic Example

```text
Age distribution: Slightly left-skewed (some infants, mostly adults)
  Mean = 29.7 years
  Median = 28 years
  These are close → Mean is reasonable to use

Fare distribution: Right-skewed (most cheap, some very expensive)
  Mean = $32.20
  Median = $14.45
  Very different! → Median better represents typical fare
  
Sex distribution: Categorical (Male/Female)
  Male: 577 passengers
  Female: 314 passengers
  Mode = Male
```

---

## 1.2 Spread (Variability): Range, Variance, Standard Deviation, IQR

### Why Measure Spread?

The center alone is incomplete. Two datasets can have identical means but completely different spreads.

**Critical Example:**

```text
Dataset A:  [29, 30, 31]      Mean = 30, Very tight
Dataset B:  [1, 30, 59]       Mean = 30, Very loose

Same mean, but Dataset A is predictable while Dataset B is chaotic.
Which would you trust for forecasting?
```

Spread measures how much variability exists around the center.

### 1. Range

#### Definition

The distance from minimum to maximum value.

#### Formula

```text
Range = Maximum - Minimum
```

#### Worked Example

```text
Titanic Ages: 0.17 to 80 years

Range = 80 - 0.17 = 79.83 years
```

#### Characteristics

| Aspect                      | Detail                                |
| --------------------------- | ------------------------------------- |
| **Sensitivity to outliers** | Extremely sensitive                   |
| **Computation**             | Simplest (just max - min)             |
| **Usefulness**              | Limited (only two data points matter) |
| **Better use**              | Quick scan, not for analysis          |

#### Problem

```text
Dataset A: [10, 20, 30, 40, 50]     Range = 40
Dataset B: [10, 25, 28, 29, 50]     Range = 40

Same range, but Dataset A is more spread out!
Range only looks at endpoints, ignoring distribution.
```

**Conclusion:** Range is too crude. We need better measures.

### 2. Variance (σ²)

#### The Problem Variance Solves

We need to measure how far observations are from the mean on average.

**First attempt:** Average deviation

```text
Deviations from mean = [x₁ - μ, x₂ - μ, ..., xₙ - μ]
Average = Σ(xᵢ - μ) / n

Problem: This sum always equals zero (mathematical property of mean)!
```

**Solution:** Square the deviations first (squaring makes everything positive).

#### Definition

The average of squared deviations from the mean.

#### Formula

```text
Population Variance:  σ² = Σ(x - μ)² / n

Sample Variance:      s² = Σ(x - x̄)² / (n - 1)

Why (n-1) for sample? Unbiased estimator (Bessel's correction)
                      We lose one degree of freedom estimating mean
```

#### Detailed Worked Example

```text
Passengers: 20, 25, 30, 35, 40 years

Step 1: Calculate mean
μ = (20 + 25 + 30 + 35 + 40) / 5 = 30

Step 2: Calculate deviations
20 - 30 = -10
25 - 30 = -5
30 - 30 = 0
35 - 30 = +5
40 - 30 = +10

Step 3: Square the deviations
(-10)² = 100
(-5)² = 25
(0)² = 0
(+5)² = 25
(+10)² = 100

Sum of squared deviations = 100 + 25 + 0 + 25 + 100 = 250

Step 4: Divide by n (or n-1 for sample)
Variance = 250 / 5 = 50 years²

Population variance σ² = 50 years²
Sample variance s² = 250 / 4 = 62.5 years²
```

#### Interpretation

**Units:** Squared units (years²). This is mathematically convenient but hard to interpret.

**Value of 50:** On average, observations are 50 squared-years away from the mean (confusing!).

**Relative interpretation:**

- σ² = 10: Very low variability
- σ² = 50: Moderate variability
- σ² = 100: High variability

But the actual units don't tell you much.

#### Characteristics

| Aspect                      | Detail                                  |
| --------------------------- | --------------------------------------- |
| **Units**                   | Squared (years²)                        |
| **Interpretability**        | Poor (hard to understand squared units) |
| **Mathematical use**        | Excellent (nice algebraic properties)   |
| **Relationship to std dev** | σ² = σ × σ                              |

#### When Variance is Used

Variance appears in:

- Statistical formulas and proofs
- Internal calculations
- Combining multiple sources of variation
- ANOVA (Analysis of Variance)

**But you almost never report variance to stakeholders.** Instead, report standard deviation.

### 3. Standard Deviation (σ)

#### Definition

The square root of variance.

#### Formula

```text
Population Std Dev:  σ = √[Σ(x - μ)² / n]

Sample Std Dev:      s = √[Σ(x - x̄)² / (n - 1)]
```

#### Detailed Worked Example

```text
Using previous example:
Variance = 50 years²

Standard Deviation = √50 = 7.07 years
```

#### Interpretation

**Units:** Back to original units (years). Much more interpretable.

**Value of 7.07:** On average, passengers deviate from the mean (30) by about 7.07 years.

**Practical reading:**

- Low σ: Values cluster tightly (predictable)
- High σ: Values spread widely (unpredictable)

#### Real Titanic Example

```text
Age: Mean = 29.7, Std Dev = 14.5 years
Interpretation: "Typical passenger deviates from 29.7 by about 14.5 years"
               "Most passengers are between 15-44 years old"

Fare: Mean = $32.2, Std Dev = $49.7
Interpretation: "Typical passenger's fare deviates from $32 by $49.70"
               "Fares are highly variable"
```

#### The 68-95-99.7 Rule (Normal Distribution)

For normally distributed data:

| Range  | Percentage | Interpretation         |
| ------ | ---------- | ---------------------- |
| μ ± 1σ | ~68%       | Most data falls here   |
| μ ± 2σ | ~95%       | Nearly all data here   |
| μ ± 3σ | ~99.7%     | Almost everything here |

**Example with Age:**

```text
Mean = 29.7, Std Dev = 14.5

±1σ: 29.7 ± 14.5 = [15.2, 44.2]    ~68% of passengers in this range
±2σ: 29.7 ± 29.0 = [0.7, 58.7]     ~95% of passengers in this range
±3σ: 29.7 ± 43.5 = [-13.8, 73.2]   ~99.7% in this range (invalid negative ages)
```

### 4. Variance vs Standard Deviation: Deep Dive

#### The Relationship

They measure the same thing (spread) but in different units:

```text
Spread exists in data
    ↓
Variance quantifies spread (squared units)
    ↓
Standard Deviation = √Variance (original units)
    ↓
Interpretation becomes easy
```

| Aspect               | Variance (σ²)       | Standard Deviation (σ)      |
| -------------------- | ------------------- | --------------------------- |
| **Formula**          | Σ(x - μ)² / n       | √[Σ(x - μ)² / n]            |
| **Units**            | Squared (years²)    | Original (years)            |
| **Interpretation**   | Hard ("50 years²?") | Easy ("7 years on average") |
| **Mathematical use** | Excellent           | Good                        |
| **Reporting**        | Not typically       | Always                      |
| **Relationship**     | σ²                  | σ = √σ²                     |

#### Which to Use?

| Situation                         | Use                           |
| --------------------------------- | ----------------------------- |
| **Publishing results**            | Standard Deviation            |
| **Mathematical proofs**           | Variance                      |
| **Feature engineering**           | Standard Deviation            |
| **Model assumptions**             | Variance                      |
| **Building confidence intervals** | Standard Deviation            |
| **Python numpy**                  | Both available (var(), std()) |

**Rule of thumb:** Report standard deviation unless specifically doing mathematical statistics.

#### Why Both Exist Mathematically

**Variance is mathematically convenient:**

```text
If you have two independent sources of variation:
Variation A contributes variance σ²_A
Variation B contributes variance σ²_B

Total variance = σ²_A + σ²_B  ← Clean addition!

If you tried with std dev:
Total std dev ≠ σ_A + σ_B  ← Doesn't work simply
```

**Standard deviation is practically convenient:**

```text
Standard deviation is in original units
You can interpret it directly
68-95-99.7 rule works
Confidence intervals use it
```

### 5. Interquartile Range (IQR)

#### Definition

The range of the middle 50% of data. Robust to outliers.

#### Quantile Terminology

| Term                           | Definition             | Position       |
| ------------------------------ | ---------------------- | -------------- |
| **Minimum (0th percentile)**   | Smallest value         | Bottom         |
| **Q1 (25th percentile)**       | 25% of data below this | Bottom quarter |
| **Median (50th percentile)**   | Middle value           | Middle         |
| **Q3 (75th percentile)**       | 75% of data below this | Top quarter    |
| **Maximum (100th percentile)** | Largest value          | Top            |

#### Visual Representation

```text
Min     Q1      Median   Q3      Max
|-------|--------|--------|--------|
 25%   middle 50%   25%
        (IQR = Q3 - Q1)
```

#### Formula

```text
IQR = Q3 - Q1

Lower outlier bound = Q1 - 1.5 × IQR
Upper outlier bound = Q3 + 1.5 × IQR
```

#### Worked Example: Titanic Fare

```text
Observations: 891 passengers

Step 1: Sort all fares and find quartiles
Q1 (25th percentile) = $7.91
Q3 (75th percentile) = $31.00

Step 2: Calculate IQR
IQR = 31.00 - 7.91 = $23.09

Step 3: Identify outliers
Lower bound = 7.91 - 1.5 × 23.09 = 7.91 - 34.64 = -26.73 (not possible, no lower outliers)
Upper bound = 31.00 + 1.5 × 23.09 = 31.00 + 34.64 = $65.64

Step 4: Count outliers
Any fare > 65.64 is an outlier
Maximum fare observed = $512.29
This is an extreme outlier (8× the upper bound)

Number of outliers in Titanic: ~134 passengers paid fares > $65.64
```

#### Interpretation

"The middle 50% of passengers paid between $7.91 and $31.00. Anything beyond ~$65 is unusually expensive."

#### Characteristics

| Aspect               | Detail                                      |
| -------------------- | ------------------------------------------- |
| **Units**            | Same as data                                |
| **Robustness**       | Excellent (outliers don't affect quartiles) |
| **Interpretability** | Good (directly about data percentiles)      |
| **Best for**         | Identifying outliers, skewed data           |

#### IQR vs Standard Deviation

For normal distributions:

```text
IQR ≈ 1.35 × σ
σ ≈ IQR / 1.35
```

**When to use which:**

| Measure     | When                             |
| ----------- | -------------------------------- |
| **Std Dev** | Normal distribution, no outliers |
| **IQR**     | Skewed data, outliers present    |

### 6. Coefficient of Variation (CV)

#### Definition

Standard deviation expressed as a percentage of the mean.

#### Formula

```text
CV = (σ / μ) × 100%
```

#### Why Use It?

Comparing spread across variables with different units or scales.

#### Worked Example

```text
Variable A: Mean = 100, Std Dev = 20
CV = (20/100) × 100% = 20%

Variable B: Mean = 50, Std Dev = 8
CV = (8/50) × 100% = 16%

Conclusion: Variable A is more variable (20% > 16%)
Even though std dev of A is higher than B, we need to account for different scales.
```

---

## 1.3 Shape: Skewness and Kurtosis

### 1. Skewness

#### Definition

Measure of asymmetry in the distribution.

#### The Three Shapes

##### Skewness = 0: Symmetric (Normal Distribution)

```text
        ╱╲
       ╱  ╲
      ╱    ╲
     ╱      ╲
    ╱        ╲

Distribution is balanced
Mean ≈ Median
Tails are equal on both sides
Example: Height, IQ scores, measurement errors
```

**Interpretation:** Data is evenly distributed around the center.

##### Skewness > 0: Right-Skewed (Positive Skew)

```text
    ╱╲
   ╱  ╲___
  ╱       ╲____
 ╱             ╲_____

Tail stretches right (toward high values)
Mean > Median
Most data clustered on left
Example: Income, wealth, Titanic fares
```

**Interpretation:** A few extreme high values pull the distribution right. Mean is pulled toward tail.

**Visual cue:** Long tail pointing right.

##### Skewness < 0: Left-Skewed (Negative Skew)

```text
____ ╱╲
    ╱  ╲
   ╱    ╲
  ╱      ╲___
 ╱____       ╲

Tail stretches left (toward low values)
Mean < Median
Most data clustered on right
Example: Exam scores (many high scores, few failures), age at death, Titanic survival
```

**Interpretation:** A few extreme low values pull the distribution left. Mean is pulled toward tail.

**Visual cue:** Long tail pointing left.

#### Formula

```text
Skewness = Σ(x - μ)³ / (n × σ³)

Why cube? Preserves sign (positive/negative) while emphasizing extremes
```

#### Interpretation Guidelines

| Skewness Value | Distribution      | Data Pattern      |
| -------------- | ----------------- | ----------------- |
| -1 to +1       | Fairly symmetric  | Roughly normal    |
| -2 to +2       | Moderately skewed | Some asymmetry    |
| < -2 or > +2   | Highly skewed     | Extreme asymmetry |

#### Real Titanic Examples

```text
Age: Skewness ≈ 0.5
  Slightly right-skewed (few infants, mostly older)
  Mean (29.7) > Median (28)

Fare: Skewness ≈ 2.5
  Highly right-skewed (most cheap fares, few expensive)
  Mean ($32.2) >> Median ($14.45)

Survived: Skewness ≈ 0.4
  Slightly right-skewed (fewer survivors than non-survivors)
```

#### Why Skewness Matters

1. **Center choice:** Use median for skewed data
2. **Model assumptions:** Linear regression assumes somewhat normal
3. **Feature engineering:** Log-transform can reduce skewness
4. **Outlier identification:** Skew suggests where outliers are

### 2. Kurtosis

#### Definition

Measure of "tailedness" — how extreme the tails are relative to normal distribution.

#### The Three Tail Types

##### Excess Kurtosis = 0: Mesokurtic (Normal Distribution)

```text
        ╱╲
       ╱  ╲
      ╱    ╲
     ╱      ╲

Tails like normal distribution
Moderate number of extreme values
Example: Normal distribution itself
```

##### Excess Kurtosis > 0: Leptokurtic (Fat Tails)

```text
        ┃  ┃
        ┃  ┃
       ╱    ╲
      ╱      ╲

Tails fatter than normal
More extreme values than expected
Higher peak in center
Example: Stock returns, error distributions in finance
```

**Interpretation:** More outliers than a normal distribution. Data has bigger swings.

##### Excess Kurtosis < 0: Platykurtic (Thin Tails)

```text
        
       ╱────────╲
      ╱          ╲
     ╱            ╲

Tails thinner than normal
Fewer extreme values than expected
Flatter peak in center
Example: Uniform distribution (all values equally likely)
```

**Interpretation:** Fewer outliers than normal. Data is more consistent.

#### Why Kurtosis Matters

1. **Risk assessment:** High kurtosis means risk of extreme events
2. **Model assumptions:** Regression assumes moderate kurtosis
3. **Robustness:** High kurtosis data needs robust methods
4. **Practical significance:** "Black swan" events are leptokurtic

---

## 1.4 Outliers: Detection and Treatment

### What is an Outlier?

An observation that is far from the rest of the data distribution.

```text
Typical values: [20, 22, 25, 28, 30]
Outlier: 500 (extremely far)
```

### IQR Method (Standard Approach)

#### The Method

```text
1. Calculate Q1 (25th percentile)
2. Calculate Q3 (75th percentile)
3. Calculate IQR = Q3 - Q1
4. Calculate bounds:
   Lower bound = Q1 - 1.5 × IQR
   Upper bound = Q3 + 1.5 × IQR
5. Any value outside bounds = Outlier
```

#### Why 1.5?

For normal distributions:

- ±1.5 × IQR captures ~99% of data
- Values outside are statistically unusual

It's an empirical rule that works well in practice.

#### Worked Example: Titanic Age

```text
Q1 = 20.125 years
Q3 = 38 years
IQR = 38 - 20.125 = 17.875

Lower bound = 20.125 - 1.5 × 17.875 = 20.125 - 26.8125 = -6.6875
Upper bound = 38 + 1.5 × 17.875 = 38 + 26.8125 = 64.8125

Outliers: Anyone older than 64.8 years
Oldest passenger: 80 years ← Outlier
Babies: ~0.17 years ← Below lower bound (but practical limit exists)

Number of age outliers: Very few (age naturally capped at ~80)
```

#### Worked Example: Titanic Fare

```text
Q1 = $7.91
Q3 = $31.00
IQR = $23.09

Lower bound = 7.91 - 1.5 × 23.09 = -26.73 (not possible)
Upper bound = 31.00 + 1.5 × 23.09 = $65.64

Outliers: Fares > $65.64
Maximum fare: $512.29 ← Extreme outlier
Next highest: ~$262 ← Outlier

Number of fare outliers: ~134 passengers (~15% of data!)
```

### Other Outlier Detection Methods

#### Z-Score Method

```text
Z-score = (x - μ) / σ

Typical threshold: |Z| > 3 (extreme outlier)
              or: |Z| > 2.5 (moderate outlier)

Example (Age):
If someone is 80 years old:
Z = (80 - 29.7) / 14.5 = 3.47
This is a 3.47 standard deviation event → Outlier
```

#### Modified Z-Score (Robust)

Uses median and median absolute deviation (MAD) instead of mean/std.
Better for very skewed data.

### What to Do With Outliers?

#### Step 1: Investigate

**Is it an error?**

- Data entry mistake (age = 999)
- Sensor malfunction
- Unit conversion error

**Is it legitimate?**

- Real observation that's unusual but valid
- Example: Titanic highest fare = $512.29 (legitimate, wealthy passenger)

#### Step 2: Decide

| Situation                     | Action                                   |
| ----------------------------- | ---------------------------------------- |
| **Confirmed error**           | Remove                                   |
| **Legitimate but extreme**    | Keep (might be important)                |
| **Ambiguous**                 | Keep (don't assume error)                |
| **Building robust model**     | Keep (tests robustness)                  |
| **Building predictive model** | Context-dependent (consult stakeholders) |

#### Step 3: Document

Always document what you did and why:

```text
"Removed 3 entries with Age > 150 years (data entry errors).
Kept 14 passengers with Fare > $200 (legitimate wealthy passengers).
```

---

## 1.5 Summary: Complete Description of a Single Variable

### The Complete Description

To fully describe any single variable, answer these four questions:

#### 1. Where is the Center?

- **Mean:** Typical value (sensitive to outliers)
- **Median:** Middle value (robust to outliers)
- **Mode:** Most common value (for categories)

#### 2. How Spread Out?

- **Range:** Max - Min (crude, only two points)
- **Variance σ²:** Average squared deviation (mathematically useful, hard to interpret)
- **Std Dev σ:** Square root of variance (interpretable, same units as data)
- **IQR:** Range of middle 50% (robust, good for outlier detection)
- **CV:** Std dev as % of mean (for comparing different scales)

#### 3. What's the Shape?

- **Skewness:** Asymmetry (-1 to +1 = somewhat symmetric)
- **Kurtosis:** Tail heaviness (> 0 = fat tails, more outliers)

#### 4. Are There Outliers?

- **IQR method:** Values outside Q1 ± 1.5×IQR
- **Z-score:** |Z| > 3 or 2.5

### Real Titanic Variable Descriptions

#### Age

```text
Center:
  Mean = 29.7 years
  Median = 28 years
  Mode = 24 years (most common)

Spread:
  Range = 0.17 to 80 years (79.83 span)
  Std Dev = 14.5 years
  Variance = 210.25 years²
  IQR = 20.125 to 38 years
  CV = 48.8%

Shape:
  Skewness ≈ 0.5 (slightly right-skewed)
  Reason: Some infants pull left, but mostly adults

Outliers:
  Upper bound = 64.8 years
  Passengers > 64.8: Outliers
  Example: 80-year-old

Interpretation:
"Passengers averaged 29.7 years old, with typical deviation of ±14.5 years.
Half were between 20 and 38. Distribution slightly left-skewed with few elderly passengers.
Three passengers over 64 are statistical outliers."
```

#### Fare

```text
Center:
  Mean = $32.20
  Median = $14.45
  Large difference indicates right skew

Spread:
  Range = $0 to $512.29
  Std Dev = $49.70
  Variance = $2470
  IQR = $7.91 to $31.00
  CV = 154.3% (very high variability)

Shape:
  Skewness ≈ 2.5 (heavily right-skewed)
  Reason: Most passengers cheap, few wealthy pay exponentially more

Outliers:
  Upper bound = $65.64
  Passengers > $65.64: ~134 outliers (15% of data!)
  Most expensive: $512.29

Interpretation:
"Median fare ($14.45) is more representative than mean ($32.20).
Most passengers concentrated at bottom, with wealthy passengers pulling distribution right.
High variability (CV=154%) shows extreme inequality in ticket prices.
Multiple wealthy passengers paid 8× the typical fare."
```

#### Sex (Categorical)

```text
Distribution:
  Male: 577 passengers (65%)
  Female: 314 passengers (35%)

Mode = Male

Interpretation:
"About 2/3 of Titanic passengers were male.
Known to be 'women and children first' evacuation priority,
affecting downstream survival distribution."
```

---

## 1.6 Practical Workflow: EDA for a Single Variable

### Template: Describing Any Variable

```python
# For numerical variable:
1. Calculate mean, median, mode
2. Calculate std dev, variance, IQR, range
3. Calculate skewness and kurtosis
4. Identify outliers (IQR method)
5. Visualize (histogram, boxplot, density)
6. Interpret findings

# For categorical variable:
1. Count frequency of each category
2. Calculate mode
3. Calculate percentages
4. Visualize (bar chart)
5. Interpret findings
```

### Python Implementation Pseudocode

```python
import numpy as np
from scipy import stats

# Numerical Variable
data = df['Age']

# Center
mean = data.mean()
median = data.median()
mode = data.mode()[0]

# Spread
std_dev = data.std()  # Uses n-1 (sample)
variance = data.var()
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1
range_ = data.max() - data.min()

# Shape
skewness = data.skew()
kurtosis = data.kurtosis()

# Outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data < lower_bound) | (data > upper_bound)]

# Summary
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Std Dev: {std_dev:.2f}")
print(f"Skewness: {skewness:.3f}")
print(f"Outliers: {len(outliers)}")
```

---

## 1.7 Connecting to Feature Engineering

### Why Level 1 Matters for ML

#### 1. Feature Understanding

Before using a feature in any model:

```text
"Does this feature have meaning?"
"Are there values that look like errors?"
"Is it skewed (might need transformation)?"
```

#### 2. Data Quality Assessment

```text
"How much missing data?"
"How many outliers?"
"Are units consistent?"
"Do values make sense?"
```

#### 3. Feature Selection Preparation

```text
"Which variables might predict the target?"
(Statistical testing comes in Level 3)
```

#### 4. Preprocessing Decisions

```text
"Does this feature need scaling?" (High std dev)
"Should I transform it?" (High skewness)
"Should I bin it?" (Categorical conversion)
"Remove outliers?" (Decision based on analysis)
```

### Example: Titanic Fare Feature

```text
Raw analysis (Level 1):
  - Highly right-skewed (skewness = 2.5)
  - High variability (std dev = $49.7)
  - Many outliers (15% above $65.64)
  - Mean ≠ Median (difference indicates skew)

Implications for preprocessing:
  ✓ Consider log transformation (reduce skewness)
  ✓ Consider standardization (high spread)
  ✓ Keep outliers (legitimate high-payers)
  ✓ Handle missing values (some tickets free)
```

---

## 1.8 Key Takeaways

### The Four Essential Questions

| Question                | Measures               | Why               |
| ----------------------- | ---------------------- | ----------------- |
| **Where's the center?** | Mean, Median, Mode     | Typical value     |
| **How spread out?**     | Std Dev, IQR, Variance | Variability       |
| **What's the shape?**   | Skewness, Kurtosis     | Distribution form |
| **Any extremes?**       | Outlier detection      | Data quality      |

### Choosing the Right Statistics

| Data Type         | Center   | Spread    | Shape Detection |
| ----------------- | -------- | --------- | --------------- |
| **Normal**        | Mean ✓   | Std Dev ✓ | Skewness ≈ 0    |
| **Skewed**        | Median ✓ | IQR ✓     | Skewness ≠ 0    |
| **With outliers** | Median ✓ | IQR ✓     | Heavy tails     |
| **Categorical**   | Mode ✓   | N/A       | Frequency dist  |

### Variance vs Standard Deviation: Final Summary

| Aspect             | Variance (σ²)                  | Standard Deviation (σ) |
| ------------------ | ------------------------------ | ---------------------- |
| **Measures**       | Same thing: spread around mean |                        |
| **Units**          | Squared (years²)               | Original (years)       |
| **Interpretation** | Hard                           | Easy                   |
| **Use**            | Math, proofs                   | Reporting, predictions |
| **Relationship**   | σ = √σ²                        |                        |
| **When reporting** | Never                          | Always                 |

**Key insight:** They're two ways of expressing the same concept. Variance is the mathematical form, standard deviation is the human-readable form.

---

## 1.9 Practice Exercises

### Exercise 1: Complete Description

**Dataset:** Titanic survival times (minutes survived): [5, 12, 8, 45, 120, 3, 8, 15, 20]

1. Calculate mean, median, mode
2. Calculate std dev (use n-1)
3. Calculate IQR and identify outliers
4. Describe the distribution in English

<details>
<summary>Click for answers</summary>

**Answers:**

1. Mean = (5+12+8+45+120+3+8+15+20)/9 = 236/9 = 26.2 min
   Sorted: [3, 5, 8, 8, 12, 15, 20, 45, 120]
   Median = 12 (position 5)
   Mode = 8 (appears twice)

2. Variance = [(3-26.2)² + (5-26.2)² + ... + (120-26.2)²] / 8 = 2397.1/8 = 299.6
   Std Dev = √299.6 = 17.3 min

3. Q1 = 8, Q3 = 32.5, IQR = 24.5
   Lower = 8 - 36.75 = -28.75 (impossible)
   Upper = 32.5 + 36.75 = 69.25
   Outlier: 120 minutes (survived longer than others)

4. "Median survival time was 12 minutes. Most passengers didn't survive long, with one outlier (120 min) who survived much longer. High std dev (17.3) indicates variable survival times."

</details>

### Exercise 2: Feature Quality Assessment

**Titanic PassengerId variable:** 1, 2, 3, ..., 891

What would this variable's:

1. Mean, median, mode be?
2. Std dev be?
3. Skewness be?
4. Tell you?

<details>
<summary>Click for answers</summary>

**Answers:**

1. Mean ≈ 446, Median = 446, Mode = None (all unique)
2. Std dev ≈ 257 (large spread from 1-891)
3. Skewness ≈ 0 (symmetric, evenly distributed)
4. **This tells you:** PassengerId is just a row number. It has NO PREDICTIVE VALUE.
   Uniform distribution (1-891) with no pattern → useless for modeling.
   **Action:** Drop this feature before building any model.

</details>

### Exercise 3: Interpreting Skewness

Which of these would be:

- Symmetric (skewness ≈ 0)?
- Right-skewed (skewness > 0)?
- Left-skewed (skewness < 0)?

1. Human ages at death
2. Income in a country
3. Test scores in a hard exam
4. Height of adult males

<details>
<summary>Click for answers</summary>

**Answers:**

1. **Left-skewed:** Most people die at old ages, few die young
2. **Right-skewed:** Most people lower income, few very wealthy
3. **Left-skewed:** If hard, most score low, few score high
4. **Symmetric:** Heights roughly normally distributed

</details>

---
