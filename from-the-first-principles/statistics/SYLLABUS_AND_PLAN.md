<!-- markdownlint-disable MD036 -->

# Statistics Learning Plan: From First Principles

## For Feature Engineering, Data Science, and Machine Learning

---

## INTRODUCTION: Why Statistics Matters

**The Reality:**

- Machine learning without statistics = Building blind (you don't understand what's happening)
- Feature engineering without statistics = Guessing (you don't know what features matter)
- Data science without statistics = Making wrong decisions (p-value misinterpretation, false patterns)

**What statistics gives you:**

- Understanding: Why does this feature matter?
- Confidence: How sure are we of this pattern?
- Decision making: Is this difference real or random?
- Optimization: Which direction should we focus on?

---

## FOUNDATION: What is Statistics?

**Definition (Simple):**
Statistics is the study of **uncertainty and variation**. It helps us:

1. Understand data patterns
2. Make decisions despite incomplete information
3. Quantify confidence in our conclusions
4. Distinguish real patterns from random noise

**Two Main Branches:**

**1. Descriptive Statistics:** Describing what you see in the data

- Mean, median, standard deviation
- Visualizations, distributions
- "This dataset has 891 rows, average age is 29.7 years"

**2. Inferential Statistics:** Making conclusions beyond the data

- Hypothesis testing, confidence intervals
- Estimating population parameters from samples
- "Probably true that younger passengers had higher survival rates"

**For our work (Linear/Logistic Regression, KNN):**

- Descriptive: Understand features before modeling
- Inferential: Test if features are predictive, confidence in predictions

---

## LEVEL 0: ABSOLUTE FUNDAMENTALS

### 0.1 Population vs Sample

**Population:** All possible data

- Example: ALL humans on Earth
- Reality: We never have the entire population

**Sample:** Subset of population we can actually observe

- Example: 1000 people surveyed
- Reality: What we have in practice

**Why it matters:**

- Sample statistic ≠ Population parameter
- Example: Sample mean ≠ True population mean
- Statistics helps us estimate how much they differ

**In our context (Titanic):**

- Population: All passengers that could have existed
- Sample: 891 actual passengers in dataset
- Question: Do patterns in this sample apply to all passengers?

---

### 0.2 Observations, Variables, and Data

**Observation (Row/Example):**

- One person/item
- Example: One passenger on Titanic

**Variable (Column/Feature):**

- One characteristic being measured
- Example: Age, Sex, Fare

**Data:**

- Table of observations × variables

**Types of Variables:**

**Numerical (Quantitative):**

- Continuous: Can take any value (Age, Fare)
  - Example: 25.5 years, 75.25 pounds
  - Infinite possible values between any two points
- Discrete: Only certain values (PassengerId, number of siblings)
  - Example: 1, 2, 3 (not 2.5 siblings)
  - Countable number of values

**Categorical (Qualitative):**

- Nominal: No natural order (Sex, Color)
  - Male/Female have no order
  - Nominal encoding: any mapping works
- Ordinal: Natural order (Pclass: 1st, 2nd, 3rd)
  - First class > Second class > Third class
  - Order matters

**Why it matters for our work:**

- Linear Regression: Needs numerical input
- Logistic Regression: Needs numerical (encode categorical)
- KNN: Needs numerical for distance calculation
- EDA: Different statistics for different types

---

### 0.3 The Concept of Variability

**Variability (Variation):**
The fact that not all observations are identical.

**Example:**

- Passenger 1: Age 25
- Passenger 2: Age 25
- Passenger 3: Age 32
- Passengers vary in age (not all the same)

**Why it matters:**

- Statistics describes variability
- No variability = No need for statistics (if all passengers same age, nothing to analyze)
- More variability = Less certain about conclusions

**In our context:**

- We see variation in Age, Fare, Sex, Survival
- Statistics helps us understand and explain this variation
- Features that explain variation in target are important

---

## LEVEL 1: DESCRIBING A SINGLE VARIABLE

### 1.1 Center (Location): Mean, Median, Mode

**Mean (Average):**

- Sum all values, divide by count
- Formula: μ = Σx / n
- Example: Ages [20, 25, 30] → Mean = 75/3 = 25
- Sensitive to outliers (one extreme value affects mean)

**Median:**

- Middle value when sorted
- Example: Ages [20, 25, 30] → Median = 25 (middle value)
- Example: Ages [20, 25, 30, 100] → Median = 27.5 (average of two middles)
- Robust to outliers (extreme value doesn't affect much)

**Mode:**

- Most common value
- Example: Ages [20, 25, 25, 30] → Mode = 25 (appears twice)
- Example: If no value repeats, no mode exists
- Useful for categorical (Sex: Male appears 577 times, Female 314)

**When to use which:**

- Mean: Normal distribution, no extreme outliers
- Median: Skewed distribution or outliers present
- Mode: Categorical data, what's most common

**In our context (Titanic):**

- Age: Mean=29.7, Median=28 (close means not too skewed)
- Fare: Mean=32.2, Median=14.5 (very different means right-skewed)
- Sex: Mode=Male (appears 577 times vs Female 314)

---

### 1.2 Spread (Variability): Variance and Standard Deviation

**Why measure spread?**

- Two datasets could have same mean but different spreads
- Example: Dataset A = [20, 25, 30] (mean=25, tight)
           Dataset B = [1, 25, 49] (mean=25, loose)
- Same mean, different reliability

**Variance (σ²):**

- Average squared deviation from mean
- Formula: σ² = Σ(x - μ)² / n
- Why square? Makes all deviations positive, emphasizes large deviations
- Units: Squared units (e.g., years²)

**Standard Deviation (σ):**

- Square root of variance
- Formula: σ = √[Σ(x - μ)² / n]
- Back in original units (e.g., years)
- Same information as variance but interpretable scale

**Interpretation:**

- Small σ: Values clustered near mean (consistent)
- Large σ: Values spread far from mean (variable)

**Rule of Thumb (Normal Distribution):**

- ±1σ contains ~68% of data
- ±2σ contains ~95% of data
- ±3σ contains ~99.7% of data

**In our context (Titanic):**

- Age: Mean=29.7, Std=14.5 (most ages between 15-44)
- Fare: Mean=32.2, Std=49.7 (wide variation in fares)

---

### 1.3 Shape: Skewness and Kurtosis

**Skewness:**
Measure of asymmetry in distribution.

**Skewness = 0 (Symmetric):**

- Distribution is balanced
- Mean ≈ Median
- Example: Normal distribution (bell curve)

**Skewness > 0 (Right-skewed):**

- Tail on right side (extreme high values)
- Mean > Median (mean pulled right by outliers)
- Example: Fare (most cheap, few very expensive)

**Skewness < 0 (Left-skewed):**

- Tail on left side (extreme low values)
- Mean < Median (mean pulled left by outliers)
- Example: Age (few newborns, most adults)

**Why it matters:**

- Skewed data affects which center to use (median better for skewed)
- Affects model assumptions (linear regression assumes somewhat normal)
- Feature engineering tool (log-transform can reduce skewness)

**In our context:**

- Age: Slightly left-skewed (some infants, mostly adults)
- Fare: Heavily right-skewed (exponential distribution)

---

### 1.4 Outliers: IQR Method

**Outlier:** Observation far from the rest

**Interquartile Range (IQR) Method:**

1. Q1 (25th percentile): 25% of data below this
2. Q3 (75th percentile): 75% of data below this
3. IQR = Q3 - Q1 (middle 50% of data)

**Outlier definition:**

- Lower bound = Q1 - 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR
- Values outside bounds = Outliers

**Example (Fare):**

- Q1 = 7.91
- Q3 = 31.0
- IQR = 23.09
- Upper bound = 31.0 + 1.5 × 23.09 = 65.63
- Passengers with Fare > 65.63 are outliers
- Actual maximum Fare = 512.29 (extreme outlier)

**Why this method?**

- Robust to extreme values (doesn't use mean/std which are affected by outliers)
- Standard in data science

**In our context:**

- Age: Few outliers (oldest ~80)
- Fare: Many outliers (rich passengers paid 200-512)
- Question: Are outliers errors (remove) or legitimate (keep)?

---

## LEVEL 2: UNDERSTANDING DISTRIBUTIONS

### 2.1 What is a Distribution?

**Distribution:** Description of how values are spread across possible range.

Shows:

- What values are common?
- What values are rare?
- What's the shape?
- Where's the center?

**Common Distributions:**

**Normal Distribution (Bell Curve):**

- Symmetric, bell-shaped
- Mean = Median = Mode
- 68% within 1σ, 95% within 2σ
- Example: Human height
- Why important: Many things in nature follow this
- For us: Linear regression assumes residuals are normal

**Uniform Distribution:**

- All values equally likely
- Flat shape
- Example: Random number generator [0, 1]
- Rare in real data

**Exponential Distribution:**

- Right-skewed, fast decay
- Most values low, few very high
- Example: Fare (most passengers cheap, few expensive)
- Why important: Recognizing helps with transformations

**Binomial Distribution:**

- Discrete outcomes (0 or 1, success/failure)
- Example: Coin flips, Survival (0=died, 1=survived)
- Shape depends on probability of success
- Why important: For classification (logistic regression)

**In our context:**

- Age: Approximately normal
- Fare: Exponential (right-skewed)
- Survived: Binomial (0 or 1, 38% vs 62%)

---

### 2.2 Probability Basics (Intuition)

**Probability:** Likelihood of an event, number from 0 to 1.

- P = 0: Impossible
- P = 0.5: 50-50
- P = 1: Certain

**Example (Survival):**

- P(Survived) = (number who survived) / (total passengers)
- P(Survived) = 342 / 891 = 0.38
- Interpretation: 38% chance a passenger survived (or 38% of passengers survived)

**Conditional Probability:** Probability given something else is true.

- P(Survived | Female): Probability survived GIVEN the passenger was female
- P(Survived | Female) = 233 / 314 = 0.74
- Interpretation: 74% of female passengers survived

**Why it matters for ML:**

- Logistic Regression outputs probabilities (0.73 = 73% chance of class 1)
- Feature importance relates to changing these probabilities
- Understanding conditional probability helps understand model

---

## LEVEL 3: RELATIONSHIPS BETWEEN TWO VARIABLES

### 3.1 Covariance and Correlation

**Covariance:** Measure of how two variables change together.

- Positive covariance: As one increases, the other tends to increase
- Negative covariance: As one increases, the other tends to decrease
- Zero covariance: No linear relationship

**Problem:** Covariance depends on scale

- If you measure height in cm vs meters, covariance changes
- Hard to interpret absolute value

**Correlation (Pearson's r):**

- Standardized covariance (removes scale dependence)
- Range: -1 to +1
- Formula: r = Covariance / (StdDev1 × StdDev2)

**Interpretation:**

- r = +1: Perfect positive correlation (one increases, other always increases)
- r = +0.7: Strong positive (generally increase together)
- r = +0.3: Weak positive (slight tendency to increase together)
- r = 0: No linear correlation
- r = -0.7: Strong negative (generally move opposite)
- r = -1: Perfect negative (one increases, other always decreases)

**Important caveat:** Correlation ≠ Causation

- Example: Ice cream sales and drowning deaths are correlated (both increase in summer)
- But ice cream doesn't cause drowning
- Both caused by third variable: temperature

**In our context (Titanic):**

- Fare & Pclass: r = -0.55 (strong negative: higher class = lower fare number, but 1st class paid MORE)
- Age & Fare: r = +0.10 (weak positive: age slightly associated with fare)
- These correlations tell us which features have redundant information

---

### 3.2 Chi-Square Test (for Categorical Variables)

**Purpose:** Test if two categorical variables are related or independent.

**Logic:**

- Expected: If variables independent, how many would we expect in each cell?
- Observed: How many actually are in each cell?
- Chi-square: Measure of difference between expected and observed
- If chi-square is large: Variables are related
- If chi-square is small: Variables are independent (no relationship)

**In our context:**

- Are Sex and Survived related?
  - Expected: If independent, 38% of males should survive, 38% of females
  - Observed: 19% of males survived, 74% of females survived
  - Chi-square is large: YES, they're related (Sex affects Survival)

**Why it matters:**

- Tells us which categorical features are predictive
- If feature independent of target, it's not useful

---

## LEVEL 4: CONFIDENCE AND UNCERTAINTY

### 4.1 Standard Error

**Problem:** Sample statistics vary.

- If we collect 100 passengers, average age might be 29.5
- If we collect another 100, average might be 30.1
- Which is the "true" average?

**Standard Error:** Measure of variability in sample statistic.

- Formula: SE = σ / √n
- Where σ = population standard deviation, n = sample size

**Interpretation:**

- Small SE: Sample statistic is stable (small variation between samples)
- Large SE: Sample statistic is unstable (large variation between samples)
- Larger sample = Smaller SE (more certain about estimate)

**Example:**

- n = 100, σ = 14.5 (Age std): SE = 14.5 / √100 = 1.45 years
- n = 1000, σ = 14.5: SE = 14.5 / √1000 = 0.46 years
- With 10x more data, SE is 3x smaller (√10 = 3.16)

**In our context:**

- How confident are we about average age estimate?
- With SE = 1.45, we can construct confidence interval

---

### 4.2 Confidence Intervals

**Purpose:** Estimate range where true parameter probably lies.

**95% Confidence Interval:** Range where we're 95% sure the true value lies.

**Formula (for mean):**

- CI = Sample Mean ± (1.96 × SE)
- 1.96 comes from normal distribution (captures 95%)

**Example (Titanic Age):**

- Sample mean = 29.7
- SE = 1.45
- CI = 29.7 ± (1.96 × 1.45) = 29.7 ± 2.84 = [26.86, 32.54]
- Interpretation: We're 95% confident true population age is between 26.86 and 32.54

**Why it matters:**

- Quantifies uncertainty in estimates
- Wider interval = More uncertain
- Narrower interval = More confident

**In our context (Feature Engineering):**

- How confident are we that feature is predictive?
- If CI for correlation crosses 0, maybe not actually correlated

---

### 4.3 Hypothesis Testing (Conceptual)

**Purpose:** Decide if observed pattern is real or just random noise.

**Logic:**

1. Assume null hypothesis is true (no relationship)
2. Calculate how likely we'd see this data if null is true
3. If unlikely (p-value < 0.05), reject null hypothesis
4. Conclude there IS a relationship

**Example (Is Fare different between survivors and non-survivors?):**

- Null: Fare is same for both groups
- Observed: Survivors had higher average fare
- Question: Is this difference real or random chance?
- Test: Calculate p-value

**P-value:** Probability we'd see this result if null hypothesis is true.

- p < 0.05: Unlikely (< 5% chance), reject null (relationship is real)
- p > 0.05: Likely (> 5% chance), fail to reject null (relationship unclear)

**Important caveat:** p < 0.05 doesn't mean 95% probability it's true.

- It means: IF null is true, 5% chance we'd see this data
- Many misinterpretations of p-values in science

**In our context (Feature Engineering):**

- Testing if Age affects Survival: p-value tells us if relationship is statistically significant
- p < 0.05: Age really affects survival (keep this feature)
- p > 0.05: No evidence Age affects survival (might drop this feature)

---

### 4.4 Type I and Type II Errors

**Type I Error (False Positive):**

- Conclude relationship exists when it doesn't
- Say "feature is predictive" when it's not
- Probability = α (usually 0.05)

**Type II Error (False Negative):**

- Fail to conclude relationship when it does exist
- Say "feature is not predictive" when it is
- Probability = β (depends on test power)

**Trade-off:** Lowering α increases β (can't eliminate both)

- α = 0.01 (more strict): Fewer false positives, more false negatives
- α = 0.05 (standard): Balance
- α = 0.10 (less strict): More false positives, fewer false negatives

**In our context:**

- Choosing α = 0.05 means 5% of irrelevant features might seem important by chance
- If we test many features, some will be "significant" by random chance
- Multiple testing correction needed (Bonferroni)

---

## LEVEL 5: CONNECTING STATISTICS TO MACHINE LEARNING

### 5.1 Features and Target Relationships

**Question:** Which features are predictive of target?

**Numerical Feature vs Numerical Target (Linear Regression):**

- Use correlation coefficient r
- Test with t-test or regression coefficient significance
- p-value < 0.05: Feature is predictive

**Categorical Feature vs Numerical Target:**

- Use ANOVA (analysis of variance)
- Compare average target across categories
- p-value < 0.05: Feature affects target

**Categorical Feature vs Categorical Target (Logistic Regression):**

- Use chi-square test
- Compare frequencies across categories
- p-value < 0.05: Feature and target are related

**In our context:**

- Age vs Survived: Numerical vs Categorical → t-test → p < 0.001 (predictive)
- Sex vs Survived: Categorical vs Categorical → chi-square → p < 0.001 (predictive)
- Pclass vs Survived: Ordinal vs Categorical → chi-square → p < 0.001 (predictive)

---

### 5.2 Sample Size and Statistical Power

**Problem:** With small sample, even strong relationships might not be detected.

**Statistical Power:** Probability of detecting true relationship if it exists.

- Power = 1 - β
- High power (0.8+): Likely to detect real relationships
- Low power (< 0.5): Miss many real relationships

**Sample size affects power:**

- Larger sample → Higher power
- Smaller sample → Lower power

**Rule of thumb:** Need minimum effect size × number of features

- For 5 features: 50-200 examples minimum
- For 20 features: 200-1000 examples minimum

**In our context:**

- With 891 passengers, good power for detecting relationships
- With 50 passengers, might miss subtle relationships

---

### 5.3 Multiple Testing Problem

**Problem:** If you test many features, some will appear significant by chance.

**Example:**

- Test 100 unrelated features against target
- At α = 0.05, expect 5% false positives = 5 features
- These 5 appear significant, but actually just random

**Solution: Bonferroni Correction**

- Divide α by number of tests
- α = 0.05 / 100 = 0.0005
- Much stricter threshold (fewer false positives)
- Trade-off: Might miss real relationships (Type II error)

**In our context:**

- If testing 50 features, use α = 0.05 / 50 = 0.001 threshold
- Or just keep top 10-15 correlated features (simpler)

---

### 5.4 Bias and Variance Tradeoff (Statistical Perspective)

**Bias:** Error from wrong assumptions.

- High bias: Model underfits (too simple)
- Example: Linear regression for non-linear data

**Variance:** Error from sensitivity to training data.

- High variance: Model overfits (too complex)
- Example: KNN with k=1 memorizes training data

**Statistical concept:**

- Low bias, low variance: Ideal
- Low bias, high variance: Works on training, fails on test
- High bias, low variance: Works poorly everywhere

**In our context:**

- Fewer features: Lower variance (simpler model), higher bias (miss patterns)
- More features: Higher variance (complex model), lower bias (capture patterns)
- Sweet spot: Right number of features for sample size

---

## LEVEL 6: STATISTICAL TESTS TOOLBOX

### 6.1 T-Test (Numerical vs Categorical with 2 Groups)

**Purpose:** Compare average value of numerical variable between 2 groups.

**Question:** Is average Age different between survivors and non-survivors?

**Output:** t-statistic and p-value

- p < 0.05: Yes, they differ significantly
- p > 0.05: No significant difference detected

**In our context:**

- Survivors: Mean age = 28.4
- Non-survivors: Mean age = 30.6
- T-test → p = 0.001 (significant difference)
- Conclusion: Age is predictive of survival

---

### 6.2 ANOVA (Numerical vs Categorical with 3+ Groups)

**Purpose:** Compare average value across 3+ groups.

**Question:** Is average Fare different between Pclass 1, 2, and 3?

**Output:** F-statistic and p-value

- p < 0.05: At least one group differs from others
- p > 0.05: No significant differences

**In our context:**

- Pclass 1: Mean fare = 87.5
- Pclass 2: Mean fare = 21.1
- Pclass 3: Mean fare = 13.2
- ANOVA → p < 0.001 (significant differences)
- Conclusion: Class affects fare (multicollinearity detected)

---

### 6.3 Chi-Square Test (Categorical vs Categorical)

**Purpose:** Test if two categorical variables are related.

**Question:** Are Sex and Survival related?

**Output:** Chi-square statistic and p-value

- p < 0.05: Variables are related
- p > 0.05: Variables appear independent

**In our context:**

- Female survival: 74%
- Male survival: 19%
- Chi-square → p < 0.001 (highly related)
- Conclusion: Sex is strong predictor of survival

---

### 6.4 Pearson's Correlation Test (Numerical vs Numerical)

**Purpose:** Test if linear relationship exists between two numerical variables.

**Output:** Correlation coefficient r and p-value

- p < 0.05: Relationship exists
- p > 0.05: No linear relationship detected

**In our context:**

- Age vs Fare: r = 0.10, p = 0.08 (weak, not significant)
- Fare vs Pclass: r = -0.55, p < 0.001 (moderate, significant)

---

## LEVEL 7: PROBABILITY DISTRIBUTIONS FOR MODELING

### 7.1 Normal Distribution (Continuous Target, Linear Regression)

**Shape:** Bell curve, symmetric
**Mean, Median, Mode:** All same
**Use case:** Continuous variables that vary randomly
**In modeling:** Linear regression assumes residuals are normally distributed

**Why important:**

- Many natural phenomena are normal
- Math is easier with normal distribution
- Prediction intervals assume normality

**Check:** Q-Q plot (compare data to theoretical normal)

---

### 7.2 Binomial Distribution (Binary Target, Logistic Regression)

**Shape:** Depends on probability

- p = 0.5: Symmetric
- p < 0.5: Right-skewed
- p > 0.5: Left-skewed

**Use case:** Counting successes/failures
**In modeling:** Logistic regression models binary outcomes (binomial)

**Why important:**

- Fundamental for classification
- Cross-entropy loss derived from binomial

---

### 7.3 Poisson Distribution (Count Data, Rare Events)

**Shape:** Right-skewed, only non-negative integers
**Use case:** Counting occurrences of rare events
**In modeling:** If target is count data (number of accidents, page views)

**Why important:**

- Different from normal or binomial
- Needs special models (Poisson regression)

---

## PRACTICAL APPLICATION WORKFLOW

### When Doing EDA

1. **Describe each variable** (mean, median, std, range)
2. **Check for outliers** (IQR method)
3. **Visualize distributions** (histogram, density plot)
4. **Test relationships** (correlation, chi-square)
5. **Document findings** (which features look predictive?)

### When Doing Feature Engineering

1. **Test if feature is predictive** (p-value < 0.05)
2. **Check for multicollinearity** (correlation > 0.8)
3. **Understand effect size** (how strong is relationship?)
4. **Decide: Keep or drop** (based on tests + domain knowledge)

### When Training Models

1. **Verify assumptions** (normal residuals for linear regression)
2. **Check overfitting** (train vs test gap)
3. **Report uncertainty** (confidence intervals on predictions)
4. **Validate statistical significance** (wasn't random luck)

---

## COMMON STATISTICAL MISTAKES IN ML

**Mistake 1: Confusing correlation with causation**

- Two features correlated doesn't mean one causes the other
- Fix: Domain knowledge + experimentation

**Mistake 2: P-hacking (testing until significant)**

- Test many hypotheses, report only significant ones
- Fix: Pre-register hypotheses, use multiple testing correction

**Mistake 3: Ignoring effect size**

- Feature statistically significant but effect tiny
- Fix: Report both p-value AND effect size

**Mistake 4: Failing to check assumptions**

- Assume data is normal without checking
- Fix: Use Q-Q plots, shapiro-wilk test

**Mistake 5: Sample size too small**

- Statistical power too low to detect real relationships
- Fix: Calculate required sample size before data collection

**Mistake 6: Train-test leakage**

- Using information from test set in training
- Fix: Split first, then compute statistics from train only

---

## LEARNING ROADMAP

### Phase 1 (Fundamentals - Days 1-2)

- Understand populations vs samples
- Learn mean, median, mode, std
- Understand distributions (normal, skewed)
- Recognize variable types (numerical, categorical)

### Phase 2 (Descriptive - Days 3-4)

- Calculate and interpret correlation
- Create histograms and boxplots
- Identify outliers
- Understand skewness and kurtosis

### Phase 3 (Relationships - Days 5-6)

- Learn about covariance
- Understand standard error
- Grasp confidence intervals
- Recognize which test to use (t-test, chi-square, correlation)

### Phase 4 (Hypothesis Testing - Days 7-8)

- Understand null hypothesis
- Interpret p-values
- Know Type I and II errors
- Understand alpha and power

### Phase 5 (Integration with ML - Days 9-10)

- Connect statistics to feature importance
- Use tests for feature selection
- Understand assumptions of models
- Validate models statistically

### Phase 6 (Practice - Days 11+)

- Apply on real datasets
- Interpret statistical outputs
- Make decisions based on statistics
- Recognize when statistics mislead

---

## RECOMMENDED LEARNING APPROACH

**DON'T:**

- Memorize formulas
- Treat statistics as math exercise
- Learn all possible tests
- Assume you need advanced statistics

**DO:**

- Understand concepts intuitively
- Learn "why" not just "how"
- Practice with real data
- Learn tests when you need them
- Use statistics to make decisions

**Practice Strategy:**

1. Take Titanic dataset
2. For each feature, calculate mean/median/std
3. Create visualizations
4. Test relationships with target
5. Decide: keep or drop feature
6. See why statistics matters

---

## KEY TAKEAWAYS

1. **Statistics is about uncertainty:** Helping decisions despite incomplete information
2. **Center vs Spread:** Both matter for understanding data
3. **Distributions matter:** Different data has different shapes
4. **Relationships are key:** Which features predict target?
5. **Correlation ≠ Causation:** Just because related doesn't mean causal
6. **Sample size matters:** Larger samples = more reliable conclusions
7. **P-value ≠ probability:** Common misinterpretation
8. **Multiple testing:** Problem with many features
9. **Effect size:** Matters alongside p-value
10. **Practical significance:** Statistical significance isn't always practically important

---
