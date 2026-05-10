<!-- markdownlint-disable MD036 -->

# Level 0: Absolute Fundamentals of Statistics

## Overview

Level 0 establishes the foundational concepts you must internalize before proceeding further. These are not optional—they form the mental models upon which all subsequent statistical concepts rest. This level contains three core concepts that answer the fundamental question: **"What are we measuring and why does it matter?"**

---

## 0.1 Population vs Sample

### Conceptual Understanding

Statistics exists because of a fundamental reality: **we rarely have access to complete information**.  
The study of populations versus samples is about understanding this limitation and working with it effectively.

#### Population: The Theoretical Complete Set

| Aspect            | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| **Definition**    | All possible observations of a phenomenon                              |
| **Scope**         | Exhaustive, complete, theoretical                                      |
| **Accessibility** | Usually impossible or impractical to measure entirely                  |
| **Parameter**     | Values describing populations are called parameters (μ, σ, p)          |
| **Notation**      | Greek letters (μ = population mean, σ = population standard deviation) |

**Examples of Populations:**

- All humans currently living on Earth
- All possible measurements of a manufacturing process
- Every passenger who could have traveled on the Titanic (not just those who did)
- All possible shufflings of a standard deck of cards
- Every molecule of air in a room

**Key insight:** Populations are often infinite or practically infinite. Even if finite, accessing them completely is prohibitively expensive.

#### Sample: The Practical Subset

| Aspect            | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| **Definition**    | A subset of the population we can actually observe                      |
| **Scope**         | Finite, observable, real                                                |
| **Accessibility** | Obtained through data collection efforts                                |
| **Statistic**     | Values computed from samples are called statistics (x̄, s, p̂)            |
| **Notation**      | Latin/English letters (x̄ = sample mean, s = sample standard deviation)  |

**Examples of Samples:**

- 1,000 people surveyed about voting preferences
- Temperature readings from 100 locations across a city
- 891 actual Titanic passengers in our dataset
- 52 cards drawn from a standard deck (with replacement)
- Air samples collected from 5 locations in a building

**Key insight:** Samples are always less than the population. The question becomes: how well does the sample represent the population?

### The Fundamental Problem

```text
Population Parameter (unknown, true)
        ↓
        └─ We want to estimate this from:
        └─ Sample Statistic (known, observable)
        ↓
        Difference = Sampling Error
```

| Scenario             | Population Mean | Sample Mean | Sampling Error |
| -------------------- | --------------- | ----------- | -------------- |
| Ideal (large sample) | μ = 100         | x̄ = 100.5   | 0.5 (small)    |
| Realistic (medium)   | μ = 100         | x̄ = 102.3   | 2.3 (moderate) |
| Poor (small sample)  | μ = 100         | x̄ = 110.5   | 10.5 (large)   |

**Why sampling error exists:**

- Randomness: each sample will be slightly different
- Small samples: less likely to capture true distribution
- Biased sampling: systematic errors in collection process

### Why This Matters for Our Work

**In Titanic Dataset Context:**

| Aspect         | Population                            | Sample                                |
| -------------- | ------------------------------------- | ------------------------------------- |
| **Definition** | All passengers who could board        | 891 actual passengers in data         |
| **Target**     | True survival rate of all             | Observed 38.4% survival rate          |
| **Question**   | Would patterns hold for others?       | How confident in patterns?            |
| **Statistics** | Help estimate population from our 891 | What we observe in our 891 passengers |

**Critical Questions:**

1. **Representativeness:** Does our sample of 891 passengers represent all passengers?
   - Are we missing certain classes? (No, we have 1st, 2nd, 3rd class)
   - Are we overrepresenting survivors? (No, 38.4% is plausible)
   - Could our 891 be fundamentally different from other possible Titanic scenarios?

2. **Generalization:** If we find Age predicts Survival, does it apply to all passengers?
   - If we only have young passengers in our sample but older passengers in reality, our pattern is biased
   - Statistical tests help quantify this uncertainty

3. **Reliability:** How much can we trust our sample statistics?
   - Standard error measures how much sample means vary
   - Confidence intervals estimate where true population parameter lies

### Key Formulations

**Sample Mean (x̄):**

- What we observe: x̄ = (Sum of all observations) / (Number of observations)
- What we assume: x̄ ≈ μ (hopefully close to true population mean)
- Uncertainty: The bigger the sample, the closer usually

**Standard Error (SE):**

- Measures how much sample means vary across different samples
- Formula: SE = σ / √n (where σ is population std dev, n is sample size)
- Insight: Larger samples → smaller SE → more confident estimate

**Practical Example with Numbers:**

Suppose true population mean Age = 30 years, population std dev = 15 years

| Sample Size | Standard Error    | Interpretation                                  |
| ----------- | ----------------- | ----------------------------------------------- |
| n = 10      | 15 / √10 = 4.74   | Large uncertainty, sample means vary widely     |
| n = 100     | 15 / √100 = 1.5   | Moderate uncertainty, sample means cluster      |
| n = 1000    | 15 / √1000 = 0.47 | Small uncertainty, sample means very consistent |

---

## 0.2 Observations, Variables, and Data

### The Structure of Data

Data is organized as a **matrix of observations and variables**. Understanding this structure is essential for selecting correct statistical methods and interpreting results.

```text
        Variable 1    Variable 2    Variable 3
        (Age)         (Sex)         (Survived)
Obs 1   [   22    ]   [ Male    ]   [   0     ]
Obs 2   [   38    ]   [ Female  ]   [   1     ]
Obs 3   [   26    ]   [ Male    ]   [   0     ]
...
Obs 891 [   18    ]   [ Female  ]   [   1     ]

↑                                       ↑
One passenger                       One characteristic
(one observation)                   (one variable)
```

### Observation (Row/Example)

| Aspect            | Explanation                                          |
| ----------------- | ---------------------------------------------------- |
| **Definition**    | Single instance of the entity being measured         |
| **Other terms**   | Row, case, example, instance, individual             |
| **In Titanic**    | One passenger                                        |
| **Uniqueness**    | Each observation is distinct                         |
| **Count in data** | n = 891 observations (passengers) in Titanic dataset |

**Critical Understanding:** Each observation is independent. In Titanic:

- Passenger 1's survival doesn't determine Passenger 2's survival (mostly)
- Each row is a complete, self-contained data point

### Variable (Column/Feature)

| Aspect            | Explanation                                           |
| ----------------- | ----------------------------------------------------- |
| **Definition**    | Single characteristic being measured                  |
| **Other terms**   | Column, feature, attribute, dimension                 |
| **In Titanic**    | Age, Sex, Pclass, Fare, Survived, etc.                |
| **Count in data** | We have 12 variables in original Titanic dataset      |
| **Consistency**   | All observations have (or should have) value for each |

**Critical Understanding:** Variables capture one type of information:

- Age: How old is this passenger?
- Sex: Is this passenger male or female?
- Survived: Did this passenger survive (1=yes, 0=no)?

### Data (The Entire Dataset)

| Aspect         | Explanation                                     |
| -------------- | ----------------------------------------------- |
| **Definition** | Complete table of observations × variables      |
| **Dimensions** | Rows (observations) and Columns (variables)     |
| **Notation**   | Often n (rows) × p (columns)                    |
| **In Titanic** | 891 × 12 matrix (891 passengers, 12 attributes) |
| **Structure**  | Rectangular array where rows are comparable     |

### Types of Variables: Complete Classification

This classification determines **which statistical methods apply**.

#### 1. Numerical Variables (Quantitative)

These variables represent **measured quantities** that have inherent numeric meaning.

##### 1a. Continuous Numerical

| Characteristic      | Details                                            |
| ------------------- | -------------------------------------------------- |
| **Definition**      | Can take any value within a range                  |
| **Possible values** | Infinite (between any two points, more exist)      |
| **Measurement**     | Measured with precision (limited by instrument)    |
| **Examples**        | Age (25.3 years), Fare ($123.45), Height (180.5cm) |
| **Real world**      | Temperature, weight, distance, time, price         |
| **In Titanic**      | Age, Fare, are continuous numerical                |

**Why distinct from discrete?**

- You can meaningfully ask: "Is age between 25 and 26?" Answer: Yes, and 25.5 is possible
- Operations make sense: Average of 20 and 30 is 25 (this is a real value)
- Graphically: Histogram shows many possible values, not distinct peaks

**Statistical handling:** Mean, median, correlation, t-test, linear regression apply well

##### 1b. Discrete Numerical

| Characteristic      | Details                                                 |
| ------------------- | ------------------------------------------------------- |
| **Definition**      | Only certain values are possible                        |
| **Possible values** | Countable, often integers                               |
| **Measurement**     | Counted (not measured continuously)                     |
| **Examples**        | Number of siblings (0, 1, 2...), PassengerId, quantity  |
| **Real world**      | Count data (people, items, occurrences)                 |
| **In Titanic**      | PassengerId, SibSp (siblings), Parch (parents/children) |

**Why distinct from continuous?**

- You cannot have 2.5 siblings (it's meaningless)
- The jump from 1 to 2 is discrete, not infinite gradations
- Graphically: Histogram shows distinct peaks at integer values

**Statistical handling:** Mostly same as continuous for large n; use Poisson distribution for rare count data

#### 2. Categorical Variables (Qualitative)

These variables represent **categories or classes** with no inherent numeric meaning (though we encode them as numbers).

##### 2a. Nominal Categorical (No Order)

| Characteristic   | Details                                               |
| ---------------- | ----------------------------------------------------- |
| **Definition**   | Categories with no natural ordering                   |
| **Relationship** | Categories are equivalent, none is "higher"           |
| **Encoding**     | Numeric values are arbitrary (1=Male, 2=Female)       |
| **Examples**     | Sex (Male/Female), Color (Red/Blue/Green), City names |
| **Real world**   | Religion, nationality, brand, type, category          |
| **In Titanic**   | Sex (Male/Female), Embarked (C/Q/S ports)             |

**Why arbitrary encoding?**

- Saying "Male = 1, Female = 2" doesn't mean Males are "less than" females
- The numbers are just labels for computational purposes
- Different encoding (Male = 0, Female = 1) is equally valid

**Statistical handling:** Mode (most common), frequency tables, chi-square tests; correlation doesn't apply directly

**Common mistake to avoid:** Treating nominal variables as if they have order (ordering matters for ordinal, not nominal)

##### 2b. Ordinal Categorical (With Order)

| Characteristic   | Details                                                |
| ---------------- | ------------------------------------------------------ |
| **Definition**   | Categories with natural, meaningful ordering           |
| **Relationship** | Categories have hierarchy (higher/lower, better/worse) |
| **Encoding**     | Numbers reflect order (but not magnitude)              |
| **Real world**   | Ratings, levels, rankings, Likert scales               |
| **In Titanic**   | Pclass (1st, 2nd, 3rd class - clear order)             |

For example: Education (High School < Bachelor < Master < PhD), Satisfaction (Poor < Fair < Good < Excellent), Ship Class (3rd < 2nd < 1st)

**Why different from nominal?**

- Pclass: 3rd class is "lower" than 1st class (meaningful order)
- Sex: Male is not "lower" or "higher" than Female (no order)
- Education: PhD is "higher" than High School (meaningful order)

**Numeric encoding captures order:**

- Pclass: 1 (first), 2 (second), 3 (third)
- Education: 1 (HS), 2 (Bachelor), 3 (Master), 4 (PhD)
- These numbers have meaning relative to each other (but not additive)

**Critical understanding:** Can you say "Master's is twice as much education as HS?" No! The differences aren't equal.

**Statistical handling:** Mode, frequency, special ordinal correlation; can sometimes treat as numerical if enough categories (5+)

### Decision Tree: Which Type of Variable?

```text
Is it numerical (can you compute mean)?
├─ Yes → Numerical
│  ├─ Can have fractional values? (25.5 years?)
│  │  ├─ Yes → Continuous
│  │  └─ No → Discrete (counting)
└─ No → Categorical
   ├─ Does order matter? (3rd class < 2nd class < 1st class)
   │  ├─ Yes → Ordinal
   │  └─ No → Nominal
```

### Classification Table: Titanic Dataset

| Variable    | Type        | Subtype    | Explanation                                     |
| ----------- | ----------- | ---------- | ----------------------------------------------- |
| Age         | Numerical   | Continuous | Measured in years, can be fractional            |
| Fare        | Numerical   | Continuous | Measured in currency, can be fractional         |
| PassengerId | Numerical   | Discrete   | Counting identifier, integers only              |
| SibSp       | Numerical   | Discrete   | Count of siblings/spouses, integers             |
| Parch       | Numerical   | Discrete   | Count of parents/children, integers             |
| Sex         | Categorical | Nominal    | Male/Female, no order between them              |
| Embarked    | Categorical | Nominal    | Port (C/Q/S), no inherent ordering              |
| Pclass      | Categorical | Ordinal    | Class (1/2/3), clear ordering: 1>2>3            |
| Survived    | Categorical | Nominal    | Binary (0/1), but conventionally treated binary |

### Why Variable Type Matters

This is **critical**—variable type determines which analyses you can perform.

#### Type → Method Mapping

| Your Variables              | Question          | Method to Use           | Can't Use   |
| --------------------------- | ----------------- | ----------------------- | ----------- |
| Numerical + Numerical       | Are they related? | Correlation, regression | Chi-square  |
| Numerical + Categorical(2)  | Do groups differ? | T-test                  | Correlation |
| Numerical + Categorical(3+) | Do groups differ? | ANOVA                   | T-test      |
| Categorical + Categorical   | Are they related? | Chi-square test         | Correlation |

**Example: Why correlation won't work for Sex + Survived**

- Sex: Male=1, Female=2 (arbitrary encoding)
- If correlation = 0.5, what does it mean? Nothing meaningful
- Chi-square correctly asks: "Are frequencies of males/females different in survivors/non-survivors?"

#### Type → Visualization Mapping

| Variable Type      | When Numerical          | When Categorical           |
| ------------------ | ----------------------- | -------------------------- |
| Show distribution  | Histogram, density plot | Bar chart, frequency table |
| Show two variables | Scatter plot            | Grouped bar chart, mosaic  |
| Show spread        | Box plot                | Not applicable             |

### Summary and Practical Implications

**For our Titanic analysis:**

1. Before any statistical test, identify variable types
2. Age vs Survived → Numerical vs Categorical → Use T-test (do survivors have different average age?)
3. Sex vs Survived → Categorical vs Categorical → Use Chi-square (is sex composition different between survivors?)
4. Fare vs Pclass → Numerical vs Ordinal → Can use correlation (does price increase by class?)

---

## 0.3 The Concept of Variability (Variation)

### What is Variability?

**Simple Definition:** The fact that not all observations are identical.

**More Precise:** Variability is the **differences among observations**—the spread, scatter, or dispersion of values in a dataset.

### Why Variability is the Foundation of Statistics

Without variability, statistics would not exist. Consider these scenarios:

**Scenario 1: No Variability**

```text
Age of all Titanic passengers: [29, 29, 29, 29, 29, ... , 29]
```

- Every passenger is exactly 29 years old
- Nothing to analyze—no patterns, no questions to ask
- Statistics has no purpose

**Scenario 2: Complete Variability**

```text
Age of passengers: [1, 5, 9, 15, 22, 29, 34, 45, 58, 78, ...]
```

- No two passengers same age
- Extremely varied
- Hard to predict or explain

**Scenario 3: Realistic (Some Variability)**

```text
Age of passengers: [22, 38, 26, 35, 35, 28, 16, ... ]
```

- Most passengers between 18-50
- Some cluster around common ages
- Some outliers (infants, elderly)
- **This is where statistics thrives**

### Variability vs Uncertainty: Subtle but Important

| Concept          | Meaning                                          | Example                               |
| ---------------- | ------------------------------------------------ | ------------------------------------- |
| **Variability**  | Differences we observe in our sample             | Passengers have different ages        |
| **Uncertainty**  | Not knowing true population value from sample    | Not knowing avg age of all passengers |
| **Relationship** | More variability → More uncertainty in estimates | Wider spread → Harder to predict      |

Statistics addresses both:

1. **Describes variability:** What patterns exist in our data?
2. **Quantifies uncertainty:** How confident are we in our conclusions?

### Sources of Variability

Understanding where variability comes from helps interpret it correctly:

#### Natural Variability (Legitimate)

| Source                     | Example in Titanic                                        | Why it exists               |
| -------------------------- | --------------------------------------------------------- | --------------------------- |
| **Individual differences** | People are different ages, different socioeconomic status | Biology, life circumstances |
| **Biological variation**   | Same age, different health outcomes                       | Genetics, immune system     |
| **Behavior variation**     | Different passengers had different behavioral responses   | Individual choices          |
| **Random chance**          | Pure chance events during disaster                        | Unpredictable events        |

**Statistical perspective:** This is the variability we want to understand and explain through features/causes.

#### Measurement Variability (Problematic)

| Source                | Example in Titanic                   | Why it occurs                  |
| --------------------- | ------------------------------------ | ------------------------------ |
| **Measurement error** | Age recorded as 30 but actually 29.8 | Rounding, instrument precision |
| **Recording error**   | Age entered as 38 but actually 28    | Human error in data entry      |
| **Missing data**      | Age not recorded for some passengers | Data collection failure        |

**Statistical perspective:** This is "noise"—variability we want to minimize through better data collection.

### Formal Understanding of Variability

**How statisticians measure variability:**

Each observation i has value: x_i
The mean is: x̄ = (x₁ + x₂ + ... + x_n) / n

**Deviation from mean:** How far is each observation from center?

- Deviation_i = x_i - x̄

**Total variability:** Sum of all deviations

- ∑(x_i - x̄) = Always 0 (symmetry property)
- This is why we use variance (square deviations) instead

**Variance:** Average squared deviation (measures variability)

- σ² = ∑(x_i - x̄)² / n

**Standard Deviation:** Square root of variance (back to original units)

- σ = √[∑(x_i - x̄)² / n]

### Variability in Our Context: Titanic

#### Example 1: Age Variability

```text
Passengers:  P1(22) P2(38) P3(26) P4(35) P5(29)

Mean = (22 + 38 + 26 + 35 + 29) / 5 = 30

Deviations from mean (30):
P1: 22 - 30 = -8
P2: 38 - 30 = +8
P3: 26 - 30 = -4
P4: 35 - 30 = +5
P5: 29 - 30 = -1

Squared deviations:
P1: (-8)² = 64
P2: (+8)² = 64
P3: (-4)² = 16
P4: (+5)² = 25
P5: (-1)² = 1

Variance = (64 + 64 + 16 + 25 + 1) / 5 = 170 / 5 = 34 years²

Standard Deviation = √34 = 5.83 years
```

**Interpretation:**

- On average, passengers' ages deviate from the mean by about 5.83 years
- Some near 22, some near 38 (8-year deviations)
- This spread is moderate (not all same age, not wildly different)

#### Example 2: Fare Variability (Hypothetical)

Suppose three groups of passengers:

- Group 1: Everyone paid $15 (low variability: σ ≈ $0)
- Group 2: Paid $10-$20 (moderate variability: σ ≈ $3)
- Group 3: Paid $1-$500 (high variability: σ ≈ $100)

| Group | Mean | Std Dev | Interpretation                          |
| ----- | ---- | ------- | --------------------------------------- |
| 1     | $15  | $0      | Completely predictable (no variability) |
| 2     | $15  | $3      | Fairly predictable (tight range)        |
| 3     | $15  | $100    | Very unpredictable (wide range)         |

### Why Variability Matters for Machine Learning

#### 1. **Variability in Features Explains Target**

The fundamental ML question: "Can I use feature X to predict target Y?"

**Example:**

- Null hypothesis: Age has no relationship with Survival
  - Survivors' ages = [22, 35, 28, 45, 19] → Variability high
  - Non-survivors' ages = [38, 42, 55, 48, 60] → Variability high
  - Overlap is massive → Age doesn't explain survival
  
- Alternative hypothesis: Age predicts Survival
  - Survivors' ages = [18, 22, 19, 20, 21] → All young
  - Non-survivors' ages = [50, 55, 62, 48, 58] → All old
  - **Age variability explains Survival variability** → Age is predictive

#### 2. **Residual Variability = Unexplained Variation**

After fitting a model, leftover errors show remaining variability:

```text
Total Variability = Explained by Model + Unexplained (Residual)

Good model: Explained is large, Residual is small
Poor model: Explained is small, Residual is large
```

**Example with Age predicting Survival:**

- We observe ages vary from 1 to 80 (high variability)
- Model predicts: Young survive, Old don't (explains some variability)
- Residuals: Some young still died, some old still survived (remaining variability)
- Question: What explains remaining variability? (Maybe Sex, Class, etc.)

#### 3. **More Variability = More Uncertainty**

**Key formula:** Standard Error = σ / √n

- If σ is large (high variability), SE is large (uncertain estimates)
- If σ is small (low variability), SE is small (confident estimates)

**Practical example:**

- Age (σ ≈ 14 years) → Estimates less certain
- Sex (binary, σ ≈ 0.5) → Estimates more certain
- Features with low variability are harder to use for prediction

### Summary: Why Variability is Fundamental

| Aspect            | Why It Matters                                              |
| ----------------- | ----------------------------------------------------------- |
| **Existence**     | Statistics only applies when variability exists             |
| **Direction**     | Do two variables' variabilities align? (Correlation)        |
| **Magnitude**     | How much uncertainty in our conclusions? (Standard error)   |
| **Explanation**   | Can features explain target variability? (R² in regression) |
| **Decomposition** | How much variability is natural vs noise? (Signal vs noise) |

**Core insight:** Every statistical test boils down to asking: "Is the variability we observe in our data more than we'd expect by random chance alone?"

---

## Key Takeaways from Level 0

### Concept 1: Population vs Sample

- **Population** = All possible data (unknown, theoretical)
- **Sample** = Data we actually have (known, observed)
- **Core problem** = How to use sample to estimate population
- **Solution** = Statistics helps quantify uncertainty in estimates
- **For us** = Our 891 Titanic passengers represent all possible Titanic passengers

### Concept 2: Observations, Variables, and Data Types

- **Observation** = One row (one passenger)
- **Variable** = One column (one characteristic)
- **Data** = n × p matrix (891 passengers × characteristics)
- **Types matter**:
  - Numerical continuous (Age, Fare)
  - Numerical discrete (PassengerId, SibSp)
  - Categorical nominal (Sex, Port)
  - Categorical ordinal (Pclass, Education)
- **Why** = Variable type determines statistical method to use

### Concept 3: Variability

- **Definition** = Differences among observations
- **Existence** = Statistics requires variability to analyze
- **Measurement** = Standard deviation, variance quantify it
- **Uncertainty** = More variability → Less certain conclusions
- **Prediction** = Features that align with target variability are predictive

---

## Practice Exercises (Self-Check)

### Exercise 1: Population vs Sample

**Scenario:** You want to understand exam scores of all students at a university.

1. What is the population in this scenario?
2. What could be a sample?
3. Why might the sample mean differ from population mean?
4. How would larger sample size change your estimate?

<details>
<summary>Click for answers</summary>

1. **Population:** All current and past students of the university (or: all students who could ever attend)
2. **Sample:** 200 randomly selected current students who took exams
3. **Sampling error:** Random chance means our 200 students' average might be 78, but all students might be 75 or 81
4. **Larger sample:** Standard Error = σ / √n, so larger n means smaller SE, which means our estimate gets closer to true value

</details>

### Exercise 2: Variable Classification

**For each variable, identify the type and subtype:**

1. Number of books read per year
2. Favorite color
3. Customer satisfaction rating (1-5 stars)
4. House price in dollars
5. Highest education level completed
6. Number of defects in a batch of 100 products

<details>
<summary>Click for answers</summary>

1. **Discrete Numerical** (countable, integers)
2. **Nominal Categorical** (no order—red isn't "less than" blue)
3. **Ordinal Categorical** (5 stars is "better than" 1 star)
4. **Continuous Numerical** (any value, measured)
5. **Ordinal Categorical** (PhD > Masters > Bachelor > HS > None)
6. **Discrete Numerical** (count, integers from 0-100)

</details>

### Exercise 3: Understanding Variability

**Scenario:** Two exam classes of 10 students each

- Class A scores: [85, 86, 84, 85, 86, 85, 84, 85, 86, 85]
- Class B scores: [70, 92, 88, 78, 95, 82, 76, 91, 85, 63]

1. Calculate mean for each class
2. Which class has more variability?
3. For the class with high variability, why might predictions be less certain?
4. Which class would be easier to teach?

<details>
<summary>Click for answers</summary>

1. **Both have mean ≈ 85** (sum/10 = 850/10)
2. **Class B has much more variability** (scores range 63-95) vs Class A (84-86)
3. **Class B:** High variability means harder to predict student scores; some students need more help (70, 63), others excel (92, 95)
4. **Class A would be easier:** Uniform performance suggests consistent understanding across class

</details>

---
