# Exploratory Data Analysis (EDA) - Part 1: Understanding Your Data

## Introduction: Why EDA Matters

Before we build any machine learning model (linear regression, logistic regression, or KNN), we must understand our data intimately.  
This is EDA.  

### The Reality of ML Projects

```text
Data Collection
    ↓
[EDA and Data Cleaning] ← 60-80% of time
    ↓
[Feature Engineering] ← 15-20% of time
    ↓
[Model Selection] ← 5-10% of time
    ↓
[Model Training] ← 3-5% of time
    ↓
[Deployment] ← 2-3% of time
```

Most people focus on model training (5-10% of effort) but ignore EDA (60-80% of effort).  
This is backwards.  

### What Happens Without EDA?

```text
Scenario 1: You don't check for missing values
→ Model training crashes or silently produces garbage
→ Wasted hours debugging

Scenario 2: You don't understand feature distributions
→ You scale the data wrong
→ Linear regression fails (assumed linear relationship doesn't exist)
→ Logistic regression converges slowly (features have wildly different ranges)
→ KNN performs poorly (irrelevant features dominate distance calculations)

Scenario 3: You don't detect outliers
→ Linear regression weights get pulled by outliers
→ Predictions are unreliable
→ You think your model is bad when it's actually the data

Scenario 4: You don't understand class imbalance
→ For Titanic: 62% survived, 38% died
→ You build a model that predicts "everyone survived"
→ Gets 62% accuracy (looks good!)
→ But completely useless (never predicts death)
```

### The Titanic Dataset: Why It's Perfect for Learning

```text
Why Titanic?

1. MIXED DATA TYPES:
   - Quantitative: Age, Fare, PassengerId
   - Qualitative: Sex, Embarked, Cabin
   - Ordinal: Pclass (1st, 2nd, 3rd class)
   
2. REALISTIC PROBLEMS:
   - Missing values (Age: 177 missing, Cabin: 687 missing)
   - Outliers (some passengers paid 512 pounds, most paid < 100)
   - Class imbalance (62% survived, 38% died)
   - Feature relationships (higher class = higher survival)
   
3. SUITABLE FOR OUR ALGORITHMS:
   - Logistic Regression: Predict survival (binary classification)
   - Linear Regression: Predict fare (regression)
   - KNN: Distance-based learning works well here
   
4. BUSINESS CONTEXT:
   - Clear question: "Who survived and why?"
   - Easy to interpret results: "Women and children first"
   - Actionable insights
```

---

## Part 1: Setting Up and Loading Data

### Step 1: Import Libraries and Load Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
```

**Output:**

```text
Dataset shape: (891, 12)
```

**First few rows:**

| # | PassengerId | Survived | Pclass | Name    | Sex    |  Age | SibSp | Parch | Ticket                 |   Fare | Cabin | Embarked |
|---|-------------|----------|--------|---------|--------|-----:|------:|------:|------------------------|-------:|-------|----------|
| 0 | 1           | 0        | 3      | Braund  | male   | 22.0 |     1 |     0 | A/5 21171              |  7.250 | NaN   | S        |
| 1 | 2           | 1        | 1      | Cumings | female | 38.0 |     1 |     0 | PC 17599               | 71.250 | C85   | C        |
| 2 | 3           | 1        | 3      | Healy   | female | 26.0 |     0 |     0 | STON/O2. 3101282       |  7.925 | NaN   | S        |
| 3 | 4           | 1        | 1      | Ashley  | female | 35.0 |     1 |     0 | 113803                 | 53.100 | C123  | S        |
| 4 | 5           | 0        | 3      | Allen   | male   | 35.0 |     0 |     0 | 373450                 |  8.050 | NaN   | S        |

### Step 2: Understand Data Types and Structure

```python
# Data types
print("Data Types:")
print(df.dtypes)
print("\n" + "="*80)

# Data info
print("\nData Info:")
print(df.info())
print("\n" + "="*80)

# Missing values
print("\nMissing Values:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])
```

**Output:**

```text
Data Types:
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: int64

Missing Values:
            Missing Count  Percentage
Age                  177       19.87%
Cabin                687       77.10%
Embarked               2        0.22%
```

### Step 3: Overview of Data

```python
# Basic statistics
print("Basic Statistics:")
print(df.describe())
print("\n" + "="*80)

# Categorical columns
print("\nCategorical Columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())
```

**Output:**

| Stat  | PassengerId | Survived | Pclass |    Age | SibSp | Parch |    Fare |
|-------|------------:|---------:|-------:|-------:|------:|------:|--------:|
| count |      891.00 |   891.00 | 891.00 | 714.00 |891.00 |891.00 |  891.00 |
| mean  |      446.00 |     0.38 |   2.31 |  29.70 |  0.52 |  0.38 |   32.20 |
| std   |      257.91 |     0.49 |   0.84 |  14.53 |  1.10 |  0.81 |   49.69 |
| min   |        1.00 |     0.00 |   1.00 |   0.42 |  0.00 |  0.00 |    0.00 |
| 25%   |      223.50 |     0.00 |   2.00 |  20.00 |  0.00 |  0.00 |    7.69 |
| 50%   |      446.00 |     0.00 |   3.00 |  28.00 |  0.00 |  0.00 |   14.46 |
| 75%   |      668.50 |     1.00 |   3.00 |  38.00 |  1.00 |  0.00 |   31.00 |
| max   |      891.00 |     1.00 |   3.00 |  80.00 |  8.00 |  6.00 |  512.29 |

---

## Part 2: Column-by-Column Understanding

### Understanding Each Column (Before Any Analysis)

```python
def understand_column(df, col):
    """Get detailed info about a column"""
    print(f"\n{'='*80}")
    print(f"COLUMN: {col}")
    print(f"{'='*80}")
    
    print(f"\nData Type: {df[col].dtype}")
    print(f"Non-null Count: {df[col].count()} / {len(df)} ({df[col].count()/len(df)*100:.1f}%)")
    print(f"Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
    
    if df[col].dtype in ['int64', 'float64']:
        print(f"\nRange: [{df[col].min()}, {df[col].max()}]")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Std Dev: {df[col].std():.2f}")
    else:
        print(f"\nUnique Values: {df[col].nunique()}")
        print("\nValue Counts:")
        print(df[col].value_counts())

# Understand each column
for col in df.columns:
    understand_column(df, col)
```

### What Each Column Represents (Domain Knowledge)

```python
"""
COLUMN MEANINGS:

1. PassengerId: Unique identifier
   - Role: Not useful for prediction (just an ID)
   - Action: Should be removed during feature engineering

2. Survived: Target variable (what we want to predict)
   - 0 = Did not survive
   - 1 = Survived
   - This is why we use LOGISTIC REGRESSION

3. Pclass: Ticket class (proxy for socioeconomic status)
   - 1 = First class (richest)
   - 2 = Second class
   - 3 = Third class (poorest)
   - Role: Strong predictor (higher class = higher survival)

4. Name: Passenger name
   - Role: Not directly useful, but could extract titles (Mr., Mrs., Dr.)
   - This is FEATURE ENGINEERING

5. Sex: Gender (male/female)
   - Role: Strong predictor ("women and children first" policy)
   - This is QUALITATIVE DATA → Need to convert to numbers

6. Age: Age in years
   - Role: Moderate predictor (children had better survival)
   - Issue: 177 missing values (19.9%)

7. SibSp: Number of siblings/spouses aboard
   - Role: Family relationships affect survival
   - Insight: Traveling alone vs with family

8. Parch: Number of parents/children aboard
   - Role: Similar to SibSp
   - Could combine with SibSp as "family size"

9. Ticket: Ticket number
   - Role: Probably not useful (just a tracking number)
   - Action: Should be removed

10. Fare: Ticket price in pounds sterling
    - Role: Another proxy for wealth (similar to Pclass)
    - Issue: 1 missing value, some extreme outliers (512 pounds)
    - This is OUTLIER DETECTION

11. Cabin: Cabin number
    - Role: Could indicate location on ship (affect evacuation)
    - Issue: 77% missing! (687 out of 891)
    - Action: Hard to use, might extract deck letter instead

12. Embarked: Port of embarkation
    - Options: S (Southampton), C (Cherbourg), Q (Queenstown)
    - Role: Might indicate social class
    - Issue: Only 2 missing values
"""

# Practical: Create a column description dictionary
column_info = {
    'PassengerId': {'type': 'ID', 'action': 'drop'},
    'Survived': {'type': 'Target', 'action': 'keep'},
    'Pclass': {'type': 'Ordinal', 'action': 'keep'},
    'Name': {'type': 'Text', 'action': 'engineer (extract title)'},
    'Sex': {'type': 'Categorical', 'action': 'encode'},
    'Age': {'type': 'Numerical', 'action': 'handle missing'},
    'SibSp': {'type': 'Numerical', 'action': 'keep or combine'},
    'Parch': {'type': 'Numerical', 'action': 'keep or combine'},
    'Ticket': {'type': 'ID', 'action': 'drop'},
    'Fare': {'type': 'Numerical', 'action': 'handle outliers'},
    'Cabin': {'type': 'Categorical', 'action': 'engineer or drop'},
    'Embarked': {'type': 'Categorical', 'action': 'encode'},
}

print("Column Actions Plan:")
for col, info in column_info.items():
    print(f"{col:12} → Type: {info['type']:12} → {info['action']}")
```

---

## Part 3: Why This Matters for Your Models

### For Logistic Regression

```python
"""
LOGISTIC REGRESSION ASSUMPTIONS:
1. Binary target (0 or 1) ✓ We have: Survived (0/1)
2. Numerical features (or encoded categorical)
3. Linear decision boundary

WHAT EDA REVEALS:
- Sex: Categorical → Must encode to 0/1
- Pclass: Already numerical → Good
- Age: Missing values → Must handle before model
- Embarked: Categorical → Must encode

FEATURE SCALING:
- Age: Range [0.42, 80] → Large range
- Fare: Range [0, 512.29] → Large range
- Pclass: Range [1, 3] → Small range

Without EDA:
→ You'd feed raw data to logistic regression
→ Age dominates because of its scale
→ Pclass is ignored
→ Model is suboptimal

With EDA:
→ You normalize features to same scale
→ All features contribute fairly
→ Model performs better
"""

print("""
KEY INSIGHT FOR LOGISTIC REGRESSION:
Feature scaling is critical!
Age and Fare have different ranges.
Logistic regression is sensitive to this.

Solution: Standardize all numerical features
""")
```

### For KNN

```python
"""
KNN ASSUMPTIONS:
1. Distance-based: Closer points are similar
2. Requires ALL features to be numerical

WHAT EDA REVEALS:
- Sex: Categorical → Must encode
- Embarked: Categorical → Must encode
- Age: Missing values → Must handle (can't compute distance with NaN)

FEATURE SCALING IS EVEN MORE CRITICAL:
- Without scaling: Fare (0-512) dominates, Age (0-80) ignored
- Distance calculation: sqrt((fare_diff)^2 + (age_diff)^2)
- If fare_diff = 100, age_diff = 5:
  Distance ≈ sqrt(10000 + 25) ≈ 100 (age doesn't matter!)

With scaling (mean=0, std=1):
- Both fare and age contribute equally to distance
- KNN finds truly similar passengers

Without EDA → Wrong distance metric → Wrong neighbors → Wrong predictions
"""

print("""
KEY INSIGHT FOR KNN:
Feature scaling is CRITICAL for KNN!
KNN relies on distance calculations.
Different scales break distance metrics.

Solution: Standardize ALL features before KNN
""")
```

---

## Part 4: Key Decisions Needed (EDA Output)

After EDA, you must answer these questions:

```python
decisions = {
    "Missing Values": {
        "Age": "177 missing (19.9%) → Impute with median? Drop?",
        "Cabin": "687 missing (77%) → Too much, consider dropping or extract deck",
        "Embarked": "2 missing (0.2%) → Impute with mode (most common)"
    },
    
    "Outliers": {
        "Fare": "One passenger paid 512 pounds (max), most < 100 → Investigate",
        "Age": "Some infants (0.42 years), some very old (80) → Normal or outliers?"
    },
    
    "Categorical Features": {
        "Sex": "How to encode? (0/1, male=1 or female=1?)",
        "Embarked": "3 ports → One-hot encode or ordinal?",
        "Cabin": "Drop due to 77% missing, or extract deck letter?"
    },
    
    "Feature Engineering": {
        "Title": "Extract from Name (Mr., Mrs., Dr., etc.)?",
        "FamilySize": "Combine SibSp + Parch?",
        "IsAlone": "Create binary feature for traveling alone?"
    },
    
    "Feature Selection": {
        "PassengerId": "Drop (just an ID, no predictive power)",
        "Ticket": "Drop (just a tracking number)",
        "Name": "Extract title, then drop"
    }
}

print("\nDECISIONS NEEDED AFTER EDA:\n")
for category, items in decisions.items():
    print(f"\n{category}:")
    for item, decision in items.items():
        print(f"  {item}: {decision}")
```

---

## Summary: What EDA Accomplishes

```python
print("""
EDA OUTPUT CHECKLIST:

✓ Understand data types and structure
✓ Identify missing values and their patterns
✓ Detect outliers and understand their nature
✓ Check for class imbalance (62% survived, 38% died)
✓ Understand feature distributions
✓ Identify relationships between features
✓ Understand relationships with target variable

DECISIONS MADE FROM EDA:

1. How to handle missing values
2. Whether to remove outliers
3. How to encode categorical features
4. Which features to keep/drop
5. Whether to create new features
6. How to scale features
7. Which model is appropriate

WITHOUT EDA:
→ You guess, make wrong decisions, model fails

WITH EDA:
→ You understand your data, make informed decisions, model works
""")
```

---
