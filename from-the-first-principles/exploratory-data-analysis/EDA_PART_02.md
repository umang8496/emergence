# Exploratory Data Analysis (EDA) - Part 2: Univariate Analysis

## Introduction

Univariate analysis means analyzing **one variable at a time** in isolation. We examine:

- What values it takes
- How they're distributed
- Whether there are outliers
- Whether there are missing values
- Whether it's appropriate for our algorithms

---

## Part 1: Numerical Features

### 1.1 Age - The Key Predictor

#### Understanding Age

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv')

# Age statistics
print("AGE ANALYSIS:")
print(f"Count: {df['Age'].count()} (missing: {df['Age'].isnull().sum()})")
print(f"Mean: {df['Age'].mean():.2f}")
print(f"Median: {df['Age'].median():.2f}")
print(f"Std Dev: {df['Age'].std():.2f}")
print(f"Min: {df['Age'].min():.2f}")
print(f"Max: {df['Age'].max():.2f}")
print(f"\nPercentiles:")
print(df['Age'].quantile([0.25, 0.5, 0.75]))
```

**Output:**

```text
AGE ANALYSIS:
Count: 714 (missing: 177)
Mean: 29.70
Median: 28.00
Std Dev: 14.53
Min: 0.42
Max: 80.00

Percentiles:
0.25    20.0
0.50    28.0
0.75    38.0
```

#### Why This Matters for Your Models

**For Logistic Regression:**

```python
"""
AGE SCALE PROBLEM:
- Range: [0.42, 80]
- If we don't scale, age coefficient will be small
- Gradient descent might ignore age updates

Example:
Without scaling:
  w_age = 0.01 (tiny because range is 0-80)
  
With scaling:
  w_age = 0.5 (larger because range is -3 to +3)

Both are equivalent mathematically, but affects:
- Gradient magnitude
- Learning rate choice
- Convergence speed
"""

print("Age will need standardization before logistic regression!")
```

**For KNN:**

```python
"""
KNN DISTANCE PROBLEM:
Without scaling: Fare dominates (range 0-512)
With scaling: Age contributes equally

Example distance between two passengers:
Without scaling:
  d = sqrt((fare1-fare2)^2 + (age1-age2)^2)
  d ≈ sqrt(100^2 + 10^2) = sqrt(10100) ≈ 100
  (fare difference of 100 dominates age difference of 10)

With scaling (both 0-1):
  d = sqrt((fare1'-fare2')^2 + (age1'-age2')^2)
  d ≈ sqrt(0.2^2 + 0.1^2) = sqrt(0.05) ≈ 0.22
  (both contribute equally)
"""

print("Age MUST be scaled before KNN!")
```

#### Visualization: Age Distribution

```python
# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(df['Age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Age Distribution (Histogram)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age (years)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {df["Age"].mean():.1f}')
axes[0, 0].axvline(df['Age'].median(), color='green', linestyle='--', label=f'Median: {df["Age"].median():.1f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(df['Age'].dropna(), vert=True)
axes[0, 1].set_title('Age Distribution (Box Plot)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Age (years)')
axes[0, 1].grid(True, alpha=0.3)

# Density plot
df['Age'].plot(kind='density', ax=axes[1, 0], color='purple', linewidth=2)
axes[1, 0].set_title('Age Distribution (Density)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Age (years)')
axes[1, 0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_age = np.sort(df['Age'].dropna())
axes[1, 1].plot(sorted_age, np.arange(1, len(sorted_age)+1) / len(sorted_age), linewidth=2)
axes[1, 1].set_title('Age Distribution (Cumulative)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Age (years)')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('age_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: age_distribution.png")
```

#### Key Insights: Age

```python
print("""
AGE INSIGHTS:

1. DISTRIBUTION:
   - Roughly bell-shaped (normal-ish distribution)
   - Right-skewed (more people in 20-40 age group)
   - Some very young (babies) and very old (80+)

2. MISSING VALUES:
   - 177 missing (19.9%)
   - Can we ignore? NO! Age affects survival significantly
   - Solution: Impute with median (28) or mean (29.7)

3. OUTLIERS:
   - Youngest: 0.42 years (infant, makes sense)
   - Oldest: 80 years (elderly, makes sense)
   - No obvious statistical outliers, these are realistic values

4. FOR LOGISTIC REGRESSION:
   - Need to scale/normalize before training
   - Linear relationship assumption: Does age linearly affect survival?
     → Older passengers might have had harder time
     → Younger children had better survival ("women and children first")
     → Relationship might NOT be linear!

5. FOR KNN:
   - Must scale to [-3, +3] range typically
   - Age is crucial for distance calculation
   - Missing values make distance calculation impossible
     → Must impute before KNN

RECOMMENDATION:
✓ Keep Age (strong predictor)
✓ Impute missing values with median
✓ Standardize for linear/KNN models
✓ Consider non-linear relationship (might need polynomial)
""")
```

---

### 1.2 Fare - Price Paid

#### Understanding Fare

```python
print("FARE ANALYSIS:")
print(f"Count: {df['Fare'].count()} (missing: {df['Fare'].isnull().sum()})")
print(f"Mean: {df['Fare'].mean():.2f}")
print(f"Median: {df['Fare'].median():.2f}")
print(f"Std Dev: {df['Fare'].std():.2f}")
print(f"Min: {df['Fare'].min():.2f}")
print(f"Max: {df['Fare'].max():.2f}")

# Check for extreme values
print("\nTop 10 most expensive fares:")
print(df['Fare'].nlargest(10))

print("\nCheapest fares:")
print(df['Fare'].nsmallest(10))
```

**Output:**

```text
FARE ANALYSIS:
Count: 891 (missing: 0)
Mean: 32.20
Median: 14.46
Std Dev: 49.69
Min: 0.00
Max: 512.29

Top 10 most expensive fares:
512.329, 262.375, 262.375, 227.525, 227.525, 211.5, 200.3813, 195.8, 183.1, 179.75

Cheapest fares:
0.0, 0.0, 0.0, ... (many zeros)
```

#### Why This Matters

**The Outlier Problem:**

```python
"""
FARE DISTRIBUTION PROBLEM:
- Most passengers: 7-30 pounds
- Some passengers: 500+ pounds
- This is EXTREME OUTLIERS

Why it matters:
1. For Linear Regression:
   - If predicting fare, outliers pull the regression line
   - Model focuses on fitting the extreme cases
   - Predictions for normal cases are worse

2. For Logistic Regression (predicting survival):
   - Fare is a feature, not target
   - Outliers don't break the model, but might indicate class
   - One passenger paid 512 → Likely first class → High survival
   - Model might overfit to this extreme case

3. For KNN:
   - Without scaling: Fare differences dominate
   - Distance to neighbors calculated mostly on fare
   - Age (0-80) vs Fare (0-512): Fare is 6x more important
   - Without scaling, KNN is broken

4. Actual insight:
   - High fare = High class = High survival
   - But outliers might be measurement errors or special cases
"""

# Detect outliers using IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Fare'] < lower_bound) | (df['Fare'] > upper_bound)]

print(f"\nOUTLIER DETECTION (IQR method):")
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
print(f"\nOutlier fares: {sorted(outliers['Fare'].dropna().unique())[:10]}")
```

#### Visualization: Fare Distribution

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram with all data
axes[0, 0].hist(df['Fare'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Fare Distribution (All Data)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Fare (pounds)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Histogram without extreme outliers (for clarity)
axes[0, 1].hist(df[df['Fare'] < 300]['Fare'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Fare Distribution (< 300 pounds)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Fare (pounds)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Box plot
axes[1, 0].boxplot(df['Fare'], vert=True)
axes[1, 0].set_title('Fare Distribution (Box Plot)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Fare (pounds)')
axes[1, 0].grid(True, alpha=0.3)

# Log scale histogram (to see distribution better)
axes[1, 1].hist(np.log1p(df['Fare']), bins=50, color='gold', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Fare Distribution (Log Scale)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Log(Fare + 1)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fare_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: fare_distribution.png")
```

#### Key Insights: Fare

```python
print("""
FARE INSIGHTS:

1. DISTRIBUTION:
   - Heavily right-skewed (exponential-like)
   - Most people paid 0-50 pounds
   - Few outliers paid 200-500 pounds
   - Some paid 0 (why? Maybe crew, or data error)

2. OUTLIERS:
   - Extreme values: 512 pounds (legitimate or error?)
   - Multiple values > 200 pounds (about 2% of data)
   - These represent first-class luxury accommodations
   - Probably not errors, but extreme cases

3. ZERO VALUES:
   - Some passengers paid 0 pounds
   - Could be: Crew (no ticket), data error, or free passage
   - Affects model training (creates artificial class)

4. FOR LOGISTIC REGRESSION:
   - Fare is proxy for social class
   - Strong predictor of survival
   - Right-skew means relationship might not be linear
   - Consider log-transforming: log(Fare + 1)

5. FOR KNN:
   - MUST scale before using
   - Without scaling, distance = mostly based on fare difference
   - Log transformation might help (more uniform distribution)

RECOMMENDATION:
✓ Keep Fare (strong predictor of survival)
✓ Handle zero values (investigate or remove)
✓ Don't remove outliers (they're real first-class passengers)
✓ Consider log transformation for better distribution
✓ Standardize/normalize before KNN or linear models
""")
```

---

## Part 2: Categorical Features

### 2.1 Sex - Gender

#### Understanding Sex

```python
print("SEX ANALYSIS:")
print(df['Sex'].value_counts())
print(f"\nPercentage:")
print(df['Sex'].value_counts(normalize=True) * 100)

# Missing check
print(f"Missing values: {df['Sex'].isnull().sum()}")
```

**Output:**

```text
SEX ANALYSIS:
male      577
female    314

Percentage:
male      64.76%
female    35.24%

Missing values: 0
```

#### Visualization: Sex Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
sex_counts = df['Sex'].value_counts()
axes[0].bar(sex_counts.index, sex_counts.values, color=['lightcoral', 'lightblue'], edgecolor='black')
axes[0].set_title('Sex Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Sex')
for i, v in enumerate(sex_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart
axes[1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
            colors=['lightcoral', 'lightblue'], startangle=90)
axes[1].set_title('Sex Distribution (Percentage)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('sex_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: sex_distribution.png")
```

#### Key Insights: Sex

```python
print("""
SEX INSIGHTS:

1. DISTRIBUTION:
   - Male: 577 (64.8%)
   - Female: 314 (35.2%)
   - No missing values (complete data)

2. ENCODING FOR MODELS:
   - Current: String values (male, female)
   - Required: Numerical (0, 1)
   
   Option A: male=0, female=1
   Option B: female=0, male=1
   
   Choice doesn't matter mathematically, but affects interpretation
   Convention: male=0, female=1 (easier to interpret)

3. CLASS IMBALANCE:
   - More males than females (~2:1 ratio)
   - In survival, might be important ("women first" policy)
   - Need to check: Did males/females have different survival rates?

4. FOR LOGISTIC REGRESSION:
   - Categorical feature → Must encode
   - After encoding: Sex becomes binary feature [0, 1]
   - Coefficient will show: How much does gender affect survival?

5. FOR KNN:
   - Must encode before distance calculation
   - Encoding: male=0, female=1
   - Distance contribution: 0 (same gender) or 1 (different gender)

RECOMMENDATION:
✓ Keep Sex (likely strong predictor)
✓ Encode as: male=0, female=1 (or use one-hot encoding)
✓ No missing values → No imputation needed
""")
```

---

### 2.2 Embarked - Port of Embarkation

#### Understanding Embarked

```python
print("EMBARKED ANALYSIS:")
print(df['Embarked'].value_counts())
print(f"\nPercentage:")
print(df['Embarked'].value_counts(normalize=True) * 100)

# Missing check
print(f"Missing values: {df['Embarked'].isnull().sum()}")

# What do these codes mean?
port_names = {
    'S': 'Southampton (UK)',
    'C': 'Cherbourg (France)',
    'Q': 'Queenstown (Ireland)'
}

print("\nPort Information:")
for code, name in port_names.items():
    count = (df['Embarked'] == code).sum()
    pct = count / len(df[df['Embarked'].notna()]) * 100
    print(f"{code} - {name}: {count} ({pct:.1f}%)")
```

**Output:**

```text
EMBARKED ANALYSIS:
S    644
C    168
Q     77

Missing values: 2

Port Information:
S - Southampton (UK): 644 (72.3%)
C - Cherbourg (France): 168 (18.9%)
Q - Queenstown (Ireland): 77 (8.7%)
```

#### Visualization: Embarked Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
embarked_counts = df['Embarked'].value_counts()
port_full_names = ['Southampton', 'Cherbourg', 'Queenstown']
axes[0].bar(embarked_counts.index, embarked_counts.values, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
axes[0].set_title('Embarkation Port Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Port')
axes[0].set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
for i, v in enumerate(embarked_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart
axes[1].pie(embarked_counts.values, labels=['Southampton', 'Cherbourg', 'Queenstown'], 
            autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], startangle=90)
axes[1].set_title('Embarkation Port Distribution (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('embarked_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: embarked_distribution.png")
```

#### Key Insights: Embarked

```python
print("""
EMBARKED INSIGHTS:

1. DISTRIBUTION:
   - Southampton (S): 644 (72.3%) - UK port
   - Cherbourg (C): 168 (18.9%) - French port
   - Queenstown (Q): 77 (8.7%) - Irish port

2. MISSING VALUES:
   - 2 missing (0.22%) - negligible
   - Solution: Impute with mode (Southampton, most common)

3. ENCODING OPTIONS:

   Option A: One-Hot Encoding
   - Create 3 binary features: [S], [C], [Q]
   - Problem: Creates multicollinearity (3 features, but only 2 are needed)
   - Solution: Drop one column (e.g., drop [S])
   - Result: Only [C] and [Q] needed
   
   Option B: Label Encoding
   - S=0, C=1, Q=2
   - Problem: Implies order (Q > C > S) which isn't true
   - Not appropriate for unordered categories
   
   Option C: Ordinal Encoding (if meaningful order exists)
   - Geographic order? Wealth proxy?
   - Unclear if order matters

4. FOR LOGISTIC REGRESSION:
   - Use one-hot encoding (without multicollinearity trap)
   - Creates features: Embarked_C, Embarked_Q (drop Embarked_S)
   - Coefficients show: Impact of each port on survival

5. FOR KNN:
   - One-hot encoding most appropriate
   - Distance: 0 if same port, 1 if different port
   - Could use label encoding but less interpretable

RECOMMENDATION:
✓ Keep Embarked (might indicate social class/wealth)
✓ Handle 2 missing values with mode (Southampton)
✓ Use one-hot encoding: Embarked_C, Embarked_Q
✓ Drop one category to avoid multicollinearity
""")
```

---

## Part 3: Ordinal Features

### 3.1 Pclass - Passenger Class

#### Understanding Pclass

```python
print("PCLASS ANALYSIS:")
print(df['Pclass'].value_counts().sort_index())
print(f"\nPercentage:")
print(df['Pclass'].value_counts(normalize=True).sort_index() * 100)

# Missing check
print(f"Missing values: {df['Pclass'].isnull().sum()}")

# What do these mean?
class_info = {
    1: 'First Class (Upper)',
    2: 'Second Class (Middle)',
    3: 'Third Class (Lower)'
}

print("\nClass Information:")
for cls, desc in class_info.items():
    count = (df['Pclass'] == cls).sum()
    pct = count / len(df) * 100
    avg_fare = df[df['Pclass'] == cls]['Fare'].mean()
    print(f"Class {cls} - {desc}: {count} ({pct:.1f}%) - Avg Fare: £{avg_fare:.2f}")
```

**Output:**

```text
PCLASS ANALYSIS:
1    216
2    184
3    491

Class Information:
Class 1 - First Class (Upper): 216 (24.2%) - Avg Fare: £87.51
Class 2 - Second Class (Middle): 184 (20.6%) - Avg Fare: £21.09
Class 3 - Third Class (Lower): 491 (55.1%) - Avg Fare: £13.15
```

#### Visualization: Pclass Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
pclass_counts = df['Pclass'].value_counts().sort_index()
class_names = ['First', 'Second', 'Third']
axes[0].bar(pclass_counts.index, pclass_counts.values, color=['gold', 'silver', '#CD7F32'], edgecolor='black', width=0.6)
axes[0].set_title('Passenger Class Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Class')
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(class_names)
for i, cls in enumerate([1, 2, 3]):
    v = pclass_counts[cls]
    axes[0].text(cls, v + 10, str(v), ha='center', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Average fare by class
avg_fare_by_class = df.groupby('Pclass')['Fare'].mean()
axes[1].bar(avg_fare_by_class.index, avg_fare_by_class.values, 
            color=['gold', 'silver', '#CD7F32'], edgecolor='black', width=0.6)
axes[1].set_title('Average Fare by Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Fare (pounds)')
axes[1].set_xlabel('Class')
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(class_names)
for i, cls in enumerate([1, 2, 3]):
    v = avg_fare_by_class[cls]
    axes[1].text(cls, v + 2, f'£{v:.0f}', ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pclass_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: pclass_distribution.png")
```

#### Key Insights: Pclass

```python
print("""
PCLASS INSIGHTS:

1. DISTRIBUTION:
   - First Class: 216 (24.2%) - Richest passengers
   - Second Class: 184 (20.6%) - Middle class
   - Third Class: 491 (55.1%) - Poorest passengers
   - Clear relationship with fare (proxy for wealth)

2. ORDINAL NATURE:
   - Values have meaningful order: 1 < 2 < 3
   - Not just categories, but hierarchy
   - Can keep as numerical (1, 2, 3) without encoding
   - OR use ordinal encoding: Low(1), Medium(2), High(3)

3. NO MISSING VALUES:
   - All 891 passengers have class information

4. PREDICTIVE POWER:
   - First class passengers: Highest survival rate
   - Third class passengers: Lowest survival rate
   - STRONG predictor of survival

5. FOR LOGISTIC REGRESSION:
   - Can keep as numerical (already ordinal)
   - Or one-hot encode (creates 2 binary features)
   - Keeping numerical is simpler, works well
   - Coefficient will show: Impact of class increase on survival

6. FOR KNN:
   - Keeping as numerical: distance = |class1 - class2|
   - First vs Second: distance = 1
   - First vs Third: distance = 2
   - This is appropriate for ordinal data
   - NO scaling needed (already in 1-3 range)

RECOMMENDATION:
✓ Keep Pclass as is (already numerical, ordinal)
✓ Don't scale (small range 1-3, won't dominate)
✓ No missing values (complete data)
✓ Strong predictor of survival (likely important feature)
""")
```

---

## Summary: Univariate Analysis Checklist

```python
summary_checklist = {
    'Age': {
        'Type': 'Numerical',
        'Missing': '177 (19.9%)',
        'Range': '[0.42, 80]',
        'Action': 'Impute + Scale',
        'For_Logistic': 'Scale needed',
        'For_KNN': 'Scale CRITICAL'
    },
    'Fare': {
        'Type': 'Numerical',
        'Missing': '0 (0%)',
        'Range': '[0, 512.29]',
        'Action': 'Scale + Check zeros',
        'For_Logistic': 'Scale helpful',
        'For_KNN': 'Scale CRITICAL'
    },
    'Sex': {
        'Type': 'Categorical',
        'Missing': '0 (0%)',
        'Categories': '2 (male, female)',
        'Action': 'Encode to 0/1',
        'For_Logistic': 'Must encode',
        'For_KNN': 'Must encode'
    },
    'Embarked': {
        'Type': 'Categorical',
        'Missing': '2 (0.22%)',
        'Categories': '3 (S, C, Q)',
        'Action': 'Impute mode + Encode',
        'For_Logistic': 'One-hot encode',
        'For_KNN': 'One-hot encode'
    },
    'Pclass': {
        'Type': 'Ordinal',
        'Missing': '0 (0%)',
        'Range': '[1, 3]',
        'Action': 'Keep as is',
        'For_Logistic': 'No scaling needed',
        'For_KNN': 'No scaling needed'
    }
}

print("UNIVARIATE ANALYSIS SUMMARY:\n")
for feature, info in summary_checklist.items():
    print(f"\n{feature}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
```

---
