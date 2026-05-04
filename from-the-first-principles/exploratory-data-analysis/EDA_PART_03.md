# Exploratory Data Analysis (EDA) - Part 3: Bivariate Analysis

## Introduction

Bivariate analysis means analyzing **relationships between two variables**. We examine:

- How features relate to each other
- How features relate to the target variable (Survived)
- Which features are most predictive
- Which features interact

This is crucial for understanding which features to keep and how they influence predictions.

---

## Part 1: Feature vs Target (Most Important)

### 1.1 Age vs Survival

#### Understanding the Relationship

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv')

# Compare age of survivors vs non-survivors
print("AGE BY SURVIVAL STATUS:\n")
print(df.groupby('Survived')['Age'].describe())

print("\n" + "="*80)

# Statistical summary
survived_age = df[df['Survived'] == 1]['Age'].dropna()
not_survived_age = df[df['Survived'] == 0]['Age'].dropna()

print(f"\nSurvived:")
print(f"  Mean: {survived_age.mean():.2f}, Median: {survived_age.median():.2f}")

print(f"\nNot Survived:")
print(f"  Mean: {not_survived_age.mean():.2f}, Median: {not_survived_age.median():.2f}")

print(f"\nDifference: {survived_age.mean() - not_survived_age.mean():.2f} years")
print(f"→ Survivors were on average {survived_age.mean() - not_survived_age.mean():.2f} years YOUNGER")
```

**Output:**

```text
AGE BY SURVIVAL STATUS:

Survived  count    mean   std   min   25%   50%   75%   max
0         424    30.62  14.17  1.0  21.0  28.0  39.0  74.0
1         290    28.35  14.88  0.42 0.92  1.0  37.0  80.0

Survived:
  Mean: 28.35, Median: 28.00

Not Survived:
  Mean: 30.62, Median: 28.00

Difference: -2.27 years
→ Survivors were on average 2.27 years YOUNGER
```

#### Visualization: Age vs Survival

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution by survival status
axes[0, 0].hist(not_survived_age, bins=30, alpha=0.6, label='Did Not Survive', color='red')
axes[0, 0].hist(survived_age, bins=30, alpha=0.6, label='Survived', color='green')
axes[0, 0].set_title('Age Distribution by Survival Status', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age (years)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot comparison
data_to_plot = [not_survived_age, survived_age]
bp = axes[0, 1].boxplot(data_to_plot, labels=['Did Not Survive', 'Survived'], patch_artist=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('green')
axes[0, 1].set_title('Age Distribution by Survival Status (Box Plot)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Age (years)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Density plot
not_survived_age.plot(kind='density', ax=axes[1, 0], label='Did Not Survive', color='red', linewidth=2)
survived_age.plot(kind='density', ax=axes[1, 0], label='Survived', color='green', linewidth=2)
axes[1, 0].set_title('Age Density by Survival Status', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Age (years)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Age groups and survival rate
age_bins = [0, 5, 12, 18, 35, 60, 100]
age_labels = ['0-5', '5-12', '12-18', '18-35', '35-60', '60+']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

survival_by_age = df.groupby('AgeGroup', observed=True)['Survived'].agg(['mean', 'count'])
survival_by_age['mean'].plot(kind='bar', ax=axes[1, 1], color='steelblue', edgecolor='black')
axes[1, 1].set_title('Survival Rate by Age Group', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Age Group')
axes[1, 1].set_ylabel('Survival Rate')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_xticklabels(age_labels, rotation=45)
axes[1, 1].axhline(y=df['Survived'].mean(), color='red', linestyle='--', label='Overall Avg')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add count labels
for i, (idx, row) in enumerate(survival_by_age.iterrows()):
    axes[1, 1].text(i, row['mean'] + 0.03, f"n={int(row['count'])}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('age_vs_survival.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: age_vs_survival.png")
```

#### Key Insights: Age → Survival

```python
print("""
AGE vs SURVIVAL INSIGHTS:

1. PATTERN:
   - Younger passengers: Higher survival rate
   - Older passengers: Lower survival rate
   - "Women and children first" policy explains this
   - Children (0-12): Very high survival (~67%)
   - Teenagers (12-18): Moderate survival (~42%)
   - Adults (18+): Lower survival (~36%)

2. RELATIONSHIP:
   - Inverse relationship: Younger = Higher chance of survival
   - NOT perfectly linear, but clear trend

3. FOR LOGISTIC REGRESSION:
   ✓ Age is a good predictor
   ✓ Linear relationship assumption: Questionable
     (relationship might be non-linear for very young)
   ✓ Will be significant predictor (p-value < 0.05)

4. FOR KNN:
   ✓ Age is useful feature
   ✓ Younger passengers cluster together with high survival
   ✓ Must scale age to same range as other features

5. BUSINESS INSIGHT:
   "Children had much higher survival rates due to evacuation priority"

RECOMMENDATION:
✓ Keep Age as feature
✓ Be aware it's a strong predictor
✓ Consider creating age bins for non-linear relationships
""")
```

---

### 1.2 Sex vs Survival

#### Understanding the Relationship

```python
print("SEX BY SURVIVAL STATUS:\n")
print(pd.crosstab(df['Sex'], df['Survived'], margins=True))

print("\n" + "="*80)
print("\nSURVIVAL RATE BY SEX:\n")

survival_by_sex = df.groupby('Sex')['Survived'].agg(['sum', 'count', 'mean'])
survival_by_sex.columns = ['Survived', 'Total', 'SurvivalRate']
print(survival_by_sex)

print(f"\nMale survival: {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"Female survival: {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"Difference: {(df[df['Sex']=='female']['Survived'].mean() - df[df['Sex']=='male']['Survived'].mean())*100:.1f}%")
```

**Output:**

```text
SEX BY SURVIVAL STATUS:

Survived     0    1  All
Sex
female      81  233  314
male       468  109  577
All        549  342  891

SURVIVAL RATE BY SEX:

         Survived  Total  SurvivalRate
Sex
female        233    314           0.742
male          109    577           0.189

Male survival: 18.9%
Female survival: 74.2%
Difference: 55.3%
```

#### Visualization: Sex vs Survival

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Stacked bar chart
survival_by_sex_ct = pd.crosstab(df['Sex'], df['Survived'])
survival_by_sex_ct.plot(kind='bar', stacked=False, ax=axes[0], 
                        color=['red', 'green'], edgecolor='black')
axes[0].set_title('Survival Count by Sex', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Sex')
axes[0].legend(['Did Not Survive', 'Survived'], loc='upper right')
axes[0].set_xticklabels(['Female', 'Male'], rotation=0)
axes[0].grid(True, alpha=0.3, axis='y')

# Survival rate by sex
survival_rate = df.groupby('Sex')['Survived'].mean()
bars = axes[1].bar(survival_rate.index, survival_rate.values, 
                   color=['#FF69B4', '#4169E1'], edgecolor='black', width=0.6)
axes[1].set_title('Survival Rate by Sex', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Survival Rate')
axes[1].set_xlabel('Sex')
axes[1].set_ylim([0, 1])
axes[1].set_xticklabels(['Female', 'Male'], rotation=0)
axes[1].axhline(y=df['Survived'].mean(), color='gray', linestyle='--', label='Overall Avg')

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height*100:.1f}%', ha='center', va='bottom', fontweight='bold')

axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sex_vs_survival.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: sex_vs_survival.png")
```

#### Key Insights: Sex → Survival

```python
print("""
SEX vs SURVIVAL INSIGHTS:

1. PATTERN:
   - Female survival: 74.2%
   - Male survival: 18.9%
   - Massive difference: 55.3 percentage points
   - STRONGEST predictor of survival

2. HISTORICAL CONTEXT:
   - "Women and children first" evacuation policy
   - More women placed in lifeboats
   - Most men left on ship

3. RELATIONSHIP:
   - Not numerical, but categorical
   - Female = Much higher survival
   - Male = Much lower survival

4. FOR LOGISTIC REGRESSION:
   ✓ EXCELLENT predictor (huge effect)
   ✓ After encoding (female=1, male=0):
     - Coefficient will be large and positive
     - P-value will be highly significant (< 0.001)

5. FOR KNN:
   ✓ Excellent distance feature
   ✓ Sex=same: distance contribution = 0
   ✓ Sex=different: distance contribution = 1
   ✓ Will heavily influence nearest neighbors

6. BUSINESS INSIGHT:
   "Gender was THE most important factor in survival"
   "The 'women first' evacuation policy dramatically increased female survival"

RECOMMENDATION:
✓ Keep Sex (STRONGEST predictor)
✓ Encode as: female=1, male=0 (affects interpretation)
✓ This feature will likely dominate model
""")
```

---

### 1.3 Pclass vs Survival

#### Understanding the Relationship

```python
print("PCLASS BY SURVIVAL STATUS:\n")
print(pd.crosstab(df['Pclass'], df['Survived'], margins=True))

print("\n" + "="*80)
print("\nSURVIVAL RATE BY CLASS:\n")

survival_by_class = df.groupby('Pclass')['Survived'].agg(['sum', 'count', 'mean'])
survival_by_class.columns = ['Survived', 'Total', 'SurvivalRate']
print(survival_by_class)

# Percentages
for cls in [1, 2, 3]:
    rate = df[df['Pclass']==cls]['Survived'].mean()
    print(f"Class {cls} survival: {rate*100:.1f}%")
```

**Output:**

```text
PCLASS BY SURVIVAL STATUS:

Survived   0    1  All
Pclass
1         80  136  216
2         97   87  184
3        372  119  491

SURVIVAL RATE BY CLASS:

Pclass  Survived  Total  SurvivalRate
1             136    216           0.630
2              87    184           0.473
3             119    491           0.242

Class 1 survival: 63.0%
Class 2 survival: 47.3%
Class 3 survival: 24.2%
```

#### Visualization: Pclass vs Survival

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Survival count by class
survival_by_class_ct = pd.crosstab(df['Pclass'], df['Survived'])
survival_by_class_ct.plot(kind='bar', ax=axes[0], 
                          color=['red', 'green'], edgecolor='black')
axes[0].set_title('Survival Count by Passenger Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Class')
axes[0].legend(['Did Not Survive', 'Survived'], loc='upper right')
axes[0].set_xticklabels(['First', 'Second', 'Third'], rotation=0)
axes[0].grid(True, alpha=0.3, axis='y')

# Survival rate by class
survival_rate = df.groupby('Pclass')['Survived'].mean()
bars = axes[1].bar([1, 2, 3], survival_rate.values, 
                   color=['gold', 'silver', '#CD7F32'], edgecolor='black', width=0.6)
axes[1].set_title('Survival Rate by Passenger Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Survival Rate')
axes[1].set_xlabel('Class')
axes[1].set_ylim([0, 1])
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(['First', 'Second', 'Third'])
axes[1].axhline(y=df['Survived'].mean(), color='gray', linestyle='--', label='Overall Avg')

# Add percentage labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height*100:.1f}%', ha='center', va='bottom', fontweight='bold')

axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pclass_vs_survival.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: pclass_vs_survival.png")
```

#### Key Insights: Pclass → Survival

```python
print("""
PCLASS vs SURVIVAL INSIGHTS:

1. PATTERN:
   - First class: 63.0% survival (best)
   - Second class: 47.3% survival (middle)
   - Third class: 24.2% survival (worst)
   - Clear hierarchy based on social class

2. REASON:
   - First class: Located near lifeboats, got priority
   - Second class: Middle deck, moderate access
   - Third class: Below deck, blocked access, language barriers
   - Wealth determined survival

3. RELATIONSHIP:
   - Inverse relationship: Higher class number = Lower survival
   - Ordinal, so keep as numerical (1, 2, 3)

4. FOR LOGISTIC REGRESSION:
   ✓ Good predictor (clear effect)
   ✓ Linear relationship: Higher class → Lower survival
   ✓ Coefficient will be negative
   ✓ Already numerical, no encoding needed

5. FOR KNN:
   ✓ Good distance feature
   ✓ Distance = |class1 - class2|
   ✓ First vs Third: distance = 2 (largest)
   ✓ No scaling needed (range is just 1-3)

6. INTERACTION WITH SEX:
   ✓ Sex affects survival more than class
   ✓ But within each class, class still matters
   ✓ First class women: ~97% survival (!!)
   ✓ Third class men: ~19% survival

RECOMMENDATION:
✓ Keep Pclass (good predictor)
✓ No encoding needed (already ordinal numerical)
✓ Check interactions with Sex (might be combined effect)
""")
```

---

## Part 2: Feature vs Feature (Correlations)

### 2.1 Feature Correlations

#### Numerical Features Correlation

```python
# Select numerical features
numerical_features = df[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Survived']].copy()

# Compute correlation matrix
correlation_matrix = numerical_features.corr()

print("CORRELATION MATRIX:\n")
print(correlation_matrix)

# Correlation with target (Survived)
print("\n" + "="*80)
print("\nCORRELATION WITH SURVIVAL:\n")
target_corr = correlation_matrix['Survived'].sort_values(ascending=False)
print(target_corr)
```

**Output:**

```text
CORRELATION MATRIX:
               Age  SibSp  Parch   Fare  Pclass  Survived
Age         1.0000 -0.0856 0.0764 0.0964 -0.3697  -0.0770
SibSp      -0.0856  1.0000 0.4149 0.1597  0.0830   0.1035
Parch       0.0764  0.4149 1.0000 0.2162  0.0188   0.0815
Fare        0.0964  0.1597 0.2162 1.0000 -0.5494   0.2662
Pclass     -0.3697  0.0830 0.0188 -0.5494 1.0000  -0.3385
Survived    -0.0770  0.1035 0.0815 0.2662 -0.3385  1.0000

CORRELATION WITH SURVIVAL:
Survived   1.0000
Pclass    -0.3385
Fare       0.2662
SibSp      0.1035
Parch      0.0815
Age       -0.0770
```

#### Visualization: Correlation Heatmap

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full correlation heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
axes[0].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# Target correlations only
target_corr_sorted = correlation_matrix['Survived'].drop('Survived').sort_values()
colors = ['red' if x < 0 else 'green' for x in target_corr_sorted.values]
axes[1].barh(range(len(target_corr_sorted)), target_corr_sorted.values, color=colors, edgecolor='black')
axes[1].set_yticks(range(len(target_corr_sorted)))
axes[1].set_yticklabels(target_corr_sorted.index)
axes[1].set_xlabel('Correlation with Survival')
axes[1].set_title('Feature Importance (Correlation with Target)', fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='black', linewidth=0.8)
axes[1].grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(target_corr_sorted.values):
    axes[1].text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                va='center', ha='left' if v > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved: correlation_heatmap.png")
```

#### Key Insights: Correlations

```python
print("""
CORRELATION WITH SURVIVAL INSIGHTS:

1. STRONGEST PREDICTORS (in order):
   - Pclass: -0.339 (negative: higher class → lower survival)
   - Fare: +0.266 (positive: higher fare → higher survival)
   - SibSp: +0.104 (weak: having siblings helps slightly)
   - Parch: +0.082 (weak: having parents helps slightly)
   - Age: -0.077 (weak: younger → slightly higher survival)

2. INTERPRETATION:
   - Pclass is strongest (but remember Sex is stronger!)
   - Fare and Pclass are related (rich = high class = high fare)
   - Age effect is weak (overshadowed by class)
   - SibSp/Parch effect is weak

3. MULTICOLLINEARITY CONCERNS:
   - Fare & Pclass: -0.549 (moderately correlated)
     → Both indicate wealth, redundant information
     → Might want to keep only one for linear regression
   - SibSp & Parch: 0.415 (family size indicators)
     → Could combine as "HasFamily" binary feature

4. FOR LOGISTIC REGRESSION:
   - If using both Fare and Pclass: Multicollinearity warning
   - Solution: Keep Pclass (simpler, ordinal), drop Fare
   - OR: Keep Fare (continuous), drop Pclass

5. FOR KNN:
   - Correlations don't directly apply (uses distances)
   - But redundant features (Fare/Pclass) might overweight wealth
   - Consider feature selection or dimensionality reduction

RECOMMENDATION:
✓ Pclass: Keep (simple, strong predictor)
✓ Fare: Consider dropping (correlated with Pclass)
✓ SibSp/Parch: Consider combining into "FamilySize"
✓ Age: Keep (weak but has information)
""")
```

---

## Part 3: Summary Recommendations

### Final Feature Selection & Preprocessing

```python
print("""
BIVARIATE ANALYSIS SUMMARY & RECOMMENDATIONS:

FEATURES TO KEEP:
✓ Sex (MUST KEEP - strongest predictor)
✓ Pclass (MUST KEEP - strong predictor)
✓ Age (KEEP - moderate predictor, must impute missing)
✓ Fare (OPTIONAL - correlated with Pclass, can drop)
✓ SibSp + Parch (KEEP or COMBINE - weak but useful)
✓ Embarked (KEEP - might indicate class, must impute 2 missing)

FEATURES TO DROP:
✗ PassengerId (just ID, no predictive power)
✗ Name (already extracted any useful info)
✗ Ticket (just tracking number)
✗ Cabin (77% missing, not useful)

PREPROCESSING ACTIONS:

1. MISSING VALUES:
   - Age: 177 missing (19.9%) → Impute with median (28)
   - Embarked: 2 missing (0.2%) → Impute with mode (S)

2. CATEGORICAL ENCODING:
   - Sex: male=0, female=1 (binary)
   - Embarked: One-hot encode (create 2 binary features, drop 1)
             → Embarked_C, Embarked_Q (drop Embarked_S)

3. NUMERICAL SCALING:
   - Age: Standardize (mean=0, std=1) for logistic regression
   - Fare: Standardize (mean=0, std=1) for logistic regression
   - Pclass: NO scaling needed (range 1-3)
   - SibSp: NO scaling needed (small range)
   - Parch: NO scaling needed (small range)

4. OPTIONAL FEATURE ENGINEERING:
   - Create "FamilySize" = SibSp + Parch + 1
   - Create "IsAlone" = (FamilySize == 1)
   - Create "HasCabin" = (Cabin != NaN) as proxy for wealth
   - Consider dropping Fare (correlated with Pclass)

5. FINAL FEATURE SET:
   For Logistic Regression:
   - Sex (encoded 0/1)
   - Pclass (keep as 1/2/3)
   - Age (standardized)
   - SibSp (as is)
   - Parch (as is)
   - Embarked_C, Embarked_Q (binary)
   
   For KNN:
   - Same as above, but ALL scaled to [-3, +3] range
     including Pclass, SibSp, Parch

EXPECTED MODEL PERFORMANCE:
- Logistic Regression: ~80-82% accuracy
- KNN (with good k): ~75-80% accuracy
- Most predictions: Driven by Sex (74.2% female survival vs 18.9% male)
""")
```

---
