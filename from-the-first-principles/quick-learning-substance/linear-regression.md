# Linear Regression: 25 Interview-Level Questions

These questions test deep understanding, not memorization. They cover theory, implementation, edge cases, and production scenarios.

---

## 1. Feature Scaling and Convergence

**Question:** You train a linear regression model with two features: one ranging from 0-5000 (e.g., house square footage) and another ranging from 1-10 (e.g., number of rooms). Without feature scaling, the optimization would be extremely slow. Explain why the learning rate affects these two features differently, and what happens to the gradient of each weight during the first iteration.

**Why it matters:** Tests understanding of gradient magnitude differences and why feature scaling is critical, not optional.

**Answer:**

**Key insight:** Gradients scale with feature magnitude. A feature with range 0-5000 will produce gradients ~1000x larger than a feature with range 1-10. This causes the learning rate to be too large for one feature and too small for the other.

**Deeper thinking:** Without scaling, the algorithm optimizes the high-magnitude feature first, leaving the low-magnitude feature barely updated until the high-magnitude feature converges. This is inefficient, not mathematically wrong.

---

## 2. The Bias-Variance Trade-off

**Question:** You observe that your linear regression model has low training error but significantly higher validation error. Is this always a sign of overfitting? Explain what could cause this gap, and discuss whether linear regression is even capable of "overfitting" in the traditional sense given its simplicity.

**Why it matters:** Tests understanding that underfitting can cause the same symptoms as overfitting, and challenges assumptions about linear regression's capacity.

**Answer:**

**Key insight:** Linear regression is a high-bias, low-variance model. It can't capture complex patterns (underfitting), but its simplicity means it generalizes well. Validation error > training error can mean:

- Data distribution changed (data drift)
- Validation set is harder than training set
- Model is too simple (underfitting)

It's not necessarily overfitting.

---

## 3. Extrapolation vs Interpolation

**Question:** Your training data contains house prices for houses ranging from 1000-5000 square feet. A user asks for a prediction for a 10,000 square foot house. Your model predicts $2 million. Discuss whether this prediction is reliable. What assumptions does linear regression make that break when extrapolating? How would you handle this in production?

**Why it matters:** Tests understanding of model scope, assumptions, and production constraints.

**Answer:**

**Key insight:** Linear regression assumes the relationship continues indefinitely in both directions. Extrapolating 10,000 sqft from training data of 1000-5000 sqft violates this assumption severely. The model has no information about how things behave in unexplored regions.

**Production solution:** Implement input validation to reject out-of-range requests, or return "unreliable" predictions with warnings.

---

## 4. The Normal Equation Alternative

**Question:** Linear regression can be solved two ways: (1) Gradient Descent (iterative), or (2) Normal Equation (closed-form). Why would someone choose gradient descent over the Normal Equation if the closed-form solution exists? Under what conditions does the Normal Equation fail?

**Why it matters:** Tests deep understanding of trade-offs between methods and numerical stability.

**Answer:**

**Key insight:**

- Normal Equation: Solves directly via (X^T X)^-1 X^T y. Requires computing matrix inverse, which is O(n³) where n is number of features. Can fail if X^T X is singular.
- Gradient Descent: Iterative, requires choosing learning rate, but scalable to millions of features.

Choose gradient descent when n is large (256 features is manageable, but millions require it).

---

## 5. Feature Correlation and Multicollinearity

**Question:** You notice that two of your 256 features are highly correlated (r > 0.95). Does this create a problem for linear regression? Explain what happens to the weights and why. How would you detect this problem programmatically?

**Why it matters:** Tests understanding of linear algebra, the singular matrix problem, and numerical stability.

**Answer:**

**Key insight:** If two features are perfectly correlated, X^T X becomes singular, and the inverse doesn't exist. The weights become unstable and unreliable. Small changes in data cause large weight changes.

**Detection:** Calculate correlation matrix, check condition number of X^T X, or fit the model and observe very large weights.

---

## 6. The Role of Bias in Predictions

**Question:** If you set the bias to zero (force the regression line to pass through the origin), how does this change the nature of predictions? Give a concrete example where forcing bias=0 would be fundamentally wrong. Can you recover from this mistake after training?

**Why it matters:** Tests understanding of the intercept's role and model constraints.

**Answer:**

**Key insight:** Bias allows the line to shift vertically. Without it, you force predictions to be 0 when all features are 0, which is wrong for most real problems.

**Example:** If all features are 0, you'd predict $0 houses, which is nonsense. The bias should be ~$50k (baseline land value).

**Recovery:** No, you can't recover. The model is trained with this constraint from the start.

---

## 7. Cost Function Selection and Outliers

**Question:** You have a dataset where 95% of houses cost $200k-$300k, but 5% are luxury properties worth $5M+. You trained using MSE and got poor results. You switched to MAE and improved. Explain the mechanism of why MSE failed and MAE worked. What's the trade-off you made?

**Why it matters:** Tests understanding of cost function behavior and their impact on learned weights.

**Answer:**

**Key insight:** MSE squares errors, so a $5M luxury property with $1M prediction error contributes (1M)^2 = 1 trillion to the cost. This dominates, pulling the entire model toward fitting the outliers.

MAE treats all errors linearly: $1M error = 1M cost. Outliers have impact but don't dominate.

**Trade-off:** With MSE, the model fits the center of the data well. With MAE, it's more balanced but doesn't penalize worst-case predictions as hard.

---

## 8. Gradient Descent Convergence

**Question:** You observe that your loss curve looks like this: [100, 50, 25, 12, 6, 6, 6, 6, 6...]. The loss stopped decreasing after iteration 5. Is this normal? How do you distinguish between true convergence and being stuck in a local minimum? (Hint: Linear regression has only one global minimum.)

**Why it matters:** Tests understanding of convergence diagnostics and the convexity property of linear regression.

**Answer:**

**Key insight:** Loss stopping at a constant value is normal convergence. Since linear regression is convex, this constant is the true minimum, not a local minimum.

To verify: Check if gradients are near zero. If gradients are zero, you've converged.

---

## 9. Learning Rate and Stability

**Question:** With learning rate α=0.1, your loss converges smoothly. With α=0.5, the loss oscillates but still decreases. With α=1.0, the loss diverges to infinity. Explain the mechanism causing divergence. Is there a theoretical maximum learning rate?

**Why it matters:** Tests understanding of the gradient descent update rule and numerical stability.

**Answer:**

**Key insight:** Update rule: w := w - α * ∇J

If α is too large, the update can be larger than needed to reach the minimum, causing oscillation and divergence.

For quadratic functions, there's a theoretical maximum learning rate related to the eigenvalues of the Hessian. Beyond it, divergence is guaranteed.

---

## 10. Feature Scaling Consistency in Production

**Question:** During training, feature 1 had mean=2000, std=800. Six months later, you load the model and receive new data where feature 1 has mean=2100, std=950. Should you update the feature scaling parameters? What happens if you do? What if you don't?

**Why it matters:** Tests understanding of data drift and the critical importance of consistent preprocessing.

**Answer:**

**Key insight:** If you update the scaling parameters, you're retraining the model without retraining the weights. The relationship between raw features and weights is broken.

**Correct approach:** Keep the training scaling parameters fixed. If data distribution changes significantly (data drift), retrain the entire model.

---

## 11. The Meaning of Negative Weights

**Question:** Your trained model has a weight of -500 for the "age" feature. This means older houses are cheaper. But what if the true relationship is quadratic (very new houses and very old houses are cheaper, middle-aged houses are expensive)? How would linear regression handle this? Is the negative weight wrong?

**Why it matters:** Tests understanding of model limitations and the importance of understanding assumptions.

**Answer:**

**Key insight:** A negative weight means older houses are cheaper. If the true relationship is quadratic, linear regression can't capture it. The weight is "correct" given the linear assumption, but the assumption is violated.

This is a **model limitation**, not a problem with the algorithm.

---

## 12. Regularization and Linear Regression

**Question:** You've implemented basic linear regression without regularization. Adding L2 regularization (penalty on large weights) improves validation performance. Explain why smaller weights would help. What's the trade-off you're making?

**Why it matters:** Tests understanding of overfitting prevention even in simple models, and the bias-variance trade-off.

**Answer:**

**Key insight:** L2 regularization adds penalty for large weights: Total Cost = MSE + λ * (sum of squares of weights)

Smaller weights mean simpler models, less prone to fitting noise. The λ parameter controls the trade-off between fitting the training data and simplicity.

---

## 13. The Impact of Adding a Feature

**Question:** You train a model with 255 features. Then you add one more (random noise as feature 256). Does the training error increase, decrease, or stay the same? Why? Would validation error increase or decrease? What does this tell you?

**Why it matters:** Tests understanding of how models respond to irrelevant features and why validation sets are important.

**Answer:**

**Key insight:** Training error will either stay the same or decrease (never increase). The extra feature might capture noise in the training set, appearing to improve the model.

Validation error will increase because the noise doesn't generalize. This is overfitting on a noise feature.

---

## 14. Symmetry of Prediction Errors

**Question:** Your model's predictions are sometimes $10k too high and sometimes $10k too low. The mean error is zero. Is the model good? Does the cost function care about this symmetry? What does this tell you about using mean error vs mean absolute error vs mean squared error?

**Why it matters:** Tests understanding of error metrics and what they optimize for.

**Answer:**

**Key insight:** Mean error = 0 means the model is unbiased on average. But mean squared error cares about magnitude, not direction.

- MSE penalizes large errors regardless of direction
- MAE penalizes magnitude of error
- Mean error (without absolute value) is useless—errors cancel out

---

## 15. Batch Size and Convergence

**Question:** You have 1000 training examples. With batch_size=1 (SGD), training is noisy. With batch_size=1000 (batch GD), training is smooth. With batch_size=32 (mini-batch), it's in between. Does any of these reach a better final solution? If not, why use mini-batch?

**Why it matters:** Tests understanding of the trade-off between convergence stability and computational efficiency.

**Answer:**

**Key insight:** All batch sizes converge to the same solution eventually. The difference is:

- Batch GD: Smooth, predictable, slow convergence to the minimum
- SGD: Noisy, chaotic, but reaches similar solution faster in wall-clock time (due to more frequent updates)
- Mini-batch: Balances both

Choice depends on computational resources, not final solution quality.

---

## 16. The Cold Start Problem

**Question:** You just deployed your model. A new user provides features that are unlike anything in your training data (but within reasonable ranges). Your model returns a confident prediction. Is this prediction reliable? How would you quantify uncertainty?

**Why it matters:** Tests understanding of extrapolation risk and the need for confidence intervals in production.

**Answer:**

**Key insight:** The model's confidence is based on training data, not on whether new data is "typical." A prediction is confident but potentially unreliable.

**Solution:** Add uncertainty quantification (confidence intervals based on residuals) or request human review for unusual cases.

---

## 17. Standardization vs Normalization

**Question:** You normalized features using (x - mean) / std (standardization). Someone else normalized using (x - min) / (max - min) (min-max scaling). Do these lead to different trained weights and biases? Different predictions on new data? Explain why or why not.

**Why it matters:** Tests understanding of whether the choice of scaling method matters for final predictions.

**Answer:**

**Key insight:** The final predictions will be the same. The weights will differ (because input scaling differs), but the mathematical relationship is preserved.

Why? The scaling is just a change of variables. The model learns the relationship in the scaled space, but the predictions in the original space are identical.

---

## 18. The Matrix Rank Problem

**Question:** If you have 256 features but only 200 training examples, mathematically, you have more parameters than data points. Does gradient descent still work? What might go wrong? How is this different from the closed-form Normal Equation?

**Why it matters:** Tests understanding of underdetermined systems and why data quantity matters.

**Answer:**

**Key insight:** Gradient descent still works, but the system is underdetermined. Multiple weight combinations produce the same training error. Which one you get depends on initialization and the specific gradient descent path.

The Normal Equation fails because X^T X is singular (rank deficient). Gradient descent doesn't directly use matrix inversion, so it can still work, though solutions are unstable.

---

## 19. Interpreting Weights in Context

**Question:** Feature weight for square footage is 150 ($/sqft). But square footage is normalized during training using mean=2000, std=800. So what does the weight 150 actually represent? How do you translate it back to the original scale?

**Why it matters:** Tests understanding of the difference between weights on normalized vs raw features, and the importance of feature engineering documentation.

**Answer:**

**Key insight:** The weight of 150 is for the normalized feature, not the raw feature. To interpret it on raw features:

- Raw feature: (x - 2000) / 800
- Normalized contribution: `weight * normalized = 150 * ((x - 2000) / 800)`
- To convert back: This is complicated and requires chain rule.

**Better approach:** Document weights carefully, showing they apply to normalized features.

---

## 20. Model Monitoring and Drift

**Question:** Six months after deployment, you notice that the model's predictions are systematically 5% too low compared to actual prices. The market didn't change much. What could cause this? Is retraining the model the right solution?

**Why it matters:** Tests understanding of data drift, concept drift, and when retraining is necessary vs harmful.

**Answer:**

**Key insight:** Systematic bias could mean:

- Market changed (data drift)
- Seasonal effect
- Model degradation
- Evaluation set bias

Retraining is one solution, but not always the right one. Investigate the root cause first. Retraining blindly can propagate bad data into the new model.

---

## 21. The JSON Serialization Problem

**Question:** You save your model to JSON with full precision (e.g., weight = 150.123456789012345). Due to JSON limitations, it rounds to 150.12. Does this precision loss matter in production? If you do this 256 times (for 256 weights), does the accumulated error become significant?

**Why it matters:** Tests understanding of numerical precision, accumulation of errors, and practical deployment concerns.

**Answer:**

**Key insight:** Rounding one weight to 2 decimal places causes ~0.0001 error per weight. With 256 weights, accumulated error is ~0.025. For a prediction summing 256 features, this could cause a few hundred dollars of error out of a $300k prediction.

For most applications, this is negligible. But for high-precision needs (financial predictions), you might want full precision.

---

## 22. Cross-Feature Interactions

**Question:** Your model has separate weights for "square footage" and "number of bedrooms". But in reality, the price per square foot might differ based on the number of bedrooms (a luxury 1-bedroom sells at premium per sqft). How would you capture this interaction? Does linear regression as you've implemented it allow this?

**Why it matters:** Tests understanding of model capacity and when linear regression is insufficient.

**Answer:**

**Key insight:** Linear regression as implemented fits linear combinations of features: y = w1*x1 + w2*x2 + ...

To capture interactions, you'd need to add engineered features like x1*x2 `(square footage * bedrooms)`, then fit weights to those.

This is still linear regression—you're just changing the input features.

---

## 23. The Fairness Problem

**Question:** Your deployed house price prediction model consistently underestimates prices in certain neighborhoods. When you investigate, you find that the neighborhood wasn't included as a feature. Is this a data problem or a model problem? Can you fix it with more training data?

**Why it matters:** Tests understanding of feature selection, model bias, and the importance of domain knowledge.

**Answer:**

**Key insight:** This is a feature engineering problem. The model learned without neighborhood information, so it can't use it for prediction.

**Solution:** Add neighborhood as a feature. But this reveals a deeper issue: the model has systematic bias due to missing information. This is unfair and should be detected during evaluation.

---

## 24. Uncertainty in Production

**Question:** Your trained model makes a prediction of $350,000 with training RMSE of $15,000. Does this mean the prediction is accurate to ±$15,000? Can you construct a confidence interval? What assumptions must hold for this to be valid?

**Why it matters:** Tests understanding of prediction uncertainty, standard errors, and statistical assumptions.

**Answer:**

**Key insight:** RMSE is the average error magnitude, but it doesn't directly translate to ±$15k confidence intervals.

To build a proper confidence interval, you'd need to assume residuals are normally distributed, calculate the standard error of the prediction, and use t-distribution.

The answer is "maybe ±$15k", but only under specific statistical assumptions.

---

## 25. The Inverse Problem

**Question:** A real estate agent says "I want to list a house at $400,000. What specifications should I recommend to the seller to hit this price?" Can you use your trained linear regression model to answer this? If not, why? If yes, how? Is the answer unique?

**Why it matters:** Tests understanding of the directionality of prediction, the difference between regression and inverse regression, and whether linear models are reversible.

**Answer:**

**Key insight:** You can solve for desired specifications: If $400k = 150*x1 + 1000*x2 - 500*x3 + 50k, then you have one equation with 256 unknowns.

Infinite solutions exist. You'd need to add constraints (budget constraints, seller preferences) to find a unique solution.

This is an inverse problem, fundamentally different from prediction.

---

## Meta-Insight

These 25 questions reveal a single truth: **Linear regression is simple in formula but complex in application**.

Understanding when it works (linear relationships, sufficient data, no outliers), when it struggles (extrapolation, interactions, fairness), and how to deploy it reliably (scaling consistency, uncertainty quantification, monitoring) separates practitioners from amateurs.

---
