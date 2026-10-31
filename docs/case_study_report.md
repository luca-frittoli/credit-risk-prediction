# Case Study Report

## Problem formulation and evaluation metrics

"Debt default" occurs when customers do not repay after 21 days, so we can define a boolean `default_flag` that is True if `amount_outstanding_21d` > 0 and False otherwise. A first approach to predicting the probability of debt default could be training a classifier with target `default_flag` and interpreting the classification score as the default probability. However, classifiers suffer from class imbalance (the default rate is 5.4%), and the scores might not reflect "probabilities".

For these reasons, I prefer to solve a regression problem using models that split the feature space into clusters (e.g. a decision tree) and give as output, for a point *x* in the feature space, the mean of `default_flag` over a cluster *C(x)*, i.e. the default rate over *C(x)* in the training set. This is an unbiased estimator of the expected value of the random variable `default_flag` conditioned on *C(x)*, which is by definition the default probability conditioned on *C(x)*.

Evaluation can be tricky using regression metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE): in this case, a dummy model always predicting 0 might perform (apparently) well due to class imbalance. For this reason, I also used the Area Under the ROC Curve (AUC) and the Coefficient of Determination (R-squared).

## Feature & model selection

Here are my considerations after a preliminary feature selection using F-score for classification/regression with `default_flag` as a target:

- The loan amount does not seem very relevant, and the existence of previous debt seems more relevant than the amount of said debt.
- Confirmed/failed payments seem to be more relevant in longer-term time frames (>6m).
- The repaid amounts also seem relevant, but probably due to correlation with the number of confirmed payments. I ignore them here to avoid multicollinearity.
- Some merchant category groups seem relevant.

| Model | AUC | R-squared | MSE | MAE |
|---|---|---|---|---|
| Logistic Regression | 0.5003 | -0.0622 | 0.0557 | 0.0557 |
| Linear Regression | 0.6349 | 0.0095 | 0.0519 | 0.1033 |
| **Decision Tree** | **0.6435** | **0.0142** | **0.0517** | 0.1024 |
| Random Forest | 0.5987 | -0.0837 | 0.0568 | 0.1054 |
| XGBoost | 0.6364 | 0.0012 | 0.0524 | 0.1029 |

After some minor parameter tuning, I selected a Decision Tree with `max_depth=4` and `min_samples_leaf=300`, since going deeper brings only marginal performance improvement.

## Next steps

- Define custom evaluation metrics based on business requirements.
- Extend the scope: multiplying the predicted default probabilities by the loan amounts, we could estimate the expected loss and try to learn an optimal policy end-to-end.
- Add more features that characterize the users' behaviour, e.g. country and age, and possibly personalized features about previous purchases and payments.
- Test more sophisticated models and tune parameters to improve performance.
