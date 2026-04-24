# Classification Model

## Purpose

This classification analysis predicts the sentiment category of a call (`analysis_sentiment_label`). The workflow follows the GSBA 576 classification requirements by estimating binary classification models and additional classification tools using the same pre-partitioned train, validation, and test files.

## Variables Used

- Output variable: `analysis_sentiment_label`
- Binary event modeled for logit/probit: negative sentiment calls
- Input variables:
  - `call_duration_seconds`
  - `live_intent`
  - `termination_reason`
  - `cache_ratio`
  - `has_customer`
  - `hour_of_day`
  - `is_weekend`

`analysis_sentiment_score` is intentionally excluded because it is the numeric sentiment score behind the categorical sentiment label. Including it would make the classification problem circular.

## Data Setup

- Data source: GitHub raw CSV files in `Datasets for Big Data`
- Partitions used:
  - training set for model fitting
  - validation set for tuning and model selection
  - test set for final holdout evaluation

## Models Compared

The classification workflow compares:

1. Majority-class benchmark
2. Logistic regression
3. Probit regression
4. Support vector machine, untuned and tuned
5. Classification tree, untuned and tuned
6. Gradient boosted tree, untuned and tuned

## Model Selection

Because the target is imbalanced, AUC is used as the main model-selection metric when probability estimates are available. Accuracy is still reported because the course instructions ask for comparison with accuracy.

## Test Set Use

The test set is held out until the final section. It is used only once to estimate final performance for the model selected using the validation partition.
