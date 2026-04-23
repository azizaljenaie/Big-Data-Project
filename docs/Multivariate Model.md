# Multivariate Model

## Purpose

This multivariate regression analysis predicts the total cost in USD of a call (`total_cost_usd`) using a short set of defensible call-level predictors. The workflow follows the GSBA 576 supervised learning project instructions by comparing multiple multivariate model families on the same response variable and the same input set.

## Variables Used

- Output variable (`y`): `total_cost_usd`
- Input variables (`x`):
  - `call_duration_seconds`
  - `live_intent`
  - `analysis_sentiment_score`

These predictors were chosen because they match the project framing around operational and conversational call features while avoiding direct price-component variables.

## Data Setup

- Data source: pre-partitioned project CSV files in `Datasets for Big Data`
- Partitions used:
  - training set for model fitting
  - validation set for tuning and model selection
  - test set for the final uncontaminated holdout estimate

## Correlation Check

The numeric correlation check for the final multivariate inputs showed:

- `call_duration_seconds` with `total_cost_usd`: `0.7632`
- `analysis_sentiment_score` with `total_cost_usd`: `0.4498`

`live_intent` is categorical, so it is handled with factor dummy encoding rather than a simple numeric correlation.

## Benchmark Model

A benchmark mean model was used as a baseline. This predicts the average training-set call cost for every observation.

- Benchmark train RMSE: `0.1307`
- Benchmark validation RMSE: `0.1700`

## Models Compared

The following multivariate models were estimated using the same predictor set:

1. Linear regression
2. Ridge regression (untuned and tuned)
3. Support vector regression (untuned and tuned)
4. Regression tree (untuned and tuned)
5. Random forest (untuned and tuned)

## Validation Results

Validation RMSE by model:

- Tuned SVM: `0.0851`
- Tuned Random Forest: `0.1029`
- Untuned Random Forest: `0.1044`
- Linear Regression: `0.1051`
- Tuned Ridge: `0.1058`
- Tuned Tree: `0.1250`
- Untuned Tree: `0.1253`
- Untuned SVM: `0.1279`
- Untuned Ridge: `0.1540`
- Benchmark Mean: `0.1700`

## Final Selected Model

Using the validation partition for model selection, the final chosen multivariate model was:

- `Tuned SVM`

This model gave the lowest validation RMSE among all multivariate candidates.

## Final Test Result

The selected tuned SVM model was then evaluated one time on the uncontaminated holdout testing partition:

- Test RMSE: `0.1311`

For reference:

- Benchmark test RMSE: `0.1599`

## Interpretation

The multivariate analysis shows that the chosen operational and conversational features provide meaningful predictive value beyond the benchmark mean model. The tuned support vector regression model gave the strongest validation performance, which suggests that nonlinear structure in the relationship between call duration, call intent, sentiment score, and total call cost is important.

Although the final test RMSE was higher than the validation RMSE, the tuned SVM still improved on the benchmark holdout error and remained the best selected model under the required train-validation-test workflow.

## Business Takeaway

This multivariate model is more useful than the earlier bivariate baseline because it combines:

- call duration as the main operational volume signal
- live intent as call-type context
- sentiment score as an additional conversational signal

The final model supports the business problem by offering a better basis for forecasting likely call cost and identifying which call patterns are associated with higher spending.
