# Bivariate Model

## Purpose

This bivariate regression analysis tests whether one operational call feature can help predict the total cost in USD of a call (`total_cost_usd`). The analysis follows the project requirement to compare several bivariate model forms using the same response variable and the same single predictor.

## Variables Used

- Output variable (`y`): `total_cost_usd`
- Input variable (`x`): `call_duration_seconds`

`call_duration_seconds` was chosen as the bivariate predictor because it is a more defensible operational feature than direct cost components and provides a strong positive correlation with total call cost.

## Data Setup

- Data source: pre-partitioned project CSV files in `Datasets for Big Data`
- Partitions used:
  - training set for fitting models
  - validation set for model comparison and selection
  - test set for final uncontaminated evaluation

## Correlation Check

The correlation between `call_duration_seconds` and `total_cost_usd` in the training data is:

- `0.7632`

This indicates a strong positive relationship and supports using call duration as the single predictor in the bivariate modeling section.

## Benchmark Model

A benchmark mean model was included to provide context for prediction error. This benchmark predicts the average training-set call cost for every observation.

- Benchmark mean cost: `0.08964956`

The benchmark helps answer whether using call duration is better than simply predicting average cost every time.

## Bivariate Models Compared

The following bivariate models were estimated:

1. Simple linear model
2. Log-transformed model
3. Polynomial model with degree selected by validation RMSE
4. Default spline model

## Polynomial Tuning Result

Polynomial degrees 2 through 5 were compared using validation RMSE.

- Best polynomial degree: `2`
- Partial effect at `x = 70` seconds: `0.0009541643`

This means that at a call duration of about 70 seconds, one additional second of call duration is associated with roughly `$0.000954` more predicted total cost in the quadratic model.

## Performance Summary

Validation-set performance:

- Linear RMSE: `0.1133905`
- Log-Transformed RMSE: `0.1115473`
- Polynomial Degree 2 RMSE: `0.1134051`
- Default Spline RMSE: `8.5748424`

The log-transformed model performed best on the validation set because it had the lowest validation RMSE, the lowest validation MAE, and the highest validation R-squared among the candidate models.

## Final Selected Model

Using the validation partition for model selection, the final chosen bivariate model was:

- `Log-Transformed`

## Final Test Results

The final chosen model was then evaluated on the uncontaminated test set:

- Test RMSE: `0.5885632`
- Test MAE: `0.1129092`
- Test R-squared: `-12.76977`

## Interpretation

Overall, the bivariate analysis shows that call duration is a statistically and operationally meaningful predictor of total call cost. Longer calls are generally associated with higher total cost, and this relationship is strong enough to outperform the benchmark mean model on the validation set.

Among the candidate models, the log-transformed specification generalized best on the validation data, suggesting that the relationship between duration and cost is not perfectly linear. However, the final test-set performance dropped sharply, which indicates that call duration alone is not a reliable standalone cost model for unseen data.

## Business Takeaway

This bivariate model is useful as an interpretable baseline:

- it confirms that overall call duration is meaningfully associated with total cost
- it provides a simple first-pass operational view of cost behavior
- it shows that a single-variable model is not enough for dependable business forecasting

The weak final test performance suggests that stronger business value will come from the multivariate regression section, where additional operational and conversational predictors can be included.
