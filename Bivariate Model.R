# =========================================================
# GSBA 576 - Bivariate Regression Modeling
# Output variable: total_cost_usd
# Best single predictor: call_duration_seconds
# =========================================================


# #########################################################
# ## [SETUP] Helper functions
# #########################################################

library(ggplot2)

read_partition <- function(url) {
  data <- read.csv(url, stringsAsFactors = FALSE, check.names = FALSE)
  names(data)[1] <- "call_duration_seconds"
  data
}

clean_numeric_text <- function(x) {
  if (is.numeric(x)) {
    return(x)
  }

  x <- gsub("$", "", x, fixed = TRUE)
  x <- gsub(",", "", x, fixed = TRUE)
  x <- gsub("%", "", x, fixed = TRUE)
  x <- trimws(x)
  x[x == ""] <- NA

  as.numeric(x)
}

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

r_squared <- function(actual, predicted) {
  1 - sum((actual - predicted)^2, na.rm = TRUE) /
    sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
}

get_metrics <- function(actual, predicted) {
  c(
    rmse = rmse(actual, predicted),
    mae = mae(actual, predicted),
    r_squared = r_squared(actual, predicted)
  )
}


# #########################################################
# ## [IMPORT] Read the pre-partitioned datasets from GitHub
# ## Assignment requirement: use train / validation / test
# #########################################################

train_url <- "https://raw.githubusercontent.com/azizaljenaie/Big-Data-Project/main/Datasets%20for%20Big%20Data/Train.csv"
validation_url <- "https://raw.githubusercontent.com/azizaljenaie/Big-Data-Project/main/Datasets%20for%20Big%20Data/Validation.csv"
test_url <- "https://raw.githubusercontent.com/azizaljenaie/Big-Data-Project/main/Datasets%20for%20Big%20Data/Test%20(Hold%20out).csv"

train <- read_partition(train_url)
validation <- read_partition(validation_url)
test <- read_partition(test_url)


# #########################################################
# ## [CLEANING] Convert the variables needed for regression
# #########################################################

y_var <- "total_cost_usd"
x_var <- "call_duration_seconds"

candidate_predictors <- c(
  "call_duration_seconds",
  "tts_characters_count",
  "llm_prompt_tokens",
  "llm_completion_tokens",
  "llm_prompt_cached_tokens",
  "stt_audio_duration_s",
  "hour_of_day",
  "day_of_week",
  "is_weekend",
  "is_orders",
  "has_customer"
)

columns_to_clean <- unique(c(y_var, candidate_predictors))

for (column_name in columns_to_clean) {
  train[[column_name]] <- clean_numeric_text(train[[column_name]])
  validation[[column_name]] <- clean_numeric_text(validation[[column_name]])
  test[[column_name]] <- clean_numeric_text(test[[column_name]])
}


# #########################################################
# ## [STEP 1] Correlation check for the regression target
# ## Assignment requirement: verify a moderate/strong link
# #########################################################

correlation_table <- data.frame(
  variable = candidate_predictors,
  correlation_with_total_cost_usd = sapply(
    candidate_predictors,
    function(column_name) {
      cor(train[[y_var]], train[[column_name]], use = "complete.obs")
    }
  ),
  stringsAsFactors = FALSE
)

correlation_table$abs_correlation <- abs(
  correlation_table$correlation_with_total_cost_usd
)

correlation_table <- correlation_table[
  order(-correlation_table$abs_correlation),
]

row.names(correlation_table) <- NULL


# #########################################################
# ## [STEP 2] Benchmark model
# ## compare error to a baseline
# #########################################################

benchmark_mean <- mean(train[[y_var]])

train_pred_benchmark <- rep(benchmark_mean, nrow(train))
validation_pred_benchmark <- rep(benchmark_mean, nrow(validation))


# #########################################################
# ## [STEP 3A] Simple linear bivariate model
# #########################################################

linear_model <- lm(total_cost_usd ~ call_duration_seconds, data = train)

train_pred_linear <- predict(linear_model, newdata = train)
validation_pred_linear <- predict(linear_model, newdata = validation)

linear_summary <- summary(linear_model)
linear_confint <- confint(linear_model)


# #########################################################
# ## [STEP 3B] Log-transformed bivariate model
# #########################################################

log_model <- lm(log(total_cost_usd) ~ log(call_duration_seconds + 1), data = train)

train_pred_log <- exp(predict(log_model, newdata = train))
validation_pred_log <- exp(predict(log_model, newdata = validation))

log_summary <- summary(log_model)
log_confint <- confint(log_model)


# #########################################################
# ## [STEP 3C] Polynomial bivariate model
# ## Choose degree using validation RMSE
# #########################################################

polynomial_tuning <- data.frame(
  degree = integer(),
  train_rmse = numeric(),
  validation_rmse = numeric(),
  stringsAsFactors = FALSE
)

polynomial_models <- list()

for (degree_value in 2:5) {
  current_formula <- as.formula(
    paste(
      "total_cost_usd ~ poly(call_duration_seconds, degree =",
      degree_value,
      ", raw = TRUE)"
    )
  )

  current_model <- lm(current_formula, data = train)

  current_train_pred <- predict(current_model, newdata = train)
  current_validation_pred <- predict(current_model, newdata = validation)

  polynomial_tuning <- rbind(
    polynomial_tuning,
    data.frame(
      degree = degree_value,
      train_rmse = rmse(train[[y_var]], current_train_pred),
      validation_rmse = rmse(validation[[y_var]], current_validation_pred)
    )
  )

  polynomial_models[[paste0("degree_", degree_value)]] <- current_model
}

best_degree <- polynomial_tuning$degree[
  which.min(polynomial_tuning$validation_rmse)
]

best_polynomial_model <- polynomial_models[[paste0("degree_", best_degree)]]

train_pred_poly <- predict(best_polynomial_model, newdata = train)
validation_pred_poly <- predict(best_polynomial_model, newdata = validation)

best_polynomial_summary <- summary(best_polynomial_model)
best_polynomial_confint <- confint(best_polynomial_model)

x0 <- median(train[[x_var]])
poly_coefficients <- coef(best_polynomial_model)
partial_effect <- 0

for (k in 1:best_degree) {
  partial_effect <- partial_effect + k * poly_coefficients[k + 1] * (x0^(k - 1))
}

polynomial_partial_effect <- data.frame(
  chosen_x_value = x0,
  best_polynomial_degree = best_degree,
  partial_effect_dy_dx_at_x0 = partial_effect
)


# #########################################################
# ## [STEP 3D] Default spline model
# ## No tuning, per assignment instructions
# #########################################################

spline_model <- smooth.spline(
  x = train[[x_var]],
  y = train[[y_var]]
)

train_pred_spline <- predict(spline_model, x = train[[x_var]])$y
validation_pred_spline <- predict(spline_model, x = validation[[x_var]])$y


# #########################################################
# ## [STEP 3E] Compare models on train and validation sets
# #########################################################

performance_table <- data.frame(
  model = c(
    "Benchmark Mean",
    "Linear",
    "Log-Transformed",
    paste("Polynomial Degree", best_degree),
    "Default Spline"
  ),
  train_rmse = c(
    get_metrics(train[[y_var]], train_pred_benchmark)["rmse"],
    get_metrics(train[[y_var]], train_pred_linear)["rmse"],
    get_metrics(train[[y_var]], train_pred_log)["rmse"],
    get_metrics(train[[y_var]], train_pred_poly)["rmse"],
    get_metrics(train[[y_var]], train_pred_spline)["rmse"]
  ),
  validation_rmse = c(
    get_metrics(validation[[y_var]], validation_pred_benchmark)["rmse"],
    get_metrics(validation[[y_var]], validation_pred_linear)["rmse"],
    get_metrics(validation[[y_var]], validation_pred_log)["rmse"],
    get_metrics(validation[[y_var]], validation_pred_poly)["rmse"],
    get_metrics(validation[[y_var]], validation_pred_spline)["rmse"]
  ),
  train_mae = c(
    get_metrics(train[[y_var]], train_pred_benchmark)["mae"],
    get_metrics(train[[y_var]], train_pred_linear)["mae"],
    get_metrics(train[[y_var]], train_pred_log)["mae"],
    get_metrics(train[[y_var]], train_pred_poly)["mae"],
    get_metrics(train[[y_var]], train_pred_spline)["mae"]
  ),
  validation_mae = c(
    get_metrics(validation[[y_var]], validation_pred_benchmark)["mae"],
    get_metrics(validation[[y_var]], validation_pred_linear)["mae"],
    get_metrics(validation[[y_var]], validation_pred_log)["mae"],
    get_metrics(validation[[y_var]], validation_pred_poly)["mae"],
    get_metrics(validation[[y_var]], validation_pred_spline)["mae"]
  ),
  train_r_squared = c(
    get_metrics(train[[y_var]], train_pred_benchmark)["r_squared"],
    get_metrics(train[[y_var]], train_pred_linear)["r_squared"],
    get_metrics(train[[y_var]], train_pred_log)["r_squared"],
    get_metrics(train[[y_var]], train_pred_poly)["r_squared"],
    get_metrics(train[[y_var]], train_pred_spline)["r_squared"]
  ),
  validation_r_squared = c(
    get_metrics(validation[[y_var]], validation_pred_benchmark)["r_squared"],
    get_metrics(validation[[y_var]], validation_pred_linear)["r_squared"],
    get_metrics(validation[[y_var]], validation_pred_log)["r_squared"],
    get_metrics(validation[[y_var]], validation_pred_poly)["r_squared"],
    get_metrics(validation[[y_var]], validation_pred_spline)["r_squared"]
  ),
  stringsAsFactors = FALSE
)


# #########################################################
# ## [STEP 3F] Choose the best model on validation only
# #########################################################

candidate_models <- performance_table[
  performance_table$model != "Benchmark Mean",
]

best_model_name <- candidate_models$model[
  which.min(candidate_models$validation_rmse)
]

predict_final_model <- function(model_name, new_data) {
  if (model_name == "Linear") {
    return(predict(linear_model, newdata = new_data))
  }

  if (model_name == "Log-Transformed") {
    return(exp(predict(log_model, newdata = new_data)))
  }

  if (model_name == paste("Polynomial Degree", best_degree)) {
    return(predict(best_polynomial_model, newdata = new_data))
  }

  if (model_name == "Default Spline") {
    return(predict(spline_model, x = new_data[[x_var]])$y)
  }

  stop("Unknown model name.")
}

final_test_predictions <- predict_final_model(best_model_name, test)

final_test_results <- data.frame(
  final_model = best_model_name,
  test_rmse = rmse(test[[y_var]], final_test_predictions),
  test_mae = mae(test[[y_var]], final_test_predictions),
  test_r_squared = r_squared(test[[y_var]], final_test_predictions)
)

final_test_predictions_table <- data.frame(
  actual_total_cost_usd = test[[y_var]],
  predicted_total_cost_usd = final_test_predictions,
  call_duration_seconds = test[[x_var]]
)


# #########################################################
# ## [STEP 3G] One combined bivariate plot
# #########################################################

plot_bivariate_models <- function() {
  x_grid <- seq(
    min(c(train[[x_var]], validation[[x_var]])),
    max(c(train[[x_var]], validation[[x_var]])),
    length.out = 300
  )

  plot_data <- data.frame(call_duration_seconds = x_grid)

  y_linear_grid <- predict(linear_model, newdata = plot_data)
  y_log_grid <- exp(predict(log_model, newdata = plot_data))
  y_poly_grid <- predict(best_polynomial_model, newdata = plot_data)
  y_spline_grid <- predict(spline_model, x = x_grid)$y

  point_data <- rbind(
    data.frame(
      call_duration_seconds = train[[x_var]],
      total_cost_usd = train[[y_var]],
      partition = "Train Data"
    ),
    data.frame(
      call_duration_seconds = validation[[x_var]],
      total_cost_usd = validation[[y_var]],
      partition = "Validation Data"
    )
  )

  curve_data <- rbind(
    data.frame(
      call_duration_seconds = x_grid,
      total_cost_usd = y_linear_grid,
      model = "Linear"
    ),
    data.frame(
      call_duration_seconds = x_grid,
      total_cost_usd = y_log_grid,
      model = "Log"
    ),
    data.frame(
      call_duration_seconds = x_grid,
      total_cost_usd = y_poly_grid,
      model = paste("Polynomial Degree", best_degree)
    ),
    data.frame(
      call_duration_seconds = x_grid,
      total_cost_usd = y_spline_grid,
      model = "Spline"
    )
  )

  ggplot() +
    geom_point(
      data = point_data,
      aes(
        x = call_duration_seconds,
        y = total_cost_usd,
        shape = partition
      ),
      color = "gray35",
      size = 2.4,
      alpha = 0.9
    ) +
    geom_line(
      data = curve_data,
      aes(
        x = call_duration_seconds,
        y = total_cost_usd,
        color = model
      ),
      linewidth = 1
    ) +
    scale_shape_manual(
      values = c("Train Data" = 16, "Validation Data" = 1)
    ) +
    scale_color_manual(
      values = setNames(
        c("blue", "red", "darkgreen", "purple"),
        c("Linear", "Log", paste("Polynomial Degree", best_degree), "Spline")
      )
    ) +
    labs(
      title = "Bivariate Regression Model Comparison",
      x = "call_duration_seconds",
      y = "total_cost_usd",
      shape = NULL,
      color = NULL
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "left"
    )
}

bivariate_plot <- plot_bivariate_models()

if (interactive()) {
  print(bivariate_plot)
}

bivariate_results <- list(
  assignment_requirements = list(
    uses_prepartitioned_data = TRUE,
    uses_same_y_and_x_for_all_bivariate_models = TRUE,
    validation_used_for_model_selection = TRUE,
    test_used_only_after_final_model_selection = TRUE
  )
  ,
  chosen_variables = list(
    y = y_var,
    x = x_var
  ),
  correlation_table = correlation_table,
  benchmark_mean = benchmark_mean,
  linear_summary = linear_summary,
  linear_confint = linear_confint,
  log_summary = log_summary,
  log_confint = log_confint,
  polynomial_tuning = polynomial_tuning,
  best_polynomial_summary = best_polynomial_summary,
  best_polynomial_confint = best_polynomial_confint,
  polynomial_partial_effect = polynomial_partial_effect,
  performance_table = performance_table,
  best_model_name = best_model_name,
  final_test_results = final_test_results,
  final_test_predictions = final_test_predictions_table,
  bivariate_plot = bivariate_plot
)


# #########################################################
# ## [CONSOLE CHECK] outputs for the report write-up
# #########################################################

print(correlation_table)
print(performance_table)
print(final_test_results)
