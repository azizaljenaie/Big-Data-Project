# =========================================================
# GSBA 576 - Bivariate Regression Modeling
# Output variable: total_cost_usd
# Best single predictor: call_duration_seconds
# =========================================================

##################
##LOAD LIBRARIES##
##################

library(ggplot2)
library(tidymodels) #FOR rmse()

#################
##IMPORT DATASET##
#################

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

train_url <- "https://raw.githubusercontent.com/azizaljenaie/Big-Data-Project/main/Datasets%20for%20Big%20Data/Train.csv"
validation_url <- "https://raw.githubusercontent.com/azizaljenaie/Big-Data-Project/main/Datasets%20for%20Big%20Data/Validation.csv"
test_url <- "https://raw.githubusercontent.com/azizaljenaie/Big-Data-Project/main/Datasets%20for%20Big%20Data/Test%20(Hold%20out).csv"

train <- read_partition(train_url)
validation <- read_partition(validation_url)
test <- read_partition(test_url)

####################################
##CLEANING THE VARIABLES FOR MODELING
####################################

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

#########################################################
##STEP 1: CORRELATION CHECK FOR THE REGRESSION TARGET####
#########################################################

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

###########################
##STEP 2: BENCHMARK MODEL##
###########################

benchmark_mean <- mean(train[[y_var]])

benchmark_pred_in <- rep(benchmark_mean, nrow(train))
benchmark_pred_out <- rep(benchmark_mean, nrow(validation))

pva_benchmark_in <- data.frame(prediction = benchmark_pred_in, actual = train[[y_var]])
pva_benchmark_out <- data.frame(prediction = benchmark_pred_out, actual = validation[[y_var]])

(E_IN_BENCHMARK <- as.numeric(rmse(pva_benchmark_in, actual, prediction)[3]))
(E_OUT_BENCHMARK <- as.numeric(rmse(pva_benchmark_out, actual, prediction)[3]))

##########################################
##STEP 3A: SIMPLE LINEAR BIVARIATE MODEL##
##########################################

linear_model <- lm(total_cost_usd ~ call_duration_seconds, train)
linear_summary <- summary(linear_model)
linear_summary

linear_confint <- confint(linear_model)
linear_confint

linear_pred_in <- predict(linear_model, train)
linear_pred_out <- predict(linear_model, validation)

pva_linear_in <- data.frame(prediction = linear_pred_in, actual = train[[y_var]])
pva_linear_out <- data.frame(prediction = linear_pred_out, actual = validation[[y_var]])

(E_IN_LINEAR <- as.numeric(rmse(pva_linear_in, actual, prediction)[3]))
(E_OUT_LINEAR <- as.numeric(rmse(pva_linear_out, actual, prediction)[3]))

###########################################
##STEP 3B: LOG-TRANSFORMED BIVARIATE MODEL##
###########################################

log_model <- lm(log(total_cost_usd) ~ log(call_duration_seconds + 1), train)
log_summary <- summary(log_model)
log_summary

log_confint <- confint(log_model)
log_confint

log_pred_in <- exp(predict(log_model, train))
log_pred_out <- exp(predict(log_model, validation))

pva_log_in <- data.frame(prediction = log_pred_in, actual = train[[y_var]])
pva_log_out <- data.frame(prediction = log_pred_out, actual = validation[[y_var]])

(E_IN_LOG <- as.numeric(rmse(pva_log_in, actual, prediction)[3]))
(E_OUT_LOG <- as.numeric(rmse(pva_log_out, actual, prediction)[3]))

#########################################
##STEP 3C: POLYNOMIAL BIVARIATE MODELING##
#########################################

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

  current_model <- lm(current_formula, train)

  current_pred_in <- predict(current_model, train)
  current_pred_out <- predict(current_model, validation)

  polynomial_tuning <- rbind(
    polynomial_tuning,
    data.frame(
      degree = degree_value,
      train_rmse = sqrt(mean((train[[y_var]] - current_pred_in)^2)),
      validation_rmse = sqrt(mean((validation[[y_var]] - current_pred_out)^2))
    )
  )

  polynomial_models[[paste0("degree_", degree_value)]] <- current_model
}

best_degree <- polynomial_tuning$degree[
  which.min(polynomial_tuning$validation_rmse)
]

best_polynomial_model <- polynomial_models[[paste0("degree_", best_degree)]]
best_polynomial_summary <- summary(best_polynomial_model)
best_polynomial_summary

best_polynomial_confint <- confint(best_polynomial_model)
best_polynomial_confint

poly_pred_in <- predict(best_polynomial_model, train)
poly_pred_out <- predict(best_polynomial_model, validation)

pva_poly_in <- data.frame(prediction = poly_pred_in, actual = train[[y_var]])
pva_poly_out <- data.frame(prediction = poly_pred_out, actual = validation[[y_var]])

(E_IN_POLY <- as.numeric(rmse(pva_poly_in, actual, prediction)[3]))
(E_OUT_POLY <- as.numeric(rmse(pva_poly_out, actual, prediction)[3]))

#WHAT IS THE PARTIAL EFFECT OF x ON y AT A CHOSEN VALUE OF x?
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

####################################
##STEP 3D: DEFAULT SPLINE MODELING##
####################################

spline_model <- smooth.spline(
  x = train[[x_var]],
  y = train[[y_var]]
)

spline_pred_in <- predict(spline_model, x = train[[x_var]])$y
spline_pred_out <- predict(spline_model, x = validation[[x_var]])$y

pva_spline_in <- data.frame(prediction = spline_pred_in, actual = train[[y_var]])
pva_spline_out <- data.frame(prediction = spline_pred_out, actual = validation[[y_var]])

(E_IN_SPLINE <- as.numeric(rmse(pva_spline_in, actual, prediction)[3]))
(E_OUT_SPLINE <- as.numeric(rmse(pva_spline_out, actual, prediction)[3]))

#######################################################
##STEP 3E: COMPARING MODEL PERFORMANCE ON TRAIN/VALID##
#######################################################

performance_table <- data.frame(
  model = c(
    "Benchmark Mean",
    "Linear",
    "Log-Transformed",
    paste("Polynomial Degree", best_degree),
    "Default Spline"
  ),
  train_rmse = c(
    E_IN_BENCHMARK,
    E_IN_LINEAR,
    E_IN_LOG,
    E_IN_POLY,
    E_IN_SPLINE
  ),
  validation_rmse = c(
    E_OUT_BENCHMARK,
    E_OUT_LINEAR,
    E_OUT_LOG,
    E_OUT_POLY,
    E_OUT_SPLINE
  ),
  train_mae = c(
    mean(abs(train[[y_var]] - benchmark_pred_in)),
    mean(abs(train[[y_var]] - linear_pred_in)),
    mean(abs(train[[y_var]] - log_pred_in)),
    mean(abs(train[[y_var]] - poly_pred_in)),
    mean(abs(train[[y_var]] - spline_pred_in))
  ),
  validation_mae = c(
    mean(abs(validation[[y_var]] - benchmark_pred_out)),
    mean(abs(validation[[y_var]] - linear_pred_out)),
    mean(abs(validation[[y_var]] - log_pred_out)),
    mean(abs(validation[[y_var]] - poly_pred_out)),
    mean(abs(validation[[y_var]] - spline_pred_out))
  ),
  train_r_squared = c(
    1 - sum((train[[y_var]] - benchmark_pred_in)^2) / sum((train[[y_var]] - mean(train[[y_var]]))^2),
    1 - sum((train[[y_var]] - linear_pred_in)^2) / sum((train[[y_var]] - mean(train[[y_var]]))^2),
    1 - sum((train[[y_var]] - log_pred_in)^2) / sum((train[[y_var]] - mean(train[[y_var]]))^2),
    1 - sum((train[[y_var]] - poly_pred_in)^2) / sum((train[[y_var]] - mean(train[[y_var]]))^2),
    1 - sum((train[[y_var]] - spline_pred_in)^2) / sum((train[[y_var]] - mean(train[[y_var]]))^2)
  ),
  validation_r_squared = c(
    1 - sum((validation[[y_var]] - benchmark_pred_out)^2) / sum((validation[[y_var]] - mean(validation[[y_var]]))^2),
    1 - sum((validation[[y_var]] - linear_pred_out)^2) / sum((validation[[y_var]] - mean(validation[[y_var]]))^2),
    1 - sum((validation[[y_var]] - log_pred_out)^2) / sum((validation[[y_var]] - mean(validation[[y_var]]))^2),
    1 - sum((validation[[y_var]] - poly_pred_out)^2) / sum((validation[[y_var]] - mean(validation[[y_var]]))^2),
    1 - sum((validation[[y_var]] - spline_pred_out)^2) / sum((validation[[y_var]] - mean(validation[[y_var]]))^2)
  ),
  stringsAsFactors = FALSE
)

##################################################
##STEP 3F: CHOOSE THE BEST MODEL ON VALIDATION ONLY
##################################################

candidate_models <- performance_table[
  performance_table$model != "Benchmark Mean",
]

best_model_name <- candidate_models$model[
  which.min(candidate_models$validation_rmse)
]

if (best_model_name == "Linear") {
  final_test_predictions <- predict(linear_model, test)
}

if (best_model_name == "Log-Transformed") {
  final_test_predictions <- exp(predict(log_model, test))
}

if (best_model_name == paste("Polynomial Degree", best_degree)) {
  final_test_predictions <- predict(best_polynomial_model, test)
}

if (best_model_name == "Default Spline") {
  final_test_predictions <- predict(spline_model, x = test[[x_var]])$y
}

final_test_results <- data.frame(
  final_model = best_model_name,
  test_rmse = sqrt(mean((test[[y_var]] - final_test_predictions)^2)),
  test_mae = mean(abs(test[[y_var]] - final_test_predictions)),
  test_r_squared = 1 - sum((test[[y_var]] - final_test_predictions)^2) /
    sum((test[[y_var]] - mean(test[[y_var]]))^2)
)

final_test_predictions_table <- data.frame(
  actual_total_cost_usd = test[[y_var]],
  predicted_total_cost_usd = final_test_predictions,
  call_duration_seconds = test[[x_var]]
)

#############################################
##STEP 3G: ONE COMBINED BIVARIATE COMPARISON##
#############################################

x_grid <- seq(
  min(c(train[[x_var]], validation[[x_var]])),
  max(c(train[[x_var]], validation[[x_var]])),
  length.out = 300
)

plot_data <- data.frame(call_duration_seconds = x_grid)

y_linear_grid <- predict(linear_model, plot_data)
y_log_grid <- exp(predict(log_model, plot_data))
y_poly_grid <- predict(best_polynomial_model, plot_data)
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

bivariate_plot <- ggplot() +
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

if (interactive()) {
  print(bivariate_plot)
}

bivariate_results <- list(
  assignment_requirements = list(
    uses_prepartitioned_data = TRUE,
    uses_same_y_and_x_for_all_bivariate_models = TRUE,
    validation_used_for_model_selection = TRUE,
    test_used_only_after_final_model_selection = TRUE
  ),
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

#########################################################
##[CONSOLE CHECK] OUTPUTS FOR THE REPORT WRITE-UP########
#########################################################

print(correlation_table)
print(performance_table)
print(final_test_results)
