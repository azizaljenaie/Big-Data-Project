############################
##MULTIVARIATE REGRESSION##
############################

#LOAD LIBRARIES
library(tidymodels) #FOR rmse(), decision_tree(), tune_grid(), rand_forest()
library(glmnet) #FOR glmnet() REGULARIZATION
library(e1071) #FOR svm() AND tune.svm()
library(randomForest) #ALTERNATIVE RANDOM FOREST PACKAGE FROM CLASS EXAMPLE
library(rpart.plot) #FOR DISPLAYING THE TREE

###################
#IMPORT THE DATASET#
###################

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
#CLEANING THE VARIABLES FOR MODELING#
####################################

train$total_cost_usd <- clean_numeric_text(train$total_cost_usd)
validation$total_cost_usd <- clean_numeric_text(validation$total_cost_usd)
test$total_cost_usd <- clean_numeric_text(test$total_cost_usd)

train$call_duration_seconds <- clean_numeric_text(train$call_duration_seconds)
validation$call_duration_seconds <- clean_numeric_text(validation$call_duration_seconds)
test$call_duration_seconds <- clean_numeric_text(test$call_duration_seconds)

train$analysis_sentiment_score <- clean_numeric_text(train$analysis_sentiment_score)
validation$analysis_sentiment_score <- clean_numeric_text(validation$analysis_sentiment_score)
test$analysis_sentiment_score <- clean_numeric_text(test$analysis_sentiment_score)

live_intent_levels <- sort(unique(train$live_intent))

train$live_intent <- factor(train$live_intent, levels = live_intent_levels)
validation$live_intent <- factor(validation$live_intent, levels = live_intent_levels)
test$live_intent <- factor(test$live_intent, levels = live_intent_levels)

model_columns <- c(
  "total_cost_usd",
  "call_duration_seconds",
  "live_intent",
  "analysis_sentiment_score"
)

train <- train[complete.cases(train[, model_columns]), ]
validation <- validation[complete.cases(validation[, model_columns]), ]
test <- test[complete.cases(test[, model_columns]), ]

###########################
#MODEL DESCRIPTION / FORMULA#
###########################

fmla <- total_cost_usd ~ call_duration_seconds + live_intent + analysis_sentiment_score

########################################
#STEP 2: CORRELATION CHECK FOR y-VARIABLE#
########################################

#live_intent IS CATEGORICAL, SO THE NUMERIC CORRELATION CHECK
#FOCUSES ON THE TWO NUMERIC INPUTS IN THE FINAL MODEL

CORRELATION_TABLE <- data.frame(
  variable = c("call_duration_seconds", "analysis_sentiment_score"),
  correlation_with_total_cost_usd = c(
    cor(train$total_cost_usd, train$call_duration_seconds, use = "complete.obs"),
    cor(train$total_cost_usd, train$analysis_sentiment_score, use = "complete.obs")
  )
)

CORRELATION_TABLE$abs_correlation <- abs(CORRELATION_TABLE$correlation_with_total_cost_usd)
CORRELATION_TABLE <- CORRELATION_TABLE[order(-CORRELATION_TABLE$abs_correlation), ]
row.names(CORRELATION_TABLE) <- NULL
CORRELATION_TABLE

###########################
#BENCHMARK MEAN PREDICTION#
###########################

BENCHMARK_MEAN <- mean(train$total_cost_usd)

BENCHMARK_PRED_IN <- rep(BENCHMARK_MEAN, nrow(train))
BENCHMARK_PRED_OUT <- rep(BENCHMARK_MEAN, nrow(validation))

PVA_BENCHMARK_IN <- data.frame(prediction = BENCHMARK_PRED_IN, actual = train$total_cost_usd)
PVA_BENCHMARK_OUT <- data.frame(prediction = BENCHMARK_PRED_OUT, actual = validation$total_cost_usd)

(E_IN_BENCHMARK <- as.numeric(rmse(PVA_BENCHMARK_IN, actual, prediction)[3]))
(E_OUT_BENCHMARK <- as.numeric(rmse(PVA_BENCHMARK_OUT, actual, prediction)[3]))

############################################
#4A: LINEAR MULTIVARIATE REGRESSION MODELING#
############################################

LINEAR_MODEL <- lm(fmla, train)
LINEAR_MODEL_SUMMARY <- summary(LINEAR_MODEL)
LINEAR_MODEL_SUMMARY

LINEAR_MODEL_CONFINT <- confint(LINEAR_MODEL)
LINEAR_MODEL_CONFINT

LINEAR_PRED_IN <- predict(LINEAR_MODEL, train)
LINEAR_PRED_OUT <- predict(LINEAR_MODEL, validation)

PVA_LINEAR_IN <- data.frame(prediction = LINEAR_PRED_IN, actual = train$total_cost_usd)
PVA_LINEAR_OUT <- data.frame(prediction = LINEAR_PRED_OUT, actual = validation$total_cost_usd)

(E_IN_LINEAR <- as.numeric(rmse(PVA_LINEAR_IN, actual, prediction)[3]))
(E_OUT_LINEAR <- as.numeric(rmse(PVA_LINEAR_OUT, actual, prediction)[3]))

##############################
#4B: RIDGE REGULARIZATION MODEL#
##############################

#model.matrix() DUMMIES THE CATEGORICAL VARIABLE live_intent
X_train <- model.matrix(fmla, train)[, -1]
X_validation <- model.matrix(fmla, validation)[, -1]
y_train <- train$total_cost_usd

#UNTUNED RIDGE MODEL
RIDGE_UNTUNED <- glmnet(
  x = X_train,
  y = y_train,
  alpha = 0,
  lambda = 1
)

RIDGE_UNTUNED_PRED_IN <- predict(RIDGE_UNTUNED, X_train)
RIDGE_UNTUNED_PRED_OUT <- predict(RIDGE_UNTUNED, X_validation)

PVA_RIDGE_UNTUNED_IN <- data.frame(s0 = RIDGE_UNTUNED_PRED_IN, actual = train$total_cost_usd)
PVA_RIDGE_UNTUNED_OUT <- data.frame(s0 = RIDGE_UNTUNED_PRED_OUT, actual = validation$total_cost_usd)

(E_IN_RIDGE_UNTUNED <- as.numeric(rmse(PVA_RIDGE_UNTUNED_IN, truth = actual, estimate = s0)[3]))
(E_OUT_RIDGE_UNTUNED <- as.numeric(rmse(PVA_RIDGE_UNTUNED_OUT, truth = actual, estimate = s0)[3]))

#TUNED RIDGE MODEL
set.seed(123)
CV_RIDGE <- cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 0,
  nfolds = 3,
  type.measure = "mse"
)

(RIDGE_BEST_LAMBDA <- CV_RIDGE$lambda.min)
(RIDGE_SIMPLE_LAMBDA <- CV_RIDGE$lambda.1se)

RIDGE_TUNED <- glmnet(
  x = X_train,
  y = y_train,
  alpha = 0,
  lambda = RIDGE_BEST_LAMBDA
)

coef(RIDGE_TUNED)

RIDGE_TUNED_PRED_IN <- predict(RIDGE_TUNED, X_train)
RIDGE_TUNED_PRED_OUT <- predict(RIDGE_TUNED, X_validation)

PVA_RIDGE_TUNED_IN <- data.frame(s0 = RIDGE_TUNED_PRED_IN, actual = train$total_cost_usd)
PVA_RIDGE_TUNED_OUT <- data.frame(s0 = RIDGE_TUNED_PRED_OUT, actual = validation$total_cost_usd)

(E_IN_RIDGE_TUNED <- as.numeric(rmse(PVA_RIDGE_TUNED_IN, truth = actual, estimate = s0)[3]))
(E_OUT_RIDGE_TUNED <- as.numeric(rmse(PVA_RIDGE_TUNED_OUT, truth = actual, estimate = s0)[3]))

RIDGE_TABLE <- matrix(
  c(
    E_IN_RIDGE_UNTUNED,
    E_IN_RIDGE_TUNED,
    E_OUT_RIDGE_UNTUNED,
    E_OUT_RIDGE_TUNED
  ),
  ncol = 2,
  byrow = TRUE
)

colnames(RIDGE_TABLE) <- c("UNTUNED", "TUNED")
rownames(RIDGE_TABLE) <- c("E_IN", "E_OUT")
RIDGE_TABLE

#########################
#4C: SUPPORT VECTOR MACHINE#
#########################

#SVMs ONLY WORK WITH NUMERIC INPUT DATA
SVM_X_TRAIN <- X_train
SVM_X_VALIDATION <- X_validation
KERN_TYPE <- "radial"

#UNTUNED SVM
SVM_UNTUNED <- svm(
  x = SVM_X_TRAIN,
  y = y_train,
  type = "eps-regression",
  kernel = KERN_TYPE,
  cost = 1,
  gamma = 1 / ncol(SVM_X_TRAIN),
  scale = TRUE
)

SVM_UNTUNED_PRED_IN <- predict(SVM_UNTUNED, SVM_X_TRAIN)
SVM_UNTUNED_PRED_OUT <- predict(SVM_UNTUNED, SVM_X_VALIDATION)

PVA_SVM_UNTUNED_IN <- data.frame(prediction = SVM_UNTUNED_PRED_IN, actual = train$total_cost_usd)
PVA_SVM_UNTUNED_OUT <- data.frame(prediction = SVM_UNTUNED_PRED_OUT, actual = validation$total_cost_usd)

(E_IN_SVM_UNTUNED <- as.numeric(rmse(PVA_SVM_UNTUNED_IN, actual, prediction)[3]))
(E_OUT_SVM_UNTUNED <- as.numeric(rmse(PVA_SVM_UNTUNED_OUT, actual, prediction)[3]))

#TUNING THE SVM BY CROSS-VALIDATION
set.seed(123)
SVM_TUNE <- e1071::tune.svm(
  x = SVM_X_TRAIN,
  y = y_train,
  type = "eps-regression",
  kernel = KERN_TYPE,
  tunecontrol = tune.control(cross = 3),
  cost = c(0.01, 0.1, 1, 10, 100),
  gamma = c(0.01, 0.1, 0.25, 0.5, 1)
)

SVM_TUNE$best.parameters

SVM_TUNED <- svm(
  x = SVM_X_TRAIN,
  y = y_train,
  type = "eps-regression",
  kernel = KERN_TYPE,
  cost = SVM_TUNE$best.parameters$cost,
  gamma = SVM_TUNE$best.parameters$gamma,
  scale = TRUE
)

SVM_TUNED_PRED_IN <- predict(SVM_TUNED, SVM_X_TRAIN)
SVM_TUNED_PRED_OUT <- predict(SVM_TUNED, SVM_X_VALIDATION)

PVA_SVM_TUNED_IN <- data.frame(prediction = SVM_TUNED_PRED_IN, actual = train$total_cost_usd)
PVA_SVM_TUNED_OUT <- data.frame(prediction = SVM_TUNED_PRED_OUT, actual = validation$total_cost_usd)

(E_IN_SVM_TUNED <- as.numeric(rmse(PVA_SVM_TUNED_IN, actual, prediction)[3]))
(E_OUT_SVM_TUNED <- as.numeric(rmse(PVA_SVM_TUNED_OUT, actual, prediction)[3]))

SVM_TABLE <- matrix(
  c(
    E_IN_SVM_UNTUNED,
    E_IN_SVM_TUNED,
    E_OUT_SVM_UNTUNED,
    E_OUT_SVM_TUNED
  ),
  ncol = 2,
  byrow = TRUE
)

colnames(SVM_TABLE) <- c("UNTUNED", "TUNED")
rownames(SVM_TABLE) <- c("E_IN", "E_OUT")
SVM_TABLE

#####################
#4D: REGRESSION TREE#
#####################

#UNTUNED TREE
TREE_UNTUNED <- decision_tree(
  min_n = 20,
  tree_depth = 30,
  cost_complexity = 0.01
) %>%
  set_engine("rpart") %>%
  set_mode("regression") %>%
  fit(fmla, train)

TREE_UNTUNED_PRED_IN <- predict(TREE_UNTUNED, new_data = train) %>%
  bind_cols(train)

TREE_UNTUNED_PRED_OUT <- predict(TREE_UNTUNED, new_data = validation) %>%
  bind_cols(validation)

(E_IN_TREE_UNTUNED <- as.numeric(rmse(TREE_UNTUNED_PRED_IN, truth = total_cost_usd, estimate = .pred)[3]))
(E_OUT_TREE_UNTUNED <- as.numeric(rmse(TREE_UNTUNED_PRED_OUT, truth = total_cost_usd, estimate = .pred)[3]))

#TUNED TREE
TREE_SPEC <- decision_tree(
  min_n = tune(),
  tree_depth = tune(),
  cost_complexity = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("regression")

TREE_GRID <- grid_regular(
  extract_parameter_set_dials(TREE_SPEC),
  levels = 3
)

set.seed(123)
TREE_TUNE_RESULTS <- tune_grid(
  TREE_SPEC,
  fmla,
  resamples = vfold_cv(train, v = 3),
  grid = TREE_GRID,
  metrics = metric_set(rmse)
)

TREE_SHOW_BEST <- show_best(TREE_TUNE_RESULTS, metric = "rmse")
TREE_SHOW_BEST

BEST_TREE_PARAMS <- select_best(TREE_TUNE_RESULTS, metric = "rmse")
BEST_TREE_PARAMS

TREE_TUNED <- finalize_model(TREE_SPEC, BEST_TREE_PARAMS) %>%
  fit(fmla, train)

TREE_TUNED_PRED_IN <- predict(TREE_TUNED, new_data = train) %>%
  bind_cols(train)

TREE_TUNED_PRED_OUT <- predict(TREE_TUNED, new_data = validation) %>%
  bind_cols(validation)

(E_IN_TREE_TUNED <- as.numeric(rmse(TREE_TUNED_PRED_IN, truth = total_cost_usd, estimate = .pred)[3]))
(E_OUT_TREE_TUNED <- as.numeric(rmse(TREE_TUNED_PRED_OUT, truth = total_cost_usd, estimate = .pred)[3]))

TREE_TABLE <- matrix(
  c(
    E_IN_TREE_UNTUNED,
    E_IN_TREE_TUNED,
    E_OUT_TREE_UNTUNED,
    E_OUT_TREE_TUNED
  ),
  ncol = 2,
  byrow = TRUE
)

colnames(TREE_TABLE) <- c("UNTUNED", "TUNED")
rownames(TREE_TABLE) <- c("E_IN", "E_OUT")
TREE_TABLE

TUNED_TREE_ENGINE <- extract_fit_engine(TREE_TUNED)

if (interactive()) {
  rpart.plot(TUNED_TREE_ENGINE)
}

###########################
#4E: TREE-BASED ENSEMBLE MODEL#
###########################

#UNTUNED RANDOM FOREST
RF_UNTUNED <- rand_forest(
  mtry = 2,
  trees = 100,
  min_n = 5
) %>%
  set_mode("regression") %>%
  set_engine("randomForest") %>%
  fit(fmla, train)

RF_UNTUNED_PRED_IN <- predict(RF_UNTUNED, new_data = train) %>%
  bind_cols(train)

RF_UNTUNED_PRED_OUT <- predict(RF_UNTUNED, new_data = validation) %>%
  bind_cols(validation)

(E_IN_RF_UNTUNED <- as.numeric(rmse(RF_UNTUNED_PRED_IN, truth = total_cost_usd, estimate = .pred)[3]))
(E_OUT_RF_UNTUNED <- as.numeric(rmse(RF_UNTUNED_PRED_OUT, truth = total_cost_usd, estimate = .pred)[3]))

#TUNED RANDOM FOREST
RF_SPEC <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("randomForest")

RF_PARAM_SET <- extract_parameter_set_dials(RF_SPEC)
RF_PARAM_SET <- finalize(
  RF_PARAM_SET,
  train[, c("call_duration_seconds", "live_intent", "analysis_sentiment_score")]
)

RF_PARAM_SET <- update(
  RF_PARAM_SET,
  trees = trees(c(100L, 500L)),
  min_n = min_n(c(2L, 20L))
)

RF_GRID <- grid_regular(RF_PARAM_SET, levels = 3)

set.seed(123)
RF_TUNE_RESULTS <- tune_grid(
  RF_SPEC,
  fmla,
  resamples = vfold_cv(train, v = 3),
  grid = RF_GRID,
  metrics = metric_set(rmse)
)

RF_SHOW_BEST <- show_best(RF_TUNE_RESULTS, metric = "rmse")
RF_SHOW_BEST

BEST_RF_PARAMS <- select_best(RF_TUNE_RESULTS, metric = "rmse")
BEST_RF_PARAMS

RF_TUNED <- finalize_model(RF_SPEC, BEST_RF_PARAMS) %>%
  fit(fmla, train)

RF_TUNED_PRED_IN <- predict(RF_TUNED, new_data = train) %>%
  bind_cols(train)

RF_TUNED_PRED_OUT <- predict(RF_TUNED, new_data = validation) %>%
  bind_cols(validation)

(E_IN_RF_TUNED <- as.numeric(rmse(RF_TUNED_PRED_IN, truth = total_cost_usd, estimate = .pred)[3]))
(E_OUT_RF_TUNED <- as.numeric(rmse(RF_TUNED_PRED_OUT, truth = total_cost_usd, estimate = .pred)[3]))

RF_TABLE <- matrix(
  c(
    E_IN_RF_UNTUNED,
    E_IN_RF_TUNED,
    E_OUT_RF_UNTUNED,
    E_OUT_RF_TUNED
  ),
  ncol = 2,
  byrow = TRUE
)

colnames(RF_TABLE) <- c("UNTUNED", "TUNED")
rownames(RF_TABLE) <- c("E_IN", "E_OUT")
RF_TABLE

########################################
#4F: SUMMARY TABLE AND MODEL SELECTION#
########################################

VAL_TABLE <- data.frame(
  MODEL = c(
    "BENCHMARK",
    "LINEAR",
    "RIDGE_UNTUNED",
    "RIDGE_TUNED",
    "SVM_UNTUNED",
    "SVM_TUNED",
    "TREE_UNTUNED",
    "TREE_TUNED",
    "RF_UNTUNED",
    "RF_TUNED"
  ),
  E_IN = c(
    E_IN_BENCHMARK,
    E_IN_LINEAR,
    E_IN_RIDGE_UNTUNED,
    E_IN_RIDGE_TUNED,
    E_IN_SVM_UNTUNED,
    E_IN_SVM_TUNED,
    E_IN_TREE_UNTUNED,
    E_IN_TREE_TUNED,
    E_IN_RF_UNTUNED,
    E_IN_RF_TUNED
  ),
  E_OUT = c(
    E_OUT_BENCHMARK,
    E_OUT_LINEAR,
    E_OUT_RIDGE_UNTUNED,
    E_OUT_RIDGE_TUNED,
    E_OUT_SVM_UNTUNED,
    E_OUT_SVM_TUNED,
    E_OUT_TREE_UNTUNED,
    E_OUT_TREE_TUNED,
    E_OUT_RF_UNTUNED,
    E_OUT_RF_TUNED
  )
)

VAL_TABLE <- VAL_TABLE[order(VAL_TABLE$E_OUT), ]
row.names(VAL_TABLE) <- NULL
VAL_TABLE

BEST_MODEL_NAME <- VAL_TABLE$MODEL[1]
BEST_MODEL_NAME

BEST_MODEL_VALIDATION_RMSE <- VAL_TABLE$E_OUT[1]
BEST_MODEL_VALIDATION_RMSE

###########################################
#FINAL HOLDOUT TEST ERROR FOR BEST MODEL ONLY#
###########################################

X_test <- model.matrix(fmla, test)[, -1]

if (BEST_MODEL_NAME == "LINEAR") {
  FINAL_TEST_PRED <- predict(LINEAR_MODEL, test)
  FINAL_TEST_PVA <- data.frame(prediction = FINAL_TEST_PRED, actual = test$total_cost_usd)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, actual, prediction)[3])
}

if (BEST_MODEL_NAME == "RIDGE_UNTUNED") {
  FINAL_TEST_PRED <- predict(RIDGE_UNTUNED, X_test)
  FINAL_TEST_PVA <- data.frame(s0 = FINAL_TEST_PRED, actual = test$total_cost_usd)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, truth = actual, estimate = s0)[3])
}

if (BEST_MODEL_NAME == "RIDGE_TUNED") {
  FINAL_TEST_PRED <- predict(RIDGE_TUNED, X_test)
  FINAL_TEST_PVA <- data.frame(s0 = FINAL_TEST_PRED, actual = test$total_cost_usd)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, truth = actual, estimate = s0)[3])
}

if (BEST_MODEL_NAME == "SVM_UNTUNED") {
  FINAL_TEST_PRED <- predict(SVM_UNTUNED, X_test)
  FINAL_TEST_PVA <- data.frame(prediction = FINAL_TEST_PRED, actual = test$total_cost_usd)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, actual, prediction)[3])
}

if (BEST_MODEL_NAME == "SVM_TUNED") {
  FINAL_TEST_PRED <- predict(SVM_TUNED, X_test)
  FINAL_TEST_PVA <- data.frame(prediction = FINAL_TEST_PRED, actual = test$total_cost_usd)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, actual, prediction)[3])
}

if (BEST_MODEL_NAME == "TREE_UNTUNED") {
  FINAL_TEST_PVA <- predict(TREE_UNTUNED, new_data = test) %>%
    bind_cols(test)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, truth = total_cost_usd, estimate = .pred)[3])
}

if (BEST_MODEL_NAME == "TREE_TUNED") {
  FINAL_TEST_PVA <- predict(TREE_TUNED, new_data = test) %>%
    bind_cols(test)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, truth = total_cost_usd, estimate = .pred)[3])
}

if (BEST_MODEL_NAME == "RF_UNTUNED") {
  FINAL_TEST_PVA <- predict(RF_UNTUNED, new_data = test) %>%
    bind_cols(test)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, truth = total_cost_usd, estimate = .pred)[3])
}

if (BEST_MODEL_NAME == "RF_TUNED") {
  FINAL_TEST_PVA <- predict(RF_TUNED, new_data = test) %>%
    bind_cols(test)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, truth = total_cost_usd, estimate = .pred)[3])
}

if (BEST_MODEL_NAME == "BENCHMARK") {
  FINAL_TEST_PRED <- rep(BENCHMARK_MEAN, nrow(test))
  FINAL_TEST_PVA <- data.frame(prediction = FINAL_TEST_PRED, actual = test$total_cost_usd)
  FINAL_TEST_RMSE <- as.numeric(rmse(FINAL_TEST_PVA, actual, prediction)[3])
}

FINAL_TEST_TABLE <- data.frame(
  MODEL = BEST_MODEL_NAME,
  E_TEST = FINAL_TEST_RMSE
)

FINAL_TEST_TABLE
