################################
##CLASSIFICATION MODELING#######
################################

#LOAD LIBRARIES
library(tidymodels) #FOR accuracy(), roc_auc(), decision_tree(), boost_tree(), tune_grid()
library(e1071) #FOR svm() AND tune.svm()
library(rpart.plot) #FOR DISPLAYING THE TREE
library(xgboost) #FOR GRADIENT BOOSTING

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

  x <- trimws(x)
  x[tolower(x) %in% c("true", "yes")] <- "1"
  x[tolower(x) %in% c("false", "no")] <- "0"
  x <- gsub("$", "", x, fixed = TRUE)
  x <- gsub(",", "", x, fixed = TRUE)
  x <- gsub("%", "", x, fixed = TRUE)
  x <- trimws(x)
  x[x == ""] <- NA

  as.numeric(x)
}

class_results <- function(actual, prediction, pred_negative = NULL) {
  results <- data.frame(
    actual = actual,
    prediction = factor(prediction, levels = levels(actual))
  )

  if (!is.null(pred_negative)) {
    results$pred_negative <- pred_negative
  }

  results
}

class_metrics <- function(results) {
  acc <- as.numeric(accuracy(results, actual, prediction)[3])

  if ("pred_negative" %in% names(results)) {
    auc <- as.numeric(roc_auc(results, actual, pred_negative, event_level = "first")[3])
  } else {
    auc <- NA
  }

  c(accuracy = acc, auc = auc)
}

negative_prob <- function(prediction_object) {
  attr(prediction_object, "probabilities")[, "negative"]
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

#analysis_sentiment_score IS NOT USED BECAUSE IT IS THE NUMERIC SENTIMENT SCORE
#BEHIND THE CATEGORICAL SENTIMENT LABEL WE ARE TRYING TO PREDICT.

classification_columns <- c("analysis_sentiment_label", "call_duration_seconds",
                            "live_intent", "termination_reason", "cache_ratio",
                            "has_customer", "hour_of_day", "is_weekend")

numeric_columns <- c("call_duration_seconds", "cache_ratio", "has_customer",
                     "hour_of_day", "is_weekend")

for (column_name in numeric_columns) {
  train[[column_name]] <- clean_numeric_text(train[[column_name]])
  validation[[column_name]] <- clean_numeric_text(validation[[column_name]])
  test[[column_name]] <- clean_numeric_text(test[[column_name]])
}

cache_median <- median(train$cache_ratio, na.rm = TRUE)
train$cache_ratio[is.na(train$cache_ratio)] <- cache_median
validation$cache_ratio[is.na(validation$cache_ratio)] <- cache_median
test$cache_ratio[is.na(test$cache_ratio)] <- cache_median

live_intent_levels <- sort(unique(train$live_intent))
termination_reason_levels <- sort(unique(train$termination_reason))

train$live_intent <- factor(train$live_intent, levels = live_intent_levels)
validation$live_intent <- factor(validation$live_intent, levels = live_intent_levels)
test$live_intent <- factor(test$live_intent, levels = live_intent_levels)

train$termination_reason <- factor(train$termination_reason, levels = termination_reason_levels)
validation$termination_reason <- factor(validation$termination_reason, levels = termination_reason_levels)
test$termination_reason <- factor(test$termination_reason, levels = termination_reason_levels)

train <- train[complete.cases(train[, classification_columns]), classification_columns]
validation <- validation[complete.cases(validation[, classification_columns]), classification_columns]
test <- test[complete.cases(test[, classification_columns]), classification_columns]

sentiment_levels <- c("negative", "positive")

train$analysis_sentiment_label <- factor(train$analysis_sentiment_label, levels = sentiment_levels)
validation$analysis_sentiment_label <- factor(validation$analysis_sentiment_label, levels = sentiment_levels)
test$analysis_sentiment_label <- factor(test$analysis_sentiment_label, levels = sentiment_levels)

#sentiment_bin IS USED ONLY FOR LOGIT / PROBIT INTERPRETATION:
#1 = NEGATIVE CALL, 0 = POSITIVE CALL
train$sentiment_bin <- ifelse(train$analysis_sentiment_label == "negative", 1, 0)
validation$sentiment_bin <- ifelse(validation$analysis_sentiment_label == "negative", 1, 0)
test$sentiment_bin <- ifelse(test$analysis_sentiment_label == "negative", 1, 0)

train$sentiment_factor <- train$analysis_sentiment_label
validation$sentiment_factor <- validation$analysis_sentiment_label
test$sentiment_factor <- test$analysis_sentiment_label

###########################
#MODEL DESCRIPTION / FORMULA#
###########################

class_fmla <- sentiment_factor ~ call_duration_seconds + live_intent +
  termination_reason + cache_ratio + has_customer + hour_of_day + is_weekend

glm_fmla <- sentiment_bin ~ call_duration_seconds + live_intent +
  termination_reason + cache_ratio + has_customer + hour_of_day + is_weekend

#########################################
#BENCHMARK MODEL: MAJORITY CLASS PREDICTOR#
#########################################

MAJORITY_CLASS <- names(sort(table(train$sentiment_factor), decreasing = TRUE))[1]

BENCHMARK_RESULTS_IN <- class_results(
  train$sentiment_factor,
  rep(MAJORITY_CLASS, nrow(train))
)

BENCHMARK_RESULTS_OUT <- class_results(
  validation$sentiment_factor,
  rep(MAJORITY_CLASS, nrow(validation))
)

BENCHMARK_IN_METRICS <- class_metrics(BENCHMARK_RESULTS_IN)
BENCHMARK_OUT_METRICS <- class_metrics(BENCHMARK_RESULTS_OUT)

BENCHMARK_CONFUSION_IN <- table(BENCHMARK_RESULTS_IN$prediction, BENCHMARK_RESULTS_IN$actual)
BENCHMARK_CONFUSION_OUT <- table(BENCHMARK_RESULTS_OUT$prediction, BENCHMARK_RESULTS_OUT$actual)
BENCHMARK_CONFUSION_IN
BENCHMARK_CONFUSION_OUT

##################################
#1A: LOGISTIC REGRESSION MODELING#
##################################

LOGIT_MODEL <- glm(glm_fmla, data = train, family = binomial(link = "logit"))
summary(LOGIT_MODEL)
confint.default(LOGIT_MODEL)
exp(coef(LOGIT_MODEL))

LOGIT_PROB_IN <- predict(LOGIT_MODEL, train, type = "response")
LOGIT_PROB_OUT <- predict(LOGIT_MODEL, validation, type = "response")

LOGIT_RESULTS_IN <- class_results(train$sentiment_factor,
                                  ifelse(LOGIT_PROB_IN > 0.5, "negative", "positive"),
                                  LOGIT_PROB_IN)

LOGIT_RESULTS_OUT <- class_results(validation$sentiment_factor,
                                   ifelse(LOGIT_PROB_OUT > 0.5, "negative", "positive"),
                                   LOGIT_PROB_OUT)

LOGIT_CONFUSION_IN <- table(LOGIT_RESULTS_IN$prediction, LOGIT_RESULTS_IN$actual)
LOGIT_CONFUSION_OUT <- table(LOGIT_RESULTS_OUT$prediction, LOGIT_RESULTS_OUT$actual)
LOGIT_CONFUSION_IN
LOGIT_CONFUSION_OUT

LOGIT_IN_METRICS <- class_metrics(LOGIT_RESULTS_IN)
LOGIT_OUT_METRICS <- class_metrics(LOGIT_RESULTS_OUT)

###########################
#1B: PROBIT REGRESSION MODEL#
###########################

PROBIT_MODEL <- glm(glm_fmla, data = train, family = binomial(link = "probit"))
summary(PROBIT_MODEL)
confint.default(PROBIT_MODEL)

PROBIT_PROB_IN <- predict(PROBIT_MODEL, train, type = "response")
PROBIT_PROB_OUT <- predict(PROBIT_MODEL, validation, type = "response")

PROBIT_RESULTS_IN <- class_results(train$sentiment_factor,
                                   ifelse(PROBIT_PROB_IN > 0.5, "negative", "positive"),
                                   PROBIT_PROB_IN)

PROBIT_RESULTS_OUT <- class_results(validation$sentiment_factor,
                                    ifelse(PROBIT_PROB_OUT > 0.5, "negative", "positive"),
                                    PROBIT_PROB_OUT)

PROBIT_CONFUSION_IN <- table(PROBIT_RESULTS_IN$prediction, PROBIT_RESULTS_IN$actual)
PROBIT_CONFUSION_OUT <- table(PROBIT_RESULTS_OUT$prediction, PROBIT_RESULTS_OUT$actual)
PROBIT_CONFUSION_IN
PROBIT_CONFUSION_OUT

PROBIT_IN_METRICS <- class_metrics(PROBIT_RESULTS_IN)
PROBIT_OUT_METRICS <- class_metrics(PROBIT_RESULTS_OUT)

LOGIT_PROBIT_TABLE <- matrix(
  c(
    LOGIT_IN_METRICS["accuracy"],
    PROBIT_IN_METRICS["accuracy"],
    LOGIT_OUT_METRICS["accuracy"],
    PROBIT_OUT_METRICS["accuracy"],
    LOGIT_IN_METRICS["auc"],
    PROBIT_IN_METRICS["auc"],
    LOGIT_OUT_METRICS["auc"],
    PROBIT_OUT_METRICS["auc"]
  ),
  ncol = 2,
  byrow = TRUE
)

colnames(LOGIT_PROBIT_TABLE) <- c("LOGIT", "PROBIT")
rownames(LOGIT_PROBIT_TABLE) <- c("ACC_IN", "ACC_OUT", "AUC_IN", "AUC_OUT")
LOGIT_PROBIT_TABLE

#######################################
#2A: SUPPORT VECTOR MACHINE CLASSIFIER#
#######################################

#SVMs ONLY WORK WITH NUMERIC INPUT DATA, SO CATEGORICAL VARIABLES ARE DUMMIED.
SVM_X_TRAIN <- model.matrix(class_fmla, train)[, -1]
SVM_X_VALIDATION <- model.matrix(class_fmla, validation)[, -1]
SVM_Y_TRAIN <- train$sentiment_factor
SVM_Y_VALIDATION <- validation$sentiment_factor

KERN_TYPE <- "radial"

set.seed(123)
SVM_UNTUNED <- svm(
  x = SVM_X_TRAIN,
  y = SVM_Y_TRAIN,
  type = "C-classification",
  kernel = KERN_TYPE,
  cost = 1,
  gamma = 1 / ncol(SVM_X_TRAIN),
  probability = TRUE,
  scale = TRUE
)

SVM_PRED_IN <- predict(SVM_UNTUNED, SVM_X_TRAIN, probability = TRUE)
SVM_PRED_OUT <- predict(SVM_UNTUNED, SVM_X_VALIDATION, probability = TRUE)

SVM_RESULTS_IN <- class_results(SVM_Y_TRAIN, SVM_PRED_IN, negative_prob(SVM_PRED_IN))
SVM_RESULTS_OUT <- class_results(SVM_Y_VALIDATION, SVM_PRED_OUT, negative_prob(SVM_PRED_OUT))

SVM_IN_METRICS <- class_metrics(SVM_RESULTS_IN)
SVM_OUT_METRICS <- class_metrics(SVM_RESULTS_OUT)

tune_control <- tune.control(cross = 3)

set.seed(123)
SVM_TUNE <- tune.svm(x = SVM_X_TRAIN, y = SVM_Y_TRAIN,
                     type = "C-classification", kernel = KERN_TYPE,
                     tunecontrol = tune_control, cost = c(0.1, 1, 10, 100),
                     gamma = c(0.001, 0.01, 0.1),
                     probability = TRUE, scale = TRUE)

print(SVM_TUNE)

set.seed(123)
SVM_TUNED <- svm(
  x = SVM_X_TRAIN,
  y = SVM_Y_TRAIN,
  type = "C-classification",
  kernel = KERN_TYPE,
  cost = SVM_TUNE$best.parameters$cost,
  gamma = SVM_TUNE$best.parameters$gamma,
  probability = TRUE,
  scale = TRUE
)

SVM_TUNED_PRED_IN <- predict(SVM_TUNED, SVM_X_TRAIN, probability = TRUE)
SVM_TUNED_PRED_OUT <- predict(SVM_TUNED, SVM_X_VALIDATION, probability = TRUE)

SVM_TUNED_RESULTS_IN <- class_results(SVM_Y_TRAIN, SVM_TUNED_PRED_IN, negative_prob(SVM_TUNED_PRED_IN))
SVM_TUNED_RESULTS_OUT <- class_results(SVM_Y_VALIDATION, SVM_TUNED_PRED_OUT, negative_prob(SVM_TUNED_PRED_OUT))

SVM_TUNED_IN_METRICS <- class_metrics(SVM_TUNED_RESULTS_IN)
SVM_TUNED_OUT_METRICS <- class_metrics(SVM_TUNED_RESULTS_OUT)

##############################
#2B: CLASSIFICATION TREE MODEL#
##############################

TREE_UNTUNED <- decision_tree(
  min_n = 20,
  tree_depth = 30,
  cost_complexity = 0.01
) %>%
  set_engine("rpart") %>%
  set_mode("classification") %>%
  fit(class_fmla, train)

TREE_PRED_IN <- predict(TREE_UNTUNED, new_data = train, type = "class") %>% bind_cols(train)
TREE_PRED_OUT <- predict(TREE_UNTUNED, new_data = validation, type = "class") %>% bind_cols(validation)
TREE_PROB_IN <- predict(TREE_UNTUNED, new_data = train, type = "prob") %>% bind_cols(train)
TREE_PROB_OUT <- predict(TREE_UNTUNED, new_data = validation, type = "prob") %>% bind_cols(validation)

TREE_RESULTS_IN <- class_results(TREE_PRED_IN$sentiment_factor, TREE_PRED_IN$.pred_class, TREE_PROB_IN$.pred_negative)
TREE_RESULTS_OUT <- class_results(TREE_PRED_OUT$sentiment_factor, TREE_PRED_OUT$.pred_class, TREE_PROB_OUT$.pred_negative)

TREE_IN_METRICS <- class_metrics(TREE_RESULTS_IN)
TREE_OUT_METRICS <- class_metrics(TREE_RESULTS_OUT)

if (interactive()) {
  rpart.plot(extract_fit_engine(TREE_UNTUNED), main = "Classification Tree - Untuned")
}

TREE_SPEC <- decision_tree(
  min_n = tune::tune(),
  tree_depth = tune::tune(),
  cost_complexity = tune::tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

TREE_GRID <- grid_regular(hardhat::extract_parameter_set_dials(TREE_SPEC), levels = 3)

set.seed(123)
TREE_TUNE_RESULTS <- tune_grid(
  TREE_SPEC,
  class_fmla,
  resamples = vfold_cv(train, v = 3, strata = sentiment_factor),
  grid = TREE_GRID,
  metrics = metric_set(roc_auc, accuracy)
)

TREE_BEST_PARAMS <- select_best(TREE_TUNE_RESULTS, metric = "roc_auc")
TREE_FINAL_SPEC <- finalize_model(TREE_SPEC, TREE_BEST_PARAMS)
TREE_TUNED <- TREE_FINAL_SPEC %>% fit(class_fmla, train)

TREE_TUNED_PRED_IN <- predict(TREE_TUNED, new_data = train, type = "class") %>% bind_cols(train)
TREE_TUNED_PRED_OUT <- predict(TREE_TUNED, new_data = validation, type = "class") %>% bind_cols(validation)
TREE_TUNED_PROB_IN <- predict(TREE_TUNED, new_data = train, type = "prob") %>% bind_cols(train)
TREE_TUNED_PROB_OUT <- predict(TREE_TUNED, new_data = validation, type = "prob") %>% bind_cols(validation)

TREE_TUNED_RESULTS_IN <- class_results(TREE_TUNED_PRED_IN$sentiment_factor, TREE_TUNED_PRED_IN$.pred_class, TREE_TUNED_PROB_IN$.pred_negative)
TREE_TUNED_RESULTS_OUT <- class_results(TREE_TUNED_PRED_OUT$sentiment_factor, TREE_TUNED_PRED_OUT$.pred_class, TREE_TUNED_PROB_OUT$.pred_negative)

TREE_TUNED_IN_METRICS <- class_metrics(TREE_TUNED_RESULTS_IN)
TREE_TUNED_OUT_METRICS <- class_metrics(TREE_TUNED_RESULTS_OUT)

if (interactive()) {
  rpart.plot(extract_fit_engine(TREE_TUNED), main = "Classification Tree - Tuned")
}

################################
#2C: GRADIENT BOOSTED TREE MODEL#
################################

BOOST_UNTUNED <- boost_tree(trees = 100) %>%
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  fit(class_fmla, train)

BOOST_PRED_IN <- predict(BOOST_UNTUNED, new_data = train, type = "class") %>% bind_cols(train)
BOOST_PRED_OUT <- predict(BOOST_UNTUNED, new_data = validation, type = "class") %>% bind_cols(validation)
BOOST_PROB_IN <- predict(BOOST_UNTUNED, new_data = train, type = "prob") %>% bind_cols(train)
BOOST_PROB_OUT <- predict(BOOST_UNTUNED, new_data = validation, type = "prob") %>% bind_cols(validation)

BOOST_RESULTS_IN <- class_results(BOOST_PRED_IN$sentiment_factor, BOOST_PRED_IN$.pred_class, BOOST_PROB_IN$.pred_negative)
BOOST_RESULTS_OUT <- class_results(BOOST_PRED_OUT$sentiment_factor, BOOST_PRED_OUT$.pred_class, BOOST_PROB_OUT$.pred_negative)

BOOST_IN_METRICS <- class_metrics(BOOST_RESULTS_IN)
BOOST_OUT_METRICS <- class_metrics(BOOST_RESULTS_OUT)

BOOST_SPEC <- boost_tree(
  min_n = tune::tune(),
  tree_depth = tune::tune(),
  trees = tune::tune(),
  learn_rate = tune::tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

BOOST_GRID <- expand.grid(
  min_n = c(5, 10),
  tree_depth = c(2, 4),
  trees = c(100, 300),
  learn_rate = c(0.01, 0.1)
)

set.seed(123)
BOOST_TUNE_RESULTS <- tune_grid(
  BOOST_SPEC,
  class_fmla,
  resamples = vfold_cv(train, v = 3, strata = sentiment_factor),
  grid = BOOST_GRID,
  metrics = metric_set(roc_auc, accuracy)
)

BOOST_BEST_PARAMS <- select_best(BOOST_TUNE_RESULTS, metric = "roc_auc")
BOOST_FINAL_SPEC <- finalize_model(BOOST_SPEC, BOOST_BEST_PARAMS)
BOOST_TUNED <- BOOST_FINAL_SPEC %>% fit(class_fmla, train)

BOOST_TUNED_PRED_IN <- predict(BOOST_TUNED, new_data = train, type = "class") %>% bind_cols(train)
BOOST_TUNED_PRED_OUT <- predict(BOOST_TUNED, new_data = validation, type = "class") %>% bind_cols(validation)
BOOST_TUNED_PROB_IN <- predict(BOOST_TUNED, new_data = train, type = "prob") %>% bind_cols(train)
BOOST_TUNED_PROB_OUT <- predict(BOOST_TUNED, new_data = validation, type = "prob") %>% bind_cols(validation)

BOOST_TUNED_RESULTS_IN <- class_results(BOOST_TUNED_PRED_IN$sentiment_factor, BOOST_TUNED_PRED_IN$.pred_class, BOOST_TUNED_PROB_IN$.pred_negative)
BOOST_TUNED_RESULTS_OUT <- class_results(BOOST_TUNED_PRED_OUT$sentiment_factor, BOOST_TUNED_PRED_OUT$.pred_class, BOOST_TUNED_PROB_OUT$.pred_negative)

BOOST_TUNED_IN_METRICS <- class_metrics(BOOST_TUNED_RESULTS_IN)
BOOST_TUNED_OUT_METRICS <- class_metrics(BOOST_TUNED_RESULTS_OUT)

###############################################
#CLASSIFICATION MODEL SUMMARY AND MODEL CHOICE#
###############################################

CLASSIFICATION_TABLE <- data.frame(
  model = c("Benchmark Majority", "Logit", "Probit", "SVM Untuned", "SVM Tuned",
            "Tree Untuned", "Tree Tuned", "Boosted Tree Untuned", "Boosted Tree Tuned"),
  train_accuracy = c(BENCHMARK_IN_METRICS["accuracy"], LOGIT_IN_METRICS["accuracy"], PROBIT_IN_METRICS["accuracy"],
                     SVM_IN_METRICS["accuracy"], SVM_TUNED_IN_METRICS["accuracy"], TREE_IN_METRICS["accuracy"],
                     TREE_TUNED_IN_METRICS["accuracy"], BOOST_IN_METRICS["accuracy"], BOOST_TUNED_IN_METRICS["accuracy"]),
  validation_accuracy = c(BENCHMARK_OUT_METRICS["accuracy"], LOGIT_OUT_METRICS["accuracy"], PROBIT_OUT_METRICS["accuracy"],
                          SVM_OUT_METRICS["accuracy"], SVM_TUNED_OUT_METRICS["accuracy"], TREE_OUT_METRICS["accuracy"],
                          TREE_TUNED_OUT_METRICS["accuracy"], BOOST_OUT_METRICS["accuracy"], BOOST_TUNED_OUT_METRICS["accuracy"]),
  train_auc = c(NA, LOGIT_IN_METRICS["auc"], PROBIT_IN_METRICS["auc"], SVM_IN_METRICS["auc"],
                SVM_TUNED_IN_METRICS["auc"], TREE_IN_METRICS["auc"], TREE_TUNED_IN_METRICS["auc"],
                BOOST_IN_METRICS["auc"], BOOST_TUNED_IN_METRICS["auc"]),
  validation_auc = c(NA, LOGIT_OUT_METRICS["auc"], PROBIT_OUT_METRICS["auc"], SVM_OUT_METRICS["auc"],
                     SVM_TUNED_OUT_METRICS["auc"], TREE_OUT_METRICS["auc"], TREE_TUNED_OUT_METRICS["auc"],
                     BOOST_OUT_METRICS["auc"], BOOST_TUNED_OUT_METRICS["auc"])
)

CLASSIFICATION_TABLE <- CLASSIFICATION_TABLE[order(-CLASSIFICATION_TABLE$validation_auc, na.last = TRUE), ]
row.names(CLASSIFICATION_TABLE) <- NULL
CLASSIFICATION_TABLE

BEST_CLASSIFICATION_MODEL <- CLASSIFICATION_TABLE$model[1]
BEST_CLASSIFICATION_MODEL

#############################
#FINAL HOLDOUT TEST EVALUATION#
#############################

if (BEST_CLASSIFICATION_MODEL %in% c("Logit", "Probit")) {
  FINAL_GLM <- if (BEST_CLASSIFICATION_MODEL == "Logit") LOGIT_MODEL else PROBIT_MODEL
  TEST_PROB <- predict(FINAL_GLM, test, type = "response")
  TEST_PRED <- ifelse(TEST_PROB > 0.5, "negative", "positive")
} else if (BEST_CLASSIFICATION_MODEL %in% c("SVM Untuned", "SVM Tuned")) {
  FINAL_SVM <- if (BEST_CLASSIFICATION_MODEL == "SVM Untuned") SVM_UNTUNED else SVM_TUNED
  SVM_X_TEST <- model.matrix(class_fmla, test)[, -1]
  SVM_TEST_PRED <- predict(FINAL_SVM, SVM_X_TEST, probability = TRUE)
  TEST_PRED <- SVM_TEST_PRED
  TEST_PROB <- negative_prob(SVM_TEST_PRED)
} else if (BEST_CLASSIFICATION_MODEL %in% c("Tree Untuned", "Tree Tuned")) {
  FINAL_TREE <- if (BEST_CLASSIFICATION_MODEL == "Tree Untuned") TREE_UNTUNED else TREE_TUNED
  TEST_PRED <- predict(FINAL_TREE, new_data = test, type = "class")$.pred_class
  TEST_PROB <- predict(FINAL_TREE, new_data = test, type = "prob")$.pred_negative
} else {
  FINAL_BOOST <- if (BEST_CLASSIFICATION_MODEL == "Boosted Tree Untuned") BOOST_UNTUNED else BOOST_TUNED
  TEST_PRED <- predict(FINAL_BOOST, new_data = test, type = "class")$.pred_class
  TEST_PROB <- predict(FINAL_BOOST, new_data = test, type = "prob")$.pred_negative
}

TEST_RESULTS <- class_results(test$sentiment_factor, TEST_PRED, TEST_PROB)
TEST_CONFUSION <- table(TEST_RESULTS$prediction, TEST_RESULTS$actual)
TEST_CONFUSION

FINAL_TEST_METRICS <- class_metrics(TEST_RESULTS)
TEST_BENCHMARK_RESULTS <- class_results(test$sentiment_factor, rep(MAJORITY_CLASS, nrow(test)))
TEST_BENCHMARK_METRICS <- class_metrics(TEST_BENCHMARK_RESULTS)

FINAL_TEST_TABLE <- data.frame(
  model = c("Benchmark Majority", BEST_CLASSIFICATION_MODEL),
  test_accuracy = c(unname(TEST_BENCHMARK_METRICS["accuracy"]), unname(FINAL_TEST_METRICS["accuracy"])),
  test_auc = c(NA, unname(FINAL_TEST_METRICS["auc"]))
)

FINAL_TEST_TABLE
