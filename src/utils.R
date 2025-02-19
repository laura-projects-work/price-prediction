#' Cleans the data by renaming columns, removing duplicates, converting empty strings
#' to NA, and transforming character columns into factors.
#'
#' @param dt A data.table to preprocess.
#' @return A cleaned data.table.
preprocess_data <- function(dt) {
  dt <- copy(dt)
  
  setnames(dt, 
           c("Laptop Compartment", "Weight Capacity (kg)"),
           c("Laptop_Compartment", "Weight"))
  
  dt <- unique(dt)
  
  char_cols <- names(dt)[sapply(dt, is.character)]
  for (col in char_cols) {
    x <- dt[[col]]
    x[x == ""] <- NA_character_
    set(dt, j = col, value = factor(x))
  }
  
  if ("Compartments" %in% names(dt)) {
    set(dt, j = "Compartments", 
        value = factor(dt$Compartments, levels = sort(unique(dt$Compartments))))
  }
  
  dt
}


#' Replaces missing numeric values with the median and missing categorical values
#' with the mode.
#'
#' @param dt A data.table with missing values.
#' @return A data.table with imputed values.
simple_impute <- function(dt) {
  dt <- copy(dt)
  
  num_cols <- names(dt)[sapply(dt, is.numeric)]
  for (col in num_cols) {
    med_val <- median(dt[[col]], na.rm = TRUE)
    missing_idx <- which(is.na(dt[[col]]))
    if (length(missing_idx)) set(dt, i = missing_idx, j = col, value = med_val)
  }
  
  cat_cols <- names(dt)[sapply(dt, function(x) is.factor(x) || is.character(x))]
  for (col in cat_cols) {
    tab <- table(dt[[col]])
    mode_val <- names(tab)[which.max(tab)]
    missing_idx <- which(is.na(dt[[col]]))
    if (length(missing_idx)) set(dt, i = missing_idx, j = col, value = mode_val)
  }
  
  dt
}


#' Splits the input data into training and test sets based on the specified fraction.
#'
#' @param data A data.table to split.
#' @param train_frac The fraction of data to use for training (default is 0.8).
#' @return A list with components \code{train} and \code{test}.
split_data <- function(data, train_frac = 0.8) {
  train_indices <- sample(
    seq_len(nrow(data)), 
    size = floor(train_frac * nrow(data)))
  list(train = data[train_indices, ], test = data[-train_indices, ])
}


#' Constructs a model matrix for predictors and extracts the target variable.
#'
#' @param data A data.table containing features and the target.
#' @param target_col The name of the target variable.
#' @param scale_numeric Logical indicating whether to scale the predictors (default is TRUE).
#' @return A list with elements \code{X} (feature matrix) and \code{y} (target vector).
extract_features <- function(data, target_col, scale_numeric = TRUE) {
  fml <- as.formula(paste(target_col, "~ . - 1"))
  X <- model.matrix(fml, data = data)
  
  if (scale_numeric) {
    X <- scale(X)
  }
  
  list(X = X, y = data[[target_col]])
}


# ================== 2. MODEL TRAINING FUNCTIONS ==================

#' Trains an XGBoost model using CV to determine the optimal number of rounds.
#'
#' @param X A feature matrix.
#' @param y A target vector.
#' @param nrounds Default number of rounds (overridden by cross-validation).
#' @return A trained XGBoost model.
train_xgboost <- function(X, y, nrounds = 100) {
  dtrain <- xgb.DMatrix(data = X, label = y)
  params <- list(
    objective = "reg:squarederror", 
    eval_metric = "rmse", 
    max_depth = 6, 
    eta = 0.1
  )
  
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1000,                
    nfold = 5,                     
    early_stopping_rounds = 10,
    maximize = FALSE,
    verbose = 0
  )
  
  best_nrounds <- cv_results$best_iteration
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    watchlist = list(train = dtrain),
    print_every_n = 0,
    verbose = 0
  )
  
  return(model)
}


#' Performs a random search over hyperparameters for XGBoost using cross-validation,
#' then trains a final model with the best configuration.
#'
#' @param X A feature matrix.
#' @param y A target vector.
#' @param n_iter Number of random search iterations.
#' @param extra_params A list of additional parameters.
#' @param nrounds Maximum number of boosting rounds.
#' @param early_stopping_rounds Number of rounds for early stopping.
#' @param nfold Number of cross-validation folds.
#' @param verbose Verbosity level.
#' @return A trained XGBoost model.
train_xgboost_random_search <- function(
    X, y,
    n_iter = 20, 
    extra_params = list(), 
    nrounds = 2000, 
    early_stopping_rounds = 20, 
    nfold = 5, 
    verbose = 0) {
  dtrain <- xgb.DMatrix(data = X, label = y)
  
  best_rmse <- Inf
  best_params <- NULL
  best_nrounds <- NA
  
  candidate_max_depth <- c(4, 6, 8, 10)
  candidate_eta <- c(0.01, 0.05, 0.1, 0.2)
  candidate_subsample <- c(0.6, 0.8, 1.0)
  candidate_colsample_bytree <- c(0.6, 0.8, 1.0)
  candidate_min_child_weight <- c(1, 3, 5, 7)
  candidate_gamma <- c(0, 0.1, 0.5, 1)
  
  for (i in seq_len(n_iter)) {
    params_candidate <- list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = sample(candidate_max_depth, 1),
      eta = sample(candidate_eta, 1),
      subsample = sample(candidate_subsample, 1),
      colsample_bytree = sample(candidate_colsample_bytree, 1),
      min_child_weight = sample(candidate_min_child_weight, 1),
      gamma = sample(candidate_gamma, 1)
    )
    
    params_candidate <- modifyList(params_candidate, extra_params)
    
    cv_results <- xgb.cv(
      params = params_candidate,
      data = dtrain,
      nrounds = nrounds,
      nfold = nfold,
      early_stopping_rounds = early_stopping_rounds,
      maximize = FALSE,
      verbose = verbose
    )
    
    current_rmse <- min(cv_results$evaluation_log$test_rmse_mean)
    current_nrounds <- cv_results$best_iteration
    
    if (current_rmse < best_rmse) {
      best_rmse <- current_rmse
      best_params <- params_candidate
      best_nrounds <- current_nrounds
    }
  }
  
  final_model <- xgb.train(
    params = best_params,
    data = dtrain,
    nrounds = best_nrounds,
    watchlist = list(train = dtrain),
    print_every_n = 0,
    verbose = verbose
  )
  
  return(final_model)
}


#' Trains a Random Forest model using the randomForest package.
#'
#' @param train_data A data.table with training data.
#' @param target_col The name of the target variable.
#' @param ntree Number of trees (default is 500).
#' @return A trained Random Forest model.
train_random_forest <- function(train_data, target_col, ntree = 500) {
  formula <- as.formula(paste(target_col, "~ ."))
  model <- randomForest(formula, data = train_data, ntree = ntree)
  return(model)
}


#' Trains a LightGBM model given a feature matrix and target vector.
#'
#' @param X A feature matrix.
#' @param y A target vector.
#' @param num_rounds Number of boosting rounds.
#' @return A trained LightGBM model.
train_lightgbm <- function(X, y, num_rounds = 100) {
  dtrain <- lgb.Dataset(data = X, label = y)
  params <- list(
    objective = "regression", 
    metric = "rmse", 
    learning_rate = 0.1, 
    num_leaves = 31,
    verbose = -1
  )
  model <- lgb.train(params, data = dtrain, nrounds = num_rounds)
  return(model)
}


#' Trains an Elastic Net model using cross-validation (cv.glmnet).
#'
#' @param X A feature matrix.
#' @param y A target vector.
#' @param alpha Mixing parameter between ridge (0) and lasso (1) regression.
#' @return A trained Elastic Net model.
train_elastic_net <- function(X, y, alpha = 0.5) {
  model <- cv.glmnet(X, y, alpha = alpha, nfolds = 10)
  return(model)
}


#' Trains an SVR model using the e1071 package.
#'
#' @param X A feature matrix.
#' @param y A target vector.
#' @return A trained SVR model.
train_svr <- function(X, y) {
  model <- svm(X, y, kernel = "radial")
  return(model)
}


# ================== 3. MODEL DICTIONARY ==================

#' A named list mapping model names to their corresponding training functions.
#'
#' @return A list of model training functions.
model_trainers <- list(
  "XGBoost" = train_xgboost,
  "XGBoost_grid" = train_xgboost_random_search,
  "RandomForest" = train_random_forest,
  "LightGBM" = train_lightgbm,
  "ElasticNet" = train_elastic_net,
  "SVR" = train_svr
)


# ================== 4. MODEL EVALUATION FUNCTIONS ==================

#' Compute Root Mean Squared Error (RMSE)
#'
#' @param actual A numeric vector of actual values.
#' @param predicted A numeric vector of predicted values.
#' @return The RMSE.
compute_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}


#' Compute Adjusted R-squared
#'
#' @param actual A numeric vector of actual values.
#' @param predicted A numeric vector of predicted values.
#' @param p Number of predictors.
#' @return The adjusted R-squared.
compute_adjusted_r2 <- function(actual, predicted, p) {
  SS_res <- sum((actual - predicted)^2)
  SS_tot <- sum((actual - mean(actual))^2)
  R2 <- 1 - (SS_res / SS_tot)
  n <- length(actual)
  1 - (1 - R2) * ((n - 1) / (n - p - 1))
}


#' Evaluate Models on Test Data
#'
#' @param train_data A data.table with training data.
#' @param test_data A data.table with test data.
#' @param target_col The name of the target variable.
#' @return A data.table summarizing model performance (Adjusted R-squared and RMSE).
evaluate_models <- function(train_data, test_data, target_col) {
  # Extract features for models that do not require scaling
  train_feats_unscaled <- extract_features(train_data, target_col, scale_numeric = FALSE)
  test_feats_unscaled <- extract_features(test_data, target_col, scale_numeric = FALSE)
  
  # Extract features for models that require scaling
  train_feats_scaled <- extract_features(train_data, target_col, scale_numeric = TRUE)
  test_feats_scaled <- extract_features(test_data, target_col, scale_numeric = TRUE)
  
  p <- ncol(train_feats_unscaled$X)
  
  results <- data.table(Model = character(), Adjusted_R2 = numeric(), RMSE = numeric())
  
  for (model_name in names(model_trainers)) {
    if (model_name == "RandomForest") {
      model <- train_random_forest(train_data, target_col)
      preds <- predict(model, test_data)
    } else if (model_name == "Baseline") {
      baseline_pred <- mean(train_data[[target_col]])
      preds <- rep(baseline_pred, nrow(test_data))
    } else if (model_name %in% c("ElasticNet", "SVR")) {
      model <- model_trainers[[model_name]](train_feats_scaled$X, train_feats_scaled$y)
      if (model_name == "ElasticNet") {
        preds <- predict(model, newx = test_feats_scaled$X)
      } else {
        preds <- predict(model, test_feats_scaled$X)
      }
    } else if (model_name %in% c("XGBoost", "LightGBM")) {
      model <- model_trainers[[model_name]](train_feats_unscaled$X, train_feats_unscaled$y)
      if (model_name == "XGBoost") {
        preds <- predict(model, newdata = test_feats_unscaled$X)
      } else {
        preds <- predict(model, test_feats_unscaled$X)
      }
    }
    
    R2_adj <- compute_adjusted_r2(test_feats_unscaled$y, preds, p)
    rmse   <- compute_rmse(test_feats_unscaled$y, preds)
    
    results <- rbind(results, data.table(
      Model = model_name,
      Adjusted_R2 = round(R2_adj, 3),
      RMSE = round(rmse, 3)
    ))
  }
  
  results[order(RMSE)]
}
