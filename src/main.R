rm(list = ls())

if (!require("pacman")) install.packages("pacman")
pacman::p_load(xgboost, data.table, randomForest, lightgbm, glmnet, e1071)

source("src/utils.R")
set.seed(123)


main <- function() {
  train <- fread("data/train.csv")[,-"id"]
  target_col <- "Price"
  
  data <- preprocess_data(train)
  data <- simple_impute(data)
  
  splits <- split_data(data)
  train_data <- splits$train
  test_data <- splits$test
  
  model_results <- evaluate_models(train_data, test_data, target_col)
  return(model_results)
}

if (sys.nframe() == 0) {
  main()
}


