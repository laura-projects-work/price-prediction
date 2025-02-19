This repository demonstrates a complete machine-learning pipeline in R using a synthetic dataset from Kaggle's Backpack Prediction Challenge (see www.kaggle.com/competitions/playground-series-s5e2/overview/$citation)..

The dataset includes features such as Brand, Material, Size, and others, along with a target variable, Price. Importantly, Price is generated independently of the predictors, so the signal-to-noise ratio is effectively zero. Consequently, advanced models like XGBoost, Random Forest, LightGBM, Elastic Net, and SVR are expected to perform similarly to a simple mean predictor.

This project aims to test the hypothesis that Price has no meaningful relationship with the other variables. We evaluate this by comparing the performance of several advanced models against a baseline mean predictor. If our hypothesis is correct, all models will yield similar—and likely poor—predictive performance.
