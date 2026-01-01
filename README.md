# Big-Data-Analytics
ML for E-Commerce Purchase Prediction | Big Data Analytics project at Universität St. Gallen.

This project analyzes large-scale e-commerce behavioral data to predict whether an online user session results in a purchase.
Using event-level logs from a multi-category online retailer, the study focuses on session-level behavioral patterns such as product views, cart activity, prices, and time spent on the platform.
The project implements a full big-data analytical pipeline, from data ingestion and cloud storage to feature engineering, modeling, evaluation, and interpretation, under strict memory and computational constraints.

## Methodology
Transformation of raw event-level logs into session-level features
Out-of-memory data processing using Parquet and Apache Arrow
Feature engineering of behavioral and temporal variables
Predictive modeling under severe class imbalance (~5% purchase sessions)
Two models are employed:
Ridge logistic regression for a regularized, interpretable baseline
XGBoost to capture non-linear effects and feature interactions
Model performance is evaluated using ROC-AUC and F1-score, with additional analysis through feature importance and partial dependence plots.
## Technologies
R, R Markdown, Apache Arrow, Parquet, Google Cloud Storage, biglasso, xgboost, caret, DALEX
Data Source
## The dataset used in this project is publicly available on Kaggle
(E-commerce Behavior Data from Multi-Category Store). Only a subset of the data (November 2019) is used in the analysis.
