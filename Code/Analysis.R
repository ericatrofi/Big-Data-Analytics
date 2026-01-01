
library(knitr)
library(pryr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(corrplot)
library(patchwork)
library(scales)
setwd("~/Library/CloudStorage/OneDrive-USI/ERASMUS/COURSES/BIG DATA/PROJECT")
load('workspace.RData')
# -------------------------------------
# Data exploration of session_features
# -------------------------------------
# Check 
head(session_features)
# data size
print(object.size(session_features), units = "MB" )
# data properties
str(session_features)
# Check for missing values
is.na(session_features) %>% sum()

# summary table
library(gridExtra)
sf_summary <- session_features %>%
  summarise(across(everything(), summary)) %>%
  t() %>% 
  as.data.frame()
colnames(sf_summary) <- c("Min", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max")

kable(sf_summary)


rm(sf_summary)
gc()
# -------------------------------------
# Visualizations
# -------------------------------------

# correlation matrix of numeric features
cor_matrix <- cor(session_features, use = "pairwise.complete.obs")

# Plot and save correlation matrix
png("correlation_matrix.png", width = 800, height = 600)
corrplot(cor_matrix, method = "color", type = "lower", 
         col = colorRampPalette(c("blue", "white", "red"))(200),
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlation Matrix of Session Features",
         font = 2,
         diag = FALSE, 
          mar = c(0,0,1,0))
dev.off()
# read excel f


# count proportions of the target variable
class_count <- session_features %>%
  count(n_purchase) %>%
  mutate(percentage = n / sum(n) * 100) 

ggplot(data = as.data.frame(class_count), aes(x = n_purchase, y = percentage, fill = factor(n_purchase))) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("#ff748b", "lightgreen")) +
  labs(x = "Class", y = "Percentage", title = "Purchase Class Distribution") +
  theme_bw() + 
  guides(fill=guide_legend(title="Purchase")) +
      geom_text(aes(label = paste0(round(percentage, 2), "%")), vjust = -0.5)
ggsave("barplot_purchase.png", plot = last_plot(), width = 6, height = 6)

# basic data exploration of the variables
numeric_features <- session_features %>%
                    select(n_view, n_cart, avg_price, n_unique_categories, session_duration)

# log scale histogram to plot distributions
numeric_long <- numeric_features %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value")

hist <- list()
for (feat in unique(numeric_long$feature)) {
  p <- ggplot(numeric_long %>% filter(feature == feat), aes(x = value   )) +
    geom_histogram(bins = 50, fill = "lightblue", color = "black") +
    scale_x_log10(labels = comma) +
    labs(title = paste( feat  ), x = feat, y = "Count") +
    theme_minimal()
  hist[[feat]] <- p
}

# Combine histograms into a grid
plot <- wrap_plots(hist, ncol = 2)
ggsave("histograms_features.png", plot = plot, width = 10, height = 8)

# remove objects not needed anymore to free memory
rm(cor_matrix, class_count)
gc()
# -------------------------------------
# RIDGE REGRESSION MODELING
# -------------------------------------
# set the weighted Ridge Regression with cross validation
library(caret)
library(glmnet)
library(biglasso)
library(Matrix)
set.seed(42)

# Split data into training and testing sets
train_idx <- createDataPartition(session_features$n_purchase, p = 0.6, list = FALSE)
train <- session_features[train_idx, ]
test  <- session_features[-train_idx, ]

#sparse matrix for biglasso
predictors <- c("n_view", "n_cart", "avg_price", "n_unique_categories", "session_duration", 'is_weekend', 'is_beginning_of_month')
X_train <- as.matrix(train[, predictors])
X_train_bm <- as.big.matrix(X_train, type = "double")

y_train <- train$n_purchase

X_train_bm <- filebacked.big.matrix(
  nrow = nrow(train),
  ncol = length(predictors),
  type = "double",
  backingfile = "X_train.bin",
  descriptorfile = "X_train.desc",
  init = 0
)

X_train_bm[,] <- data.matrix(train[, predictors])

# Cross validation to find optimal lambda with proper timing and memory tracking
start_time <- Sys.time() # to assess time
start_mem <- mem_used() # to assess memory

# Cross validation 
mem_increase <- mem_change(
cvfit2 <- cv.biglasso(
  X = X_train_bm,
  y = y_train,
  alpha = 0,            # ridge
  nfolds = 4,           # 4-fold CV
  family = "binomial" ,  # logistic regression
  nlambda = 50,
  trace = TRUE
))

end_time <- Sys.time()
end_mem <- mem_used()

ridge_timing <- end_time - start_time
ridge_mem_diff <- end_mem - start_mem

# Plot cross-validated deviance
plot(cvfit2)
# Coefficients at best lambda
print(coef(cvfit2, s = "lambda.min"))

# MEMORY ANALYSIS ------------------------
# Check memory size of the model
print(object.size(cvfit2), units = 'MB')
# Time taken for training
print(ridge_timing)
# Memory used during training
print(ridge_mem_diff, units = 'MB')
# check memory used so far
mem_used()

# PREDICTION AND EVALUATION--------------------
library(pROC)
# Test set
X_test <- as.matrix(test[, c("n_view", "n_cart", "avg_price", "n_unique_categories", "session_duration", 'is_weekend', 'is_beginning_of_month')])
X_test_bm <- as.big.matrix(X_test, type = "double")
y_test <- test$n_purchase

# Predict probabilities
ridge_pred <- predict(
  cvfit2,
  X_test_bm,
  type = "response",
  s = "lambda.min"
)
summary(ridge_pred)

# ROC-AUC on full dataset
ridge_roc_auc <- auc(roc(y_test, as.vector(ridge_pred)))
print(paste("ROC-AUC:", round(ridge_roc_auc, 4)))

rm(X_train_bm, X_test_bm)
gc() 

# -------------------------------------
# XGBOOST MODELING
# -------------------------------------
library(xgboost)
library(DALEX)
set.seed(123)
# ---------------------------
#Handle class imbalance
# ---------------------------
n_pos <- sum(y_train == 1)
n_neg <- sum(y_train == 0)
scale_pos_weight <- n_neg / n_pos

# Prepare data matrices
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 5,
  eta              = 0.1,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = scale_pos_weight,
  tree_method     = "hist"
)

# ---------------------------
# Train XGBoost Model with timing and memory tracking
# ---------------------------
cat("=== XGBOOST TRAINING ===\n")

start_time <- Sys.time() # to assess time
start_mem <- mem_used()

mem_change(
xgb_model <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = 300, #upper bound
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20, # stops when AUC does not improve
  print_every_n = 20
)
)
end_time <- Sys.time()
end_mem <- mem_used()

xgb_timing <- end_time - start_time
xgb_mem_diff <- end_mem - start_mem

# --------------------------
# Evaluation
# --------------------------
# ROC AUC
pred_xgb <- predict(xgb_model, dtest)

roc_xgb <- roc(y_test, pred_xgb)
auc_xgb <- auc(roc_xgb)
auc_xgb


#-------------------------
# Feature importance 
#-------------------------
imp <- xgb.importance( model = xgb_model)
head(imp)

ggplot(imp %>% slice_max(Gain, n = 10) %>% arrange(Gain), 
       aes(x = reorder(Feature, Gain), y = Gain, fill = Gain)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  geom_text(aes(label = sprintf("%.2f", Gain)), hjust = -0.1, size = 3) +
  labs(title = "Top 10 features by total Gain",
       x = NULL, y = "Gain") +
  theme_minimal()

ggsave("xgb_feature_importance.png", plot = last_plot(), width = 11, height = 6)
#--------------------------
# Partial Dependence Plot (PDP) for n_cart
#--------------------------
X_sample <- X_train[sample(1:nrow(X_train), 5000), ]
Y_sample <- y_train[sample(1:length(y_train), 5000)]


explainer <- explain(
  model            = xgb_model,
  data             = X_sample,
  y                = Y_sample,
  predict_function = predict,
  label            = "xgboost_purchase"
)

pdp_ncart <- model_profile(
  explainer,
  variables = "n_cart",
  variable_type = "numerical",
  grid_points = 11 
)

pdp_table <- pdp_ncart$agr_profiles
head(pdp_table)

ggplot(pdp_table, aes(x = `_x_`, y = `_yhat_`)) +
  geom_line( col = "blue") +
  theme_bw() +
  labs(
    title = "Partial Dependence Plot for n_cart",
    x = "n_cart",
    y = "Predicted Probability of Purchase"
  )
ggsave("xgb_pdp_ncart.png", plot = last_plot(), width = 8, height = 6)

# clean up large objects
rm(roc_obj, dtrain, dtest, explainer, X_sample, Y_sample, pdp_ncart, pdp_table, imp, params)
gc()
mem_used()
# -------------------------------------
# COMPARISON 
# -------------------------------------
# -----------------------
# PERFORMANCE COMPARISON
# -----------------------

# Initialize results storage
comparison_results <- data.frame(
  Model = character(),
  Training_Time_Seconds = numeric(),
  Memory_Usage_MB = numeric(),
  AUC = numeric()
)
# store Ridge results
comparison_results <- rbind(comparison_results, data.frame(
  Model = "Ridge Regression",
  Training_Time_Seconds = ridge_timing,
  Memory_Usage_MB = ridge_mem_diff / (1024^2),
  AUC = ridge_roc_auc
))

cat("Ridge AUC:", round(ridge_roc_auc, 4), "\n")
cat("Ridge Training Time:", round(ridge_timing, 2), "minutes\n")
# -----------------------
# XGBOOST METRICS
# -----------------------
# Store XGBoost results
comparison_results <- rbind(comparison_results, data.frame(
  Model = "XGBoost",
  Training_Time_Seconds = xgb_timing,
  Memory_Usage_MB = xgb_mem_diff / (1024^2),
  AUC = auc_xgb
))

cat("XGBoost AUC:", round(auc_xgb, 4), "\n")
cat("XGBoost Training Time:", round(xgb_timing, 2), "minutes\n")

# ----------------------
# COMPARISON TABLE
# ----------------------
kable(comparison_results, caption = "Model Performance Comparison", digits = 3)
# Performance differences
cat("\n=== PERFORMANCE DIFFERENCES ===\n")
cat("AUC Difference (XGB - Ridge):", round(auc_xgb - ridge_roc_auc, 4), "\n")
time_diff <- as.numeric(xgb_timing, units = "mins") - as.numeric(ridge_timing, units = "mins")
cat("Training Time Difference (XGB - Ridge):", round(time_diff, 2), "minutes\n")
# ------------------------
# VISUALIZATION COMPARISON
# ------------------------
# Combine the ROC curves
# plot ridge roc on sample set
# use true positieve rate (sensitivity) vs false positive rate (1 - specificity)
idx <- sample(1:length(y_test), 30000)
roc_ridge <- roc(y_test[idx], as.vector(ridge_pred[idx]))
roc_xgb <- roc(y_test[idx], pred_xgb[idx])

ggplot() +
  geom_line(data = data.frame(
    fpr = 1 - roc_ridge$specificities,
    tpr = roc_ridge$sensitivities
  ), aes(x = fpr, y = tpr), color = "blue") +
  geom_line(data = data.frame(
    fpr = 1 - roc_xgb$specificities,
    tpr = roc_xgb$sensitivities
  ), aes(x = fpr, y = tpr), color = "red") +
  labs(
    title = "ROC Curve Comparison",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  geom_abline(linetype = "dashed") +
 theme_bw() +
 annotate("text", x = 0.8, y = 0.4, label = paste("Ridge AUC =", round(ridge_roc_auc, 3)), color = "blue") +
 annotate("text", x = 0.8, y = 0.35, label = paste("XGBoost AUC =", round(auc_xgb, 3)), color = "red")
# Save the ROC comparison plot
ggsave("roc_comparison.png", plot = last_plot(), width = 8, height = 6)

# ------------------------
# CONFUSION MATRICES
# ------------------------
ridge_pred_class <- ifelse(ridge_pred >= 0.5, 1, 0)
pred_class_xgb <- ifelse(pred_xgb >= 0.5, 1, 0)
ridge_conf_mat <- confusionMatrix(
  factor(ridge_pred_class, levels = c('0','1')),
  factor(y_test, levels = c('0','1')),
  positive = "1"
)

xgb_conf_mat <- confusionMatrix(
  factor(pred_class_xgb, levels = c('0','1')),
  factor(y_test, levels = c('0','1')),
  positive = '1'
)
#print confusion matrices
kable(ridge_conf_mat$table, caption = "Ridge Regression Confusion Matrix")
kable(xgb_conf_mat$table, caption = "XGBoost Confusion Matrix")

# print percentages of true positives and true negatives
cat("Ridge True Positive Rate:", round(ridge_conf_mat$byClass["Sensitivity"], 3), "\n")
cat("Ridge True Negative Rate:", round(ridge_conf_mat$byClass["Specificity"], 3), "\n")
cat("XGBoost True Positive Rate:", round(xgb_conf_mat$byClass["Sensitivity"], 3), "\n")
cat("XGBoost True Negative Rate:", round(xgb_conf_mat$byClass["Specificity"], 3), "\n")

# f1 score
cat("Ridge F1-Score from Confusion Matrix:", round(ridge_conf_mat$byClass["F1"], 3), "\n")
cat("XGBoost F1-Score from Confusion Matrix:", round(xgb_conf_mat$byClass["F1"], 3), "\n")

# Add f1 to comparison table
comparison_results$F1_Score <- c(
  ridge_conf_mat$byClass["F1"],
  xgb_conf_mat$byClass["F1"]
)
# Final comparison table
kable(comparison_results, caption = "Final Model Performance Comparison", digits = 3)

#----------------------------
# save big objects for report
saveRDS(
  list(
    session_features = session_features ,
    cvfit2 = cvfit2 ,
    ridge_pred = ridge_pred,
    xgb_model = xgb_model
  ),
  "output/ridge_results.rds"
)
