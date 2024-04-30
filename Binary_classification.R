library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(pROC)
library(MASS)  
library(glmnet) 
library(GGally)


set.seed(123)
data <- read_csv("OnlineRetail.csv")
data$InvoiceDate <- as.POSIXct(data$InvoiceDate, format="%Y-%m-%d %H:%M:%S")
data$InvoiceDay <- as.Date(data$InvoiceDate)

data <- data[, !names(data) %in% c("InvoiceDate")]
data <- na.omit(data)
data$TotalPrice <- data$Quantity * data$UnitPrice

# Aggregating data
agg_data <- data %>%
  group_by(CustomerID) %>%
  summarise(TotalSpend = sum(TotalPrice),
            Frequency = n_distinct(InvoiceNo),
            Recency = as.numeric(difftime(max(InvoiceDay), min(InvoiceDay), units = "days"))) %>%
  ungroup() %>% 
  na.omit()

# Create a binary outcome
agg_data$HighSpender <- ifelse(agg_data$TotalSpend > median(agg_data$TotalSpend), 1, 0)

# Scaling the features
preProcValues <- preProcess(agg_data[, c("TotalSpend", "Frequency", "Recency")], method = c("center", "scale"))
scaled_data <- predict(preProcValues, agg_data[, c("TotalSpend", "Frequency", "Recency")])

# PCA for feature reduction
pca <- prcomp(scaled_data)
pca_data <- data.frame(pca$x)
pca_data <- cbind(pca_data, HighSpender = agg_data$HighSpender, CustomerID = agg_data$CustomerID)

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(pca_data$HighSpender, p = .8, list = FALSE, times = 1)
train_data <- pca_data[trainIndex, ]
test_data <- pca_data[-trainIndex, ]

# Models
# LDA Model
lda_fit <- lda(HighSpender ~ . - CustomerID, data=train_data)
summary(lda_fit)

# QDA Model
qda_fit <- qda(HighSpender ~ . - CustomerID, data=train_data)
summary(qda_fit)

X_train <- as.matrix(train_data[, setdiff(names(train_data), c("HighSpender", "CustomerID"))])
y_train <- train_data$HighSpender

# Lasso and Ridge Regression for Logistic Regression
model_lasso <- glmnet(X_train, y_train, alpha = 1, family = "binomial")
model_ridge <- glmnet(X_train, y_train, alpha = 0, family = "binomial")

# Cross-validation for Lasso and Ridge
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)

# Plot the CV error
plot(cv_lasso)
plot(cv_ridge)
X_test <- as.matrix(test_data[, setdiff(names(test_data), c("HighSpender", "CustomerID"))])
y_test <- test_data$HighSpender

# Model Predictions
predictions_lda <- predict(lda_fit, newdata=test_data)$posterior[,2]
predictions_qda <- predict(qda_fit, newdata=test_data)$posterior[,2]
predictions_lasso <- predict(model_lasso, newx=X_test, s=cv_lasso$lambda.min, type="response")
predictions_ridge <- predict(model_ridge, newx=X_test, s=cv_ridge$lambda.min, type="response")

roc_lda <- roc(y_test, predictions_lda)
roc_qda <- roc(y_test, predictions_qda)
roc_lasso <- roc(y_test, predictions_lasso)
roc_ridge <- roc(y_test, predictions_ridge)

# Summarize ROC results
print(summary(roc_lda))
print(summary(roc_qda))
print(summary(roc_lasso))
print(summary(roc_ridge))
threshold <- 0.5
pred_lda_binary <- ifelse(predictions_lda > threshold, 1, 0)
pred_qda_binary <- ifelse(predictions_qda > threshold, 1, 0)
pred_lasso_binary <- ifelse(predictions_lasso > threshold, 1, 0)
pred_ridge_binary <- ifelse(predictions_ridge > threshold, 1, 0)

# Confusion Matrices
confusion_lda <- confusionMatrix(factor(pred_lda_binary, levels=c(0, 1)), factor(y_test, levels=c(0, 1)))
confusion_qda <- confusionMatrix(factor(pred_qda_binary, levels=c(0, 1)), factor(y_test, levels=c(0, 1)))
confusion_lasso <- confusionMatrix(factor(pred_lasso_binary, levels=c(0, 1)), factor(y_test, levels=c(0, 1)))
confusion_ridge <- confusionMatrix(factor(pred_ridge_binary, levels=c(0, 1)), factor(y_test, levels=c(0, 1)))

# Print Confusion Matrices
print("Confusion Matrix for LDA:")
print(confusion_lda)
print("Confusion Matrix for QDA:")
print(confusion_qda)
print("Confusion Matrix for LASSO:")
print(confusion_lasso)
print("Confusion Matrix for RIDGE:")
print(confusion_ridge)