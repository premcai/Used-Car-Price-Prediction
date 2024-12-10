#Libraries
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(rpart)
library(ggplot2)
library(mlbench)
library(xgboost)
library(corrplot)

#Dataset
data1 <- read.csv("D:/University/SEM 7/R prog/Mini Project/car_prediction_data.csv")

# Data Preprocessing
data1 <- data1 %>%
  select(-Car_Name) %>% # Remove irrelevant column
  mutate(  #into categorical variables
    Fuel_Type = as.factor(Fuel_Type),
    Seller_Type = as.factor(Seller_Type),
    Transmission = as.factor(Transmission)
  ) %>%
  na.omit() # Remove rows with missing values

# Feature Selection
correlation_matrix <- cor(data1 %>% select_if(is.numeric))
corrplot(correlation_matrix, method = "color", type = "lower", tl.cex = 0.8)

set.seed(123)
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
feature_selection <- rfe(
  data1 %>% select(-Selling_Price),
  data1$Selling_Price,
  sizes = c(5, 10, 15),
  rfeControl = control
)
selected_features <- predictors(feature_selection)
data1 <- data1 %>% select(all_of(selected_features), Selling_Price)

#Stratified Sampling
evaluate_models <- function(train_size) {
  set.seed(123)
  train_index <- createDataPartition(data1$Fuel_Type, p = train_size, list = FALSE) # Stratified sampling
  train_data <- data1[train_index, ]
  test_data <- data1[-train_index, ]
  
  # Align factor levels 
  train_data$Fuel_Type <- factor(train_data$Fuel_Type, levels = levels(data1$Fuel_Type))
  test_data$Fuel_Type <- factor(test_data$Fuel_Type, levels = levels(data1$Fuel_Type))
  
  train_data$Seller_Type <- factor(train_data$Seller_Type, levels = levels(data1$Seller_Type))
  test_data$Seller_Type <- factor(test_data$Seller_Type, levels = levels(data1$Seller_Type))
  
  train_data$Transmission <- factor(train_data$Transmission, levels = levels(data1$Transmission))
  test_data$Transmission <- factor(test_data$Transmission, levels = levels(data1$Transmission))
  
  #results list
  results <- list()
  
  # KNN
  train_control <- trainControl(method = "cv", number = 5)
  knn_model <- train(Selling_Price ~ ., data = train_data, method = "knn", trControl = train_control)
  predictions_knn <- predict(knn_model, test_data)
  results$KNN <- list(
    RMSE = RMSE(predictions_knn, test_data$Selling_Price),
    MAE = MAE(predictions_knn, test_data$Selling_Price)
  )
  
  # Linear Regression
  linear_model <- lm(Selling_Price ~ ., data = train_data)
  predictions_linear <- predict(linear_model, test_data)
  results$Linear <- list(
    RMSE = RMSE(predictions_linear, test_data$Selling_Price),
    MAE = MAE(predictions_linear, test_data$Selling_Price)
  )
  
  # Random Forest
  rf_grid <- expand.grid(.mtry = c(2, 3, 4))
  rf_model <- train(
    Selling_Price ~ ., data = train_data, method = "rf",
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = rf_grid
  )
  predictions_rf <- predict(rf_model, test_data)
  results$Random_Forest <- list(
    RMSE = RMSE(predictions_rf, test_data$Selling_Price),
    MAE = MAE(predictions_rf, test_data$Selling_Price)
  )
  
  # Decision Tree
  decision_tree_model <- rpart(Selling_Price ~ ., data = train_data)
  predictions_tree <- predict(decision_tree_model, test_data)
  results$Decision_Tree <- list(
    RMSE = RMSE(predictions_tree, test_data$Selling_Price),
    MAE = MAE(predictions_tree, test_data$Selling_Price)
  )
  
  # XGBoost
  train_matrix <- as.matrix(
    train_data %>%
      mutate(across(where(is.factor), as.numeric)) %>%
      select(-Selling_Price)
  )
  test_matrix <- as.matrix(
    test_data %>%
      mutate(across(where(is.factor), as.numeric)) %>%
      select(-Selling_Price)
  )
  xgb_model <- xgboost(
    data = train_matrix,
    label = train_data$Selling_Price,
    nrounds = 100,
    eta = 0.1,
    max_depth = 3,
    objective = "reg:squarederror",
    verbose = 0
  )
  predictions_xgb <- predict(xgb_model, test_matrix)
  results$XGBoost <- list(
    RMSE = RMSE(predictions_xgb, test_data$Selling_Price),
    MAE = MAE(predictions_xgb, test_data$Selling_Price)
  )
  
  return(results)
}

#20%, 50%, and 70% Training Sizes
results_20 <- evaluate_models(0.2)
results_50 <- evaluate_models(0.5)
results_70 <- evaluate_models(0.7)

#Results
final_results <- data.frame(
  Model = rep(c("KNN", "Linear Regression", "Random Forest", "Decision Tree", "XGBoost"), 3),
  Train_Size = rep(c("20%", "50%", "70%"), each = 5),
  RMSE = c(
    results_20$KNN$RMSE, results_20$Linear$RMSE, results_20$Random_Forest$RMSE, results_20$Decision_Tree$RMSE, results_20$XGBoost$RMSE,
    results_50$KNN$RMSE, results_50$Linear$RMSE, results_50$Random_Forest$RMSE, results_50$Decision_Tree$RMSE, results_50$XGBoost$RMSE,
    results_70$KNN$RMSE, results_70$Linear$RMSE, results_70$Random_Forest$RMSE, results_70$Decision_Tree$RMSE, results_70$XGBoost$RMSE
  ),
  MAE = c(
    results_20$KNN$MAE, results_20$Linear$MAE, results_20$Random_Forest$MAE, results_20$Decision_Tree$MAE, results_20$XGBoost$MAE,
    results_50$KNN$MAE, results_50$Linear$MAE, results_50$Random_Forest$MAE, results_50$Decision_Tree$MAE, results_50$XGBoost$MAE,
    results_70$KNN$MAE, results_70$Linear$MAE, results_70$Random_Forest$MAE, results_70$Decision_Tree$MAE, results_70$XGBoost$MAE
  )
)
print(final_results)

# Visualize RMSE
ggplot(final_results, aes(x = Train_Size, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "RMSE Comparison Across Training Sizes", x = "Training Set Size", y = "RMSE") +
  theme_minimal()

# Visualize MAE 
ggplot(final_results, aes(x = Train_Size, y = MAE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "MAE Comparison Across Training Sizes", x = "Training Set Size", y = "MAE") +
  theme_minimal()
