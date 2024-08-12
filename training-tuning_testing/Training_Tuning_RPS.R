#' ---
#' title: "Training_Tuning"
#' author: Deepankar Vyas, 23200527
#' format:
#'   html:
#'     embed-resources: true
#'   pdf: default
#' execute: 
#'   warning: false
#' ---
#' 
## ---------------------------------------------------------------------------------------------------------------
#| label: Loading libraries
#| echo: false

library(doParallel)
library(e1071)
library(randomForest)
library(keras)
library(tfruns)
library(tidyverse)
library(pROC)
library(MASS)
library(ROCR)
library(ggplot2)
library(reshape2)
library(tensorflow)
library(caret)

#' 
## ---------------------------------------------------------------------------------------------------------------
#| label: Function to calculate metrics

# Function to calculate metrics for each class and return them
calculate_metrics <- function(y_true, y_pred) {
  
  levels <- sort(unique(c(y_true, y_pred)))
  y_true_factor <- factor(y_true, levels = levels)
  y_pred_factor <- factor(y_pred, levels = levels)
  
  confusion_mat <- confusionMatrix(y_pred_factor, y_true_factor)
  
  metrics_df <- data.frame(
    Class = rownames(confusion_mat$byClass),
    Sensitivity = confusion_mat$byClass[, "Sensitivity"],
    Specificity = confusion_mat$byClass[, "Specificity"],
    Balanced_Accuracy = confusion_mat$byClass[, "Balanced Accuracy"],
    F1_Score = confusion_mat$byClass[, "F1"],
    Precision = confusion_mat$byClass[, "Precision"]
  )
  
  unique_classes <- sort(unique(y_true))
  
  for (class in unique_classes) {
    y_true_binary <- ifelse(y_true == class, 1, 0)
    y_pred_binary <- ifelse(y_pred == class, 1, 0)
    
    # Calculate AUC
    roc_obj <- roc(y_true_binary, as.numeric(y_pred_binary))
    auc_value <- auc(roc_obj)
    metrics_df$AUC[metrics_df$Class == paste0("Class: ",class)] <- auc_value
    
  }
  
  return(metrics_df)
}

average_metrics <- function(metrics_list) {
  # Combine the list of data frames into a single data frame
  combined_df <- bind_rows(metrics_list)
  
  # Calculate the mean for each class and each metric
  averaged_df <- combined_df %>%
    group_by(Class) %>%
    summarise(
      mean_sensitivity = mean(Sensitivity, na.rm = TRUE),
      mean_specificity = mean(Specificity, na.rm = TRUE),
      mean_balanced_accuracy = mean(Balanced_Accuracy, na.rm = TRUE),
      mean_f1_score = mean(F1_Score, na.rm = TRUE),
      mean_precision = mean(Precision, na.rm = TRUE),
      mean_auc = mean(AUC, na.rm = TRUE)
    )
  
  if ("rps" %in% colnames(combined_df)) {
    averaged_df <- averaged_df %>%
      mutate(mean_rps = combined_df %>%
               group_by(Class) %>%
               summarise(mean_rps = mean(rps, na.rm = TRUE)) %>%
               pull(mean_rps))
  }
  
  return(averaged_df)
}

customSummary <- function(data, lev = NULL, model = NULL) {
  probs <- as.matrix(data[, lev])
  obs <- as.numeric(data$obs)
  
  rps <- calculateRPSMetric(probs, obs)
  mean_rps <- mean(rps)
  
  out <- c(RPS = mean_rps)
  return(out)
}

calculateRPSMetric <- function(predictions, observed) {
  ncat <- ncol(predictions)
  npred <- nrow(predictions)
  
  rps <- numeric(npred)
  
  for (rr in 1:npred) {
    obsvec <- rep(0, ncat)
    obsvec[observed[rr]] <- 1
    cumulative <- 0
    for (i in 1:ncat) {
      cumulative <- cumulative + (sum(predictions[rr, 1:i]) - sum(obsvec[1:i]))^2
    }
    rps[rr] <- (1 / (ncat - 1)) * cumulative
  }
  return(rps)
}


#' 
## ---------------------------------------------------------------------------------------------------------------
#| label: Training

run_analysis <- function(train_file, output_dir) {
  # Load data
  train <- read.csv(train_file)
  
  
  # Create the directory if it does not exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }

  # Seed for reproducibility
  set.seed(23200527)
  
  # Cross-validation settings
  cv_folds <- createMultiFolds(train$FTR, k = 5, times = 3 )
  
  train_ctrl <- trainControl(
    method = "repeatedcv",
    index = cv_folds,
    number = 5,
    repeats = 3,
    summaryFunction = customSummary,
    classProbs = TRUE,
    savePredictions = TRUE,
    returnResamp = "all",
    verboseIter = TRUE
  )
  
  # Adjusted grid of hyperparameters for SVM with polynomial kernel
  tune_grid_svm_poly <- expand.grid(
    C = c(0.1, 1, 5, 7, 10, 15, 20, 22, 25),
    scale = c(0.001, 0.01, 0.1, 0.2, 0.5, 1),
    degree = c(2, 3, 5, 10, 15, 20)
  )
  
  # Grid of hyperparameters for SVM with Radial Basis kernel
  tune_grid_svm_rbf <- expand.grid(
    sigma = c(0.01, 0.05, 0.1, 0.5, 0.7),
    C = c(1, 10, 100, 200, 500)
  )
  
  # Random Forest parameters
  ntree_set <- c(100, 200, 500, 1000)
  grid_rf <- expand.grid(mtry = 2:7)
  
  # Multinomial Logistic Regression with stepwise AIC selection
  set.seed(23200527)
    
  print("Starting Logistic Regression")
  fit_lr <- train(FTR ~ ., data = train,
                  method = "multinom",
                  trControl = train_ctrl,
                  metric = "RPS",
                  maximize=FALSE)
  print(fit_lr)

  # Identify the best hyperparameters
  best_hyperparameters <- fit_lr$bestTune
  # Filter predictions for the best hyperparameters
  optimal_lr_predictions <- fit_lr$pred %>%
    filter(decay == best_hyperparameters$decay)
  
  # Calculate metrics on the sampled data
  lr_metrics <- optimal_lr_predictions %>%
    group_by(Resample) %>%
    summarise(
      metrics = list(calculate_metrics(obs, pred)),
      length_of_group = n()
  )
  
  
  results_list$lr <<- list(metrics = average_metrics(lr_metrics$metrics), 
                                model = fit_lr)
  results_list$lr$metrics['rps'] <<- results_list$lr$model$results |>
    filter(decay == best_hyperparameters$decay) |> dplyr::select(RPS)
  
  print("Finished Logistic Regression")
  
  # SVM with polynomial kernel
  set.seed(23200527)
  print("Starting SVM Poly")
  fit_svm_poly <- train(
    FTR ~ ., data = train,
    method = "svmPoly",
    trControl = train_ctrl,
    tuneGrid = tune_grid_svm_poly,
    metric = "RPS",
    maximize=FALSE
  )
  print(fit_svm_poly)
  
  plot(fit_svm_poly)
  ggsave(file.path(output_dir, "SVM_Poly_Plot.pdf"))
  
  # Heatmap for SVM Polynomial Kernel Hyperparameters
  results_svm_poly <- as.data.frame(fit_svm_poly$results)
  results_svm_poly_melt <- melt(results_svm_poly, 
                                id.vars = c("C", "scale", "degree"),
                                measure.vars = c("RPS"))
  
  # Generate heatmap for SVM Polynomial Kernel
  ggplot(results_svm_poly_melt, aes(x = as.factor(C), 
                                    y = as.factor(scale), 
                                    fill = value)) +
    geom_tile() +
    facet_wrap(~ degree) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = mean(results_svm_poly$RPS,na.rm = TRUE)) +
    labs(title = "Heatmap of SVM Polynomial Kernel Hyperparameters", 
         x = "C", y = "Scale", fill = "RPS") +
    theme_minimal()
  ggsave(file.path(output_dir, "SVM_Poly_Heatmap.pdf"))
  
  
  # Identify the best hyperparameters
  best_hyperparameters <- fit_svm_poly$bestTune
  
  # Filter predictions for the best hyperparameters
  optimal_svm_poly_predictions <- fit_svm_poly$pred %>%
    filter(C == best_hyperparameters$C,
           scale == best_hyperparameters$scale,
           degree == best_hyperparameters$degree)
  
  # Calculate metrics on the sampled data
  svm_poly_metrics <- optimal_svm_poly_predictions %>%
    group_by(Resample) %>%
    summarise(
      metrics = list(calculate_metrics(obs, pred)),
      length_of_group = n()
  )
  
  results_list$svm_poly <<- list(metrics = average_metrics(svm_poly_metrics$metrics), 
                                model = fit_svm_poly)
  results_list$svm_poly$metrics['rps'] <<- results_list$svm_poly$model$results |>
    filter(C == best_hyperparameters$C,
           scale == best_hyperparameters$scale,
           degree == best_hyperparameters$degree) |> dplyr::select(RPS)
  
  print("Finished SVM Poly")
  # SVM with radial basis kernel
  set.seed(23200527)
  print("Starting SVM Gaussian")
  fit_svm_rbf <- train(
    FTR ~ ., data = train,
    method = "svmRadial",
    trControl = train_ctrl,
    tuneGrid = tune_grid_svm_rbf,
    metric = "RPS",
    maximize=FALSE
  )
  print(fit_svm_rbf)
  
  plot(fit_svm_rbf)
  ggsave(file.path(output_dir, "SVM_RBF_Plot.pdf"))
  
  # Heatmap for SVM Radial Basis Kernel Hyperparameters
  results_svm_rbf <- as.data.frame(fit_svm_rbf$results)
  results_svm_rbf_melt <- melt(results_svm_rbf, 
                               id.vars = c("C", "sigma"),
                               measure.vars = c("RPS"))
  
  # Generate heatmap for SVM Radial Basis Kernel
  ggplot(results_svm_rbf_melt, aes(x = as.factor(C), 
                                   y = as.factor(sigma), 
                                   fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", 
                         mid = "white", 
                         midpoint = mean(results_svm_rbf$RPS, na.rm = TRUE)) +
    labs(title = "Heatmap of SVM Radial Basis Kernel Hyperparameters",
         x = "C", y = "Sigma", 
         fill = "RPS") +
    theme_minimal()
  ggsave(file.path(output_dir, "SVM_RBF_Heatmap.pdf"))
  
  
  # Identify the best hyperparameters
  best_hyperparameters <- fit_svm_rbf$bestTune
  
  # Filter predictions for the best hyperparameters
  optimal_svm_rbf_predictions <- fit_svm_rbf$pred %>%
    filter(C == best_hyperparameters$C,
           sigma == best_hyperparameters$sigma)
  
  # Calculate metrics on the sampled data
  svm_rbf_metrics <- optimal_svm_rbf_predictions %>%
    group_by(Resample) %>%
    summarise(
      metrics = list(calculate_metrics(obs, pred)),
      length_of_group = n()
  )
  
  results_list$svm_rbf <<- list(metrics = average_metrics(svm_rbf_metrics$metrics), 
                                model = fit_svm_rbf)
  results_list$svm_rbf$metrics['rps'] <<- results_list$svm_rbf$model$results |>
    filter(C == best_hyperparameters$C,
           sigma == best_hyperparameters$sigma) |> dplyr::select(RPS)

  
  print("Finished SVM Gaussian")
  # Random Forest tuning
  set.seed(23200527)
  print("Starting Random Forest")
  out_rf <- vector("list", length(ntree_set))
  for (j in 1:length(ntree_set)) {
    set.seed(23200527)
    out_rf[[j]] <- train(
      FTR ~ ., data = train, 
      method = "rf", 
      metric = "RPS",
      maximize=FALSE,
      trControl = train_ctrl, 
      tuneGrid = grid_rf, 
      ntree = ntree_set[j],
      na.action = na.pass
    )
    print(paste("Training for tree - ",ntree_set[j]))
  }
  
  # extracting and tidy up AU-ROC values
  acc <- sapply(out_rf, "[[", c("results", "RPS"))
  colnames(acc) <- ntree_set
  acc <- cbind(grid_rf, acc)
  head(acc)
  
  # plotting curves
  cols <- c("purple2", "forestgreen", "darkorange3",
            "deepskyblue3",
            "cyan","red","black")
  matplot(acc$mtry, acc[,-1], type = "l", lwd = 2, lty = 1,
          col = cols[1:4],xlab = "Variables to split",
          ylab = "RPS")
  grid()
  legend("topright", legend = ntree_set, fill = cols[1:4], bty = "n")
  ggsave(file.path(output_dir, "RF_Plot.pdf"))
  
  # changing names for out list
  names(out_rf) <- paste0("ntree_", ntree_set)
  
  print(out_rf$ntree_100)
  print(out_rf$ntree_200)
  print(out_rf$ntree_500)
  print(out_rf$ntree_1000)
  
  #Finding the optimal (ntree,mtry) set from the competing 
  #random forest models
  res <- resamples(out_rf)
  
  # Heatmap for Random Forest Hyperparameters
  results_rf <- do.call(rbind, lapply(out_rf, function(x) x$results))
  results_rf$ntree <- rep(ntree_set, each = nrow(grid_rf))
  results_rf_melt <- melt(results_rf, id.vars = c("mtry", "ntree"), 
                          measure.vars = c("RPS"))
  
  # Generate heatmap for Random Forest
  ggplot(results_rf_melt, aes(x = as.factor(mtry), y = as.factor(ntree), 
                              fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", 
                         mid = "white", 
                         midpoint = mean(results_rf$RPS,na.rm = TRUE)) +
    labs(title = "Heatmap of Random Forest Hyperparameters",
         x = "Mtry", y = "Ntree", 
         fill = "RPS") +
    theme_minimal()
  ggsave(file.path(output_dir, "RF_Heatmap.pdf"))
  
  
  
  best_model_index <- which.min(sapply(out_rf, 
                                       function(x) min(x$results$RPS)))
  best_model_rf <- out_rf[[best_model_index]]
  best_mtry <- best_model_rf$bestTune$mtry
  best_ntree <- ntree_set[best_model_index]
  
  # Filter predictions for the best hyperparameters
  optimal_rf_predictions <- best_model_rf$pred %>%
    filter(mtry == best_mtry)
  
  # Calculate metrics on the sampled data
  rf_metrics <- optimal_rf_predictions %>%
    group_by(Resample) %>%
    summarise(
      metrics = list(calculate_metrics(obs, pred)),
      length_of_group = n()
  )
  
  results_list$rf <<- list(metrics = average_metrics(rf_metrics$metrics), 
                                model = best_model_rf)
  results_list$rf$metrics['rps'] <<- results_list$rf$model$results |>
    filter(mtry == best_mtry) |> dplyr::select(RPS)

  print("Finished Random Forest")
  
  print("Starting Neural Networks")
  # Hyperparameter grid for Neural Networks
  nn_grid <- expand.grid(
    dropout = c(0.3, 0.5, 0.7),
    lambda = c(0, 0.01, 0.001, 0.0001, 0.00001)
  )

  # Function to build and compile Keras model
  build_model <- function(dropout, lambda) {
    keras_model_sequential() %>%
      layer_dense(units = 128, input_shape = ncol(train)-1, activation = "relu",
                  kernel_regularizer = regularizer_l2(lambda)) %>%
      layer_dropout(rate = dropout) %>%
      layer_dense(units = 64, activation = "relu",
                  kernel_regularizer = regularizer_l2(lambda)) %>%
      layer_dropout(rate = dropout) %>%
      layer_dense(units = 32, activation = "relu",
                  kernel_regularizer = regularizer_l2(lambda)) %>%
      layer_dropout(rate = dropout) %>%
      layer_dense(units = length(unique(train$FTR)), activation = "softmax") %>%
      compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer_adam(),
        metrics = c("accuracy")
      )
  }

  # Loop for cross-validation with replications
  results_nn <- data.frame()

  for (i in 1:nrow(nn_grid)) {
    cat("Training with parameters:", nn_grid[i, ]$dropout,", ",nn_grid[i,]$lambda, "\n")
    
    cv_results <- list()
    
    for (fold in unique(cv_folds)) {
      train_index <- fold
      val_index <- setdiff(1:nrow(train), fold)
      
      train_fold <- train[train_index, ]
      val_fold <- train[val_index, ]
      
      x_train <- as.matrix(train_fold[-1])
      y_train <- to_categorical(as.integer(as.factor(train_fold$FTR))-1)
      x_val <- as.matrix(val_fold[-1])
      y_val <- to_categorical(as.integer(as.factor(val_fold$FTR))-1)
      
      model <- build_model(
        dropout = nn_grid$dropout[i],
        lambda = nn_grid$lambda[i]
      )
      
      fit <- model %>% fit(
        x = x_train, y = y_train,
        validation_data = list(x_val, y_val),
        epoch = 100,
        batch_size = 32,
        verbose = 1,
        callbacks = list(
        callback_early_stopping(
          monitor = "val_accuracy",
          patience = 20,
          restore_best_weights = TRUE  # Restore the best weights observed during training
        )
        )
      )
      
      y_pred <- model %>% predict(x_val)
      
      y_pred_label <- apply(y_pred, 1, which.max)
      
      y_true <- max.col(y_val)
      
      metrics_df <- calculate_metrics(y_true, y_pred_label)
      data <- data.frame(obs = y_true, y_pred)
      colnames(data)[-1] <- c("A", "D", "H")
      metrics_df["rps"] <- customSummary(data, lev = c('A','D','H'))
      
      results_df <- cbind(metrics_df, 
                          Dropout = nn_grid$dropout[i],
                          Lambda = nn_grid$lambda[i])
      
      cv_results <- append(cv_results, list(results_df))
    }
    
    avg_results <- average_metrics(cv_results)
    avg_results$dropout <- nn_grid[i,1]
    avg_results$lambda <- nn_grid[i,2]
    results_nn <- rbind(results_nn, avg_results)
  }
  
    print(results_nn)
  
  # Find optimal hyperparameters
  optimal_nn_params <- results_nn %>%
    group_by(dropout, lambda) %>%
    summarise(mean_rps = mean(mean_rps)) %>%
    arrange(mean_rps)

  
  # Heatmap for Neural Network Hyperparameters
  optimal_nn_params <- optimal_nn_params %>% mutate(id = row_number())
  results_nn_melt <- melt(optimal_nn_params, 
                          id.vars = c("id", "dropout", "lambda"),
                          measure.vars = c("mean_rps"))
  
  # Generate heatmap for Neural Network
  ggplot(results_nn_melt, aes(x = as.factor(dropout),
                              y = as.factor(lambda), fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", 
                         mid = "white", 
                         midpoint = mean(results_nn$mean_rps, na.rm = TRUE)) +
    labs(title = "Heatmap of Neural Network Hyperparameters", 
         x = "Dropout", y = "Lambda", fill = "mean_rps") +
    theme_minimal()
  ggsave(file.path(output_dir, "NN_Heatmap.pdf"))

  
  nn_metrics <- results_nn |>  filter (dropout==optimal_nn_params$dropout[1],
                                       lambda==optimal_nn_params$lambda[1]) 
  
  nn_metrics <- nn_metrics %>%
  mutate(Class = recode(Class, 
                        "Class: 1" = "Class: A", 
                        "Class: 2" = "Class: D", 
                        "Class: 3" = "Class: H"))
  
  results_list$nn <<- list(
    metrics = nn_metrics[, c("Class", "mean_sensitivity", 
                             "mean_specificity", "mean_balanced_accuracy",
                             "mean_f1_score", "mean_precision", "mean_auc", "mean_rps")], 
    model = nn_metrics[, c("dropout", "lambda")]
  )


  print("Finished Neural Networks")


  
  
}

#' 
## ---------------------------------------------------------------------------------------------------------------
#| label : running for each dataset

results_list <- list()
run_analysis("../Dataset/class_a_train.csv",
                                "class_a_images")
results_class_a <- results_list

saveRDS(results_class_a, "results_class_a.rds")


#' 
## ---------------------------------------------------------------------------------------------------------------

results_list <- list()
run_analysis("../Dataset/class_a_nlp_train.csv", 
                                    "class_a_nlp_images")
results_class_a_nlp <- results_list

saveRDS(results_class_a_nlp, "results_class_a_nlp.rds")


#' 
## ---------------------------------------------------------------------------------------------------------------


results_list <- list()
run_analysis("../Dataset/class_b_train.csv", 
                                "class_b_images")

results_class_b <- results_list

saveRDS(results_class_b, "results_class_b.rds")


#' 
## ---------------------------------------------------------------------------------------------------------------


results_list <- list()
run_analysis("../Dataset/class_b_nlp_train.csv", 
                                "class_b_nlp_images")

results_class_b_nlp <- results_list
saveRDS(results_class_b_nlp, "results_class_b_nlp.rds")

