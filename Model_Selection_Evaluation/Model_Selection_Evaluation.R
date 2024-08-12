#' ---
#' title: "Model_Selection_Evaluation"
#' author: Deepankar Vyas, 23200527
#' format:
#'   html:
#'     embed-resources: true
#'   pdf: default
#' execute: 
#'   warning: false
#' ---
#' 
#' ## MODEL SELECTION, GENERALIZED PERFORMANCE EVALUATION AND TESTING AGAINST BENCHMARK SETTING {.smaller}
#' 
#' ### Model Selection
#' 
#' We trained the following models for CLASS A and CLASS A NLP datasets :-
#' 
#' 1.  Logistic Regression
#' 2.  SVM with Polynomial Kernel
#' 3.  SVM with Gaussia Kernle
#' 4.  Random Forest
#' 5.  Neural Networks
#' 
#' And the following models were trained for CLASS B and CLASS B NLP :-
#' 
#' 1.  SVM with Polynomial Kernel
#' 2.  SVM with Gaussia Kernle
#' 3.  Random Forest
#' 4.  Neural Networks
#' 
#' We now have to find the best performing model and the dataset which performs the best. We will use usual evaluation metrics to select the best performing model (Balanced Accuracy, Precision, Recall, F1-Score, AUC-ROC) with a special emphasis on the DRAW class as it is the hardest to predict. The final model will be retrained on the entire training + validation set and then used to predict the test set, to determine its generalized predictive performance. We will also use specialized performance metric - RANKED PROBABILITY SCORE to determine how our selected model performed against the models employed by the betting companies.
#' 
## ---------------------------------------------------------------------------------------------------------------
#| label: loading models

results_class_a <- readRDS("../training-tuning_testing/results_class_a.rds")

results_class_a_nlp <- readRDS("../training-tuning_testing/results_class_a_nlp.rds")

results_class_b <- readRDS("../training-tuning_testing/results_class_b.rds")

results_class_b_nlp <- readRDS("../training-tuning_testing/results_class_b_nlp.rds")

# results_list <- readRDS("results_list_temp_before_changing_grid_testing_purpose.rds")

output_dir <- "eval_images"

# Create the directory if it does not exist
if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }


#' 
## ---------------------------------------------------------------------------------------------------------------
#| label: Model Selection


library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)
library(keras)
library(pROC)
library(ROCR)
library(knitr)
library(kableExtra)
library(ggpubr)
library(multiROC)
library(BBmisc)

# Function to plot metrics for all classes
plot_all_classes_metrics <- function(metrics_df, metric_name, individual =FALSE) {

  if (!(individual)) {
    # Summarize the data
    data_plot <- metrics_df %>%
      group_by(Dataset, Model) %>%
      summarise(Mean_val = mean(pull(across(all_of(metric_name))), na.rm = TRUE))
    
    # Create the bar plot for aggregated data
    ggplot(data_plot, aes(x = Model, y = Mean_val, fill = Model)) +
      geom_bar(stat = "identity", position = "dodge", color = "black", size = 0.3) +
      geom_text(aes(label = round(Mean_val, 3), y = Mean_val / 2), 
                position = position_dodge(width = 0.9), 
                vjust = 0.5, 
                color = "black", size = 3.5) +
      labs(title = paste(metric_name, "Across All Models"), 
           x = "Model", 
           y = metric_name) +
      scale_fill_brewer(palette = "Paired") +  # Use smoother color palette
      theme_minimal(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold", margin = margin(b = 20)),
        legend.position = "none",
        plot.title.position = "plot"  # Move title outside of the plot area
      ) +
      facet_wrap(~ Dataset)
  } else {
    # Filter the dataset for draw class
    draw_class_data <- metrics_combined %>% filter(Class == "Class: D")
    
    # Create the bar plot for draw class
    ggplot(draw_class_data, aes(x = Model, y = !!sym(metric_name), fill = Dataset)) +
      geom_bar(stat = "identity", position = "dodge", color = "black", size = 0.3) +
      geom_text(aes(label = round(!!sym(metric_name), 2), y = !!sym(metric_name)+0.003), 
                position = position_dodge(width = 0.9), 
                vjust = 0.5, color = "black", size = 3.5) +
      labs(title = paste(metric_name, "for Draw Class Across Models and Datasets"),
           x = "Model",
           y = metric_name) +
      scale_fill_brewer(palette = "Set3") +  # Use smoother color palette
      theme_minimal(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold", margin = margin(b = 20)),
        plot.title.position = "plot"  # Move title outside of the plot area
      )
  }
  

}

# Combine metrics from all models
combine_metrics <- function(results_list, dataset_name) {
  metrics_list <- lapply(results_list, function(x) x$metrics)
  models <- names(results_list)
  metrics_df <- data.frame(
    Model = rep(models, each = length(metrics_list[[1]]$Class)),
    Class = unlist(lapply(metrics_list, function(x) x$Class)),
    mean_sensitivity = unlist(lapply(metrics_list, function(x) x$mean_sensitivity)),
    mean_specificity = unlist(lapply(metrics_list, function(x) x$mean_specificity)),
    mean_balanced_accuracy = unlist(lapply(metrics_list, function(x) x$mean_balanced_accuracy)),
    mean_f1_score = unlist(lapply(metrics_list, function(x) x$mean_f1_score)),
    mean_precision = unlist(lapply(metrics_list, function(x) x$mean_precision)),
    mean_auc = unlist(lapply(metrics_list, function(x) x$mean_auc)),
    mean_rps = unlist(lapply(metrics_list, function(x) x$rps)),
    Dataset = dataset_name
  )
  
  return(metrics_df)
}

# Evaluate and plot metrics
evaluate_and_plot <- function(metrics_combined) {

  # Plotting balanced accuracy and mean F1 score across all classes
  plot_all_classes_metrics(metrics_combined, "mean_balanced_accuracy")
  ggsave(file.path(output_dir, paste0("Plot_mean_balanced_accuracy.pdf")))
  
  plot_all_classes_metrics(metrics_combined,"mean_f1_score")
  ggsave(file.path(output_dir, paste0("Plot_mean_f1_score.pdf")))
  
  plot_all_classes_metrics(metrics_combined,"mean_rps")
  ggsave(file.path(output_dir, paste0("Plot_mean_rps.pdf")))

  plot_all_classes_metrics(metrics_combined, "mean_f1_score", individual = TRUE)
  ggsave(file.path(output_dir, paste0("Plot_mean_f1_draw.pdf")))
  
  plot_all_classes_metrics(metrics_combined, "mean_sensitivity", individual = TRUE)
  ggsave(file.path(output_dir, paste0("Plot_mean_sensitivity_draw.pdf")))
  
  plot_all_classes_metrics(metrics_combined, "mean_precision", individual = TRUE)
  ggsave(file.path(output_dir, paste0("Plot_mean_precision_draw.pdf")))

  return(metrics_combined)
}

# Function to prepare data for multiROC
prepare_multiroc_data <- function(results_list) {
  multi_roc_data <- list()
  
  for (dataset_name in names(results_list)) {
    dataset <- results_list[[dataset_name]]
    
    for (model_name in names(dataset)) {
      if (model_name != "nn"){
        model <- dataset[[model_name]]$model
        
        true_labels <- model$pred$obs
        pred_probs <- model$pred[, c("A", "D", "H")]
        
        # Prepare data for multiROC
        roc_data <- data.frame(
          A_true = ifelse(true_labels == "A", 1, 0),
          D_true = ifelse(true_labels == "D", 1, 0),
          H_true = ifelse(true_labels == "H", 1, 0),
          A_pred = pred_probs$A,
          D_pred = pred_probs$D,
          H_pred = pred_probs$H
        )
        
        colnames(roc_data) <- c(
          "A_true", "D_true", "H_true",
          paste0("A_pred_", model_name),
          paste0("D_pred_", model_name),
          paste0("H_pred_", model_name)
        )
        
        multi_roc_data[[paste(dataset_name, model_name, sep = "_")]] <- roc_data
      }
    }
  }
  
  return(multi_roc_data)
}

# Function to plot ROC curves for each model of each dataset and for each class of the model
plot_multiroc_curves <- function(multi_roc_data) {
  plot_list <- list()
  
  for (roc_data_name in names(multi_roc_data)) {
    roc_data <- multi_roc_data[[roc_data_name]]
    roc_res <- multi_roc(roc_data)
    roc_res_df <- plot_roc_data(roc_res)
    
    # Update Group labels with AUROC values
    roc_res_df <- roc_res_df %>%
      mutate(Group = case_when(
        Group == "A" ~ paste0("A (AUROC = ", round(AUC, 2), ")"),
        Group == "D" ~ paste0("D (AUROC = ", round(AUC, 2), ")"),
        Group == "H" ~ paste0("H (AUROC = ", round(AUC, 2), ")"),
        Group == "Macro" ~ paste0("Macro (AUROC = ", round(AUC, 2), ")"),
        Group == "Micro" ~ paste0("Micro (AUROC = ", round(AUC, 2), ")"),
        TRUE ~ Group
      ))
    
    print(paste("Starting plot for ", roc_data_name))
    p <- ggplot(roc_res_df, aes(x = 1 - Specificity, y = Sensitivity, color = Group)) +
      geom_line() +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      labs(
        title = paste("ROC Curve for", roc_data_name),
        x = "False Positive Rate",
        y = "True Positive Rate"
      ) +
      theme_minimal()
    
    plot_list[[roc_data_name]] <- p
    print(paste("Done - ",roc_data_name))
  }
  
  return(plot_list)
}

# Function to build and compile Keras model
build_model <- function(dropout, lambda, input_shape) {
  keras_model_sequential() %>%
    layer_dense(units = 128, input_shape = input_shape, activation = "relu",
                kernel_regularizer = regularizer_l2(lambda)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = 64, activation = "relu",
                kernel_regularizer = regularizer_l2(lambda)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = 32, activation = "relu",
                kernel_regularizer = regularizer_l2(lambda)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = length(unique(train_val_data$FTR)), activation = "softmax") %>%
    compile(
      loss = "categorical_crossentropy",
      optimizer = optimizer_adam(),
      metrics = c("accuracy")
    )
}

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


# Retrain the best model and evaluate on test set
retrain_and_evaluate <- function(best_model,best_dataset, train_val_data, test_data) {

    if (best_model == "nn") {
    # Get optimal dropout and lambda values from stored values
    optimal_dropout <- eval(parse(text = paste0("results_list[['", 
                                                best_dataset, "']][['",
                                                best_model, "']]$model")))$dropout[[1]]
    optimal_lambda <- eval(parse(text = paste0("results_list[['", 
                                                best_dataset, "']][['",
                                                best_model, "']]$model")))$lambda[[1]]
    
    # Build and compile the Keras model
    model <- build_model(optimal_dropout, optimal_lambda, ncol(train_val_data)-1)
    
    
    x_train <- as.matrix(train_val_data[-1])
    y_train <- to_categorical(as.integer(as.factor(train_val_data$FTR))-1)
    x_test <- as.matrix(test_data[-1])
    y_test <- to_categorical(as.integer(as.factor(test_data$FTR))-1)
    
    # Fit the model
    model %>% fit(
      x = x_train, y = y_train,
      validation_data = list(x_test, y_test),
      epochs = 200,
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
    
    # Predict and evaluate on test set
    
     y_pred <- model %>% predict(x_test)
      
     y_pred_label <- apply(y_pred, 1, which.max)
      
     y_true <- max.col(y_test)
      
     metrics <- calculate_metrics(y_true, y_pred_label)
     
     data <- data.frame(obs = as.factor(test_data$FTR), y_pred)
     colnames(data)[-1] <- c("A", "D", "H")
     metrics["rps"] <- customSummary(data, lev = c('A','D','H'))
     
     metrics <- metrics %>%
       mutate(Class = recode(Class, 
                             "Class: 1" = "Class: A", 
                             "Class: 2" = "Class: D", 
                             "Class: 3" = "Class: H"))
     # Prepare data for multiROC
     
     true_labels <- as.factor(test_data$FTR)
     pred_probs <- as.data.frame(y_pred)
     colnames(pred_probs) <- c("A", "D", "H")
     roc_data_test <- data.frame(
       A_true = ifelse(true_labels == "A", 1, 0),
       D_true = ifelse(true_labels == "D", 1, 0),
       H_true = ifelse(true_labels == "H", 1, 0),
       A_pred = pred_probs$A,
       D_pred = pred_probs$D,
       H_pred = pred_probs$H
     )
     
     colnames(roc_data_test) <- c(
       "A_true", "D_true", "H_true",
       paste0("A_pred_", best_model),
       paste0("D_pred_", best_model),
       paste0("H_pred_", best_model)
     )
     
     roc_res <- multi_roc(roc_data_test)
     roc_res_df <- plot_roc_data(roc_res)
     
     # Update Group labels with AUROC values
     roc_res_df <- roc_res_df %>%
       mutate(Group = case_when(
         Group == "A" ~ paste0("A (AUROC = ", round(AUC, 2), ")"),
         Group == "D" ~ paste0("D (AUROC = ", round(AUC, 2), ")"),
         Group == "H" ~ paste0("H (AUROC = ", round(AUC, 2), ")"),
         Group == "Macro" ~ paste0("Macro (AUROC = ", round(AUC, 2), ")"),
         Group == "Micro" ~ paste0("Micro (AUROC = ", round(AUC, 2), ")"),
         TRUE ~ Group
       ))
     
     print(paste("Starting test roc plot "))
     ggplot(roc_res_df, aes(x = 1 - Specificity, y = Sensitivity, color = Group)) +
       geom_line() +
       geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
       labs(
         title = paste("Test ROC Curve"),
         x = "False Positive Rate",
         y = "True Positive Rate"
       ) +
       theme_grey()
     
     ggsave(file.path(output_dir, paste0(best_model,"_Test_ROC_Curve.pdf")))
      
 } 
  else {
    # Directly use the existing model for other types of models
    final_model <-  eval(parse(text = paste0("results_list[['", 
                                                best_dataset, "']][['",
                                                best_model, "']]$model")))
    predictions <- predict(final_model, test_data, type = "prob")
    pred_label <- predict(final_model, test_data)
    metrics <- calculate_metrics(as.factor(test_data$FTR), pred_label)
    
    data <- data.frame(obs = as.factor(test_data$FTR), predictions)
    colnames(data)[-1] <- c("A", "D", "H")
    metrics["rps"] <- customSummary(data, lev = c('A','D','H'))
    
    # Prepare data for multiROC
    
    true_labels <- as.factor(test_data$FTR)
    pred_probs <- predictions
    roc_data_test <- data.frame(
      A_true = ifelse(true_labels == "A", 1, 0),
      D_true = ifelse(true_labels == "D", 1, 0),
      H_true = ifelse(true_labels == "H", 1, 0),
      A_pred = pred_probs$A,
      D_pred = pred_probs$D,
      H_pred = pred_probs$H
    )
    
    colnames(roc_data_test) <- c(
      "A_true", "D_true", "H_true",
      paste0("A_pred_", best_model),
      paste0("D_pred_", best_model),
      paste0("H_pred_", best_model)
    )
    
    roc_res <- multi_roc(roc_data_test)
    roc_res_df <- plot_roc_data(roc_res)

    # Update Group labels with AUROC values
    roc_res_df <- roc_res_df %>%
      mutate(Group = case_when(
        Group == "A" ~ paste0("A (AUROC = ", round(AUC, 2), ")"),
        Group == "D" ~ paste0("D (AUROC = ", round(AUC, 2), ")"),
        Group == "H" ~ paste0("H (AUROC = ", round(AUC, 2), ")"),
        Group == "Macro" ~ paste0("Macro (AUROC = ", round(AUC, 2), ")"),
        Group == "Micro" ~ paste0("Micro (AUROC = ", round(AUC, 2), ")"),
        TRUE ~ Group
      ))
    
    # Plot the ROC curves with updated legend labels
    ggplot(roc_res_df, aes(x = 1 - Specificity, y = Sensitivity, color = Group)) +
      geom_line() +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      labs(
        title = paste("Test ROC Curve"),
        x = "False Positive Rate",
        y = "True Positive Rate"
      ) +
      theme_minimal()
    
    ggsave(file.path(output_dir, paste0(best_model,"_Test_ROC_Curve.pdf")))

  }
  
  return(metrics)
}


results_list <- list(
  class_a = results_class_a,
  class_b = results_class_b,
  class_a_nlp = results_class_a_nlp,
  class_b_nlp = results_class_b_nlp
)

# Loop through each dataset in results_list
for (dataset in names(results_list)) {
  # Check if the dataset has the nn model
  if ("nn" %in% names(results_list[[dataset]])) {
    # Update the mean_rps to rps
    if ("mean_rps" %in% colnames(results_list[[dataset]]$nn$metrics)) {
      results_list[[dataset]]$nn$metrics <- results_list[[dataset]]$nn$metrics %>%
        rename(rps = mean_rps)
    }
  }
}

metrics_combined_list <- lapply(names(results_list), function(dataset_name) {
  combine_metrics(results_list[[dataset_name]], dataset_name)
})

metrics_combined <- bind_rows(metrics_combined_list)

# Prepare the multiROC data
multi_roc_data <- prepare_multiroc_data(results_list)

# Plot the ROC curves
roc_plots <- plot_multiroc_curves(multi_roc_data)

# Display the ROC plots
for (plot_name in names(roc_plots)) {
  print(roc_plots[[plot_name]])
  ggsave(file.path(output_dir, paste0(plot_name," ROC Curve.pdf")))
  print(paste("saved ",plot_name))
}

metrics_combined <- evaluate_and_plot(metrics_combined)

# Calculate the means for each dataset
dataset_means <- metrics_combined %>%
  group_by(Model,Dataset) %>%
  summarise(
    mean_f1_all_classes = mean(mean_f1_score, na.rm = TRUE),
    mean_balanced_accuracy_all_classes = mean(mean_balanced_accuracy, na.rm = TRUE),
    mean_rps_all_classes = mean(mean_rps, na.rm = TRUE)
  ) |> arrange(Dataset)

# Calculate the mean_f1_score for draw class
draw_class_means <- metrics_combined %>%
  filter(Class == "Class: D") |> arrange(Dataset, Model)

# Merge the draw class means back to the original dataframe
ranking_dataset <- cbind(dataset_means,draw_class_means$mean_f1_score)

# Change the column name for the draw class mean F1 score
colnames(ranking_dataset)[ncol(ranking_dataset)] <- "mean_f1_draw"

ranking_dataset <- ranking_dataset %>% filter(!is.na(mean_f1_draw))

# Step 1: Rank based on highest mean_f1_draw
ranking_dataset <- ranking_dataset %>%
  arrange(desc(mean_f1_draw))

# Step 2: Explore different threshold values
highest_f1_draw <- ranking_dataset$mean_f1_draw[1]
threshold_values <- seq(0.01, 0.1, by = 0.01)

# Minimum threshold to ensure at least 2 models
minimum_threshold <- 0.05

# Function to count the number of models within a given threshold
count_models_within_threshold <- function(threshold) {
  threshold_f1_draw <- highest_f1_draw - threshold
  filtered_ranking_dataset <- ranking_dataset %>%
    filter(mean_f1_draw >= threshold_f1_draw)
  nrow(filtered_ranking_dataset)
}

# Apply the function to all threshold values
model_counts <- sapply(threshold_values, count_models_within_threshold)

# Plot the number of models within each threshold
plot_data <- data.frame(
  threshold = threshold_values,
  model_count = model_counts
)

ggplot(plot_data, aes(x = threshold, y = model_count)) +
  geom_line(size = 1.2, color = "#0073C2") +
  geom_point(size = 3, shape = 21, fill = "#E69F00", color = "#0073C2") +
  labs(
    title = "Number of Models within Different Thresholds",
    subtitle = "Threshold values to determine optimal model selection",
    x = "Threshold Value",
    y = "Number of Models"
  ) +
  theme_minimal(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 16),
    axis.title.x = element_text(face = "bold", size = 16),
    axis.title.y = element_text(face = "bold", size = 16),
    axis.text = element_text(size = 14),
    panel.grid.major = element_line(color = "gray80", size = 0.5),
    panel.grid.minor = element_line(color = "gray90", size = 0.25)
  ) +
  geom_text(aes(label = model_count), vjust = -1, size = 5, color = "#0073C2") +
  scale_x_continuous(breaks = seq(0.01, 0.1, by = 0.01)) +
  scale_y_continuous(breaks = 1:8)

ggsave(file.path(output_dir, paste0("Models_Thresholds.pdf")))

# Choose an optimal threshold value based on the plot
optimal_threshold <- threshold_values[which.max(model_counts >= 2 & model_counts <= 5)]

# Ensure the optimal threshold is not below the minimum threshold
if (is.na(optimal_threshold) || optimal_threshold > minimum_threshold) {
  optimal_threshold <- minimum_threshold
}

# Step 3: Filter models within the optimal threshold
threshold_f1_draw <- highest_f1_draw - optimal_threshold
filtered_ranking_dataset <- ranking_dataset %>%
  filter(mean_f1_draw >= threshold_f1_draw)

# Step 4: Among those, arrange by lowest mean_rps_all_classes
final_ranking_dataset <- filtered_ranking_dataset %>%
  arrange(mean_rps_all_classes)

# Print the final ranked dataset
print(final_ranking_dataset)

pdf(file = "eval_images/ranking_dataset.pdf", height=nrow(ranking_dataset)/3, width=11)
gridExtra::grid.table(ranking_dataset)
dev.off()

pdf(file = "eval_images/final_ranking_dataset.pdf", height=nrow(ranking_dataset)/3, width=11)
gridExtra::grid.table(final_ranking_dataset)
dev.off()

best_model <- final_ranking_dataset$Model[1]
best_dataset <- final_ranking_dataset$Dataset[1]


# Retrain the best model and evaluate on test set
# Replace 'train_val_data' and 'test_data' with your actual datasets
train_val_data <- read.csv(paste0("../Dataset/",best_dataset,"_train.csv"))  # Combine training and validation data
test_data <- read.csv(paste0("../Dataset/",best_dataset,"_test.csv"))

final_eval_metrics <-  eval(parse(text = paste0("results_list[['", 
                                                best_dataset, "']][['",
                                                best_model, "']]$metrics")))


pdf(file = "eval_images/final_eval_metrics.pdf", height=nrow(final_eval_metrics)/3, width=11)
gridExtra::grid.table(final_eval_metrics)
dev.off()


best_model_metrics <- retrain_and_evaluate(best_model,best_dataset, 
                                           train_val_data, test_data)

# Print the final evaluation metrics
print(best_model_metrics)

pdf(file = "eval_images/best_model_metrics.pdf", height=nrow(best_model_metrics)/3, width=11)
gridExtra::grid.table(best_model_metrics)
dev.off()

#' 
## ---------------------------------------------------------------------------------------------------------------

#| label : benchmark test

set.seed(23200527)

master_dataset <- read.csv("../Dataset/Master_Dataset_Final.csv")
train_indices <- createDataPartition(master_dataset$FTR, p = 0.8, list = FALSE)
train_data_odds <- master_dataset[train_indices,
                                  c("B365H","B365A","B365D","FTR") ]
test_data_odds <- master_dataset[-train_indices,
                                 c("B365H","B365A","B365D","FTR") ]


# Compute implied probabilities
implied_probs <- data.frame(
  HomeWin = 1 / test_data_odds$B365H,
  Draw = 1 / test_data_odds$B365D,
  AwayWin = 1 / test_data_odds$B365A
)

# Compute the sum of implied probabilities
booksum <- rowSums(implied_probs)

# Normalize the probabilities
normalized_probs <- implied_probs / booksum

# Calculating RPS

data <- data.frame(obs = as.factor(test_data$FTR), normalized_probs)
# Rename columns as requested
colnames(data)[-1] <- c("H", "D", "A")

# Reorder the columns
data <- data[, c("obs", "A", "D", "H")]

# Calculate the mean RPS for BET365
mean_rps_bet365 <- customSummary(data, lev = c('A','D','H'))

# Fixed value
fixed_value <- 0.2156

# RPS of the final model
best_model_rps <- best_model_metrics$rps[1]

# Create the dataframe with the desired columns
rps_comparison <- data.frame(
  BET365 = round(mean_rps_bet365,4),
  Fixed_Value = round(fixed_value,4),
  Final_Model = round(best_model_rps,4)
)

# Rename the columns as required
colnames(rps_comparison) <- c("BET365", "Baboota_Kaur", paste0(best_model, "_", best_dataset))

# Print the resulting dataframe
print(rps_comparison)

pdf(file = "eval_images/rps_comparison.pdf", height=nrow(best_model_metrics)/3, width=11)
gridExtra::grid.table(rps_comparison)
dev.off()


# Create the dataframe with the desired columns
rps_comparison <- data.frame(
  Model = c("BET365", "Baboota_Kaur", paste0(best_model, "_", best_dataset)),
  RPS = c(round(mean_rps_bet365, 4), round(fixed_value, 4), round(best_model_rps, 4))
)

# Define the colors for each model
colors <- c("BET365" = "#FF9999", "Baboota_Kaur" = "#66B2FF")
colors[paste0(best_model, "_", best_dataset)] <- "#99FF99"

# Plot the barplot
ggplot(rps_comparison, aes(x = Model, y = RPS, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = RPS), vjust = -0.3, size = 4.5, color = "black") +
  scale_fill_manual(values = colors) +
  labs(title = "Ranked Probability Score (RPS) Comparison",
       x = "Model",
       y = "RPS") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold", margin = margin(b = 20)),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(size = 12, face = "bold"),
    axis.text.y = element_text(size = 12),
    panel.grid.major = element_line(size = 0.1, color = "grey"),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white", color = NA)
  )

# Save the plot as a PDF
ggsave(file.path(output_dir, paste0("rps_comparison_plot.pdf")))




