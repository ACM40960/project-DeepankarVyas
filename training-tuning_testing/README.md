# Model Training and Tuning

This section gives a detailed explanation of the models trained, the different hyperparameters tuned and the metric used for selecting the optimal hyperparameters. The models trained for each of the 4 datasets are :-

- Logistic Regression
- SVM with Polynomial kernel
- SVM with RBF Kernel
- Random Forest
- Nueral Networks

5-fold cross validation procedure with 3 repetitions were used to tune the hyperparameters of the models. For creating folds, `createMultiFolds` procedure was used to have the same data folds for each model, so as to ensure the difference in predicitive performance was down only due to the predicitive capibilities of the model, and not because of the data being different. `caret` was used to carry out the training and tuning procedure for the first 4 models and `keras`, in conjuction with `tensorflow` was used to train and tune neural networks. 

### Ranked Probability Score (RPS)

The metric used to select the optimal hyperparameter was **Ranked Probability Score (RPS)** [4] . The Ranked Probability Score (RPS) is a measure of how good forecasts are in matching observed outcomes. For RPS :

- RPS = 0 the forecast is wholly accurate;
- RPS = 1 the forecast is wholly inaccurate.

Therefore, the lower the RPS value is, the better the predictions are .

<p align="center">
$\text{RPS} = \frac{1}{r-1} \sum_{i=1}^{r-1} \left(\sum_{j=1}^{i} \left(p_j - e_j \right)\right)^2$
</p>

Since caret does not provide a built in metric to select the optimal hyperparameter using RPS, we use Custom functions to do the same .

```
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

```
and this is defined in `trainControl` as 

```
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

```
For each model, other metrices such as `Sensitivity`, `Specificity`,`F1 Score`, `Balanced Accuracy`,`Precision` and `AUC` were also calculated based on the predictions of the model using its optimal hyperparameters. 

### Model Description

1. Multinomial Logisitic Regression - Multinomial Logistic Regression is an extension of logistic regression used for modeling outcomes with more than two categories. It models the probability of each category as a function of the independent variables using the softmax function .  There were no parameters to tune as such for logistic regression and since feture selection was already done , with our datasat divided into 4 datasets, it was applied using all the features to predict the match outcome.
2. SVM with Polynomial Kernel - A Support Vector Machine (SVM) with a polynomial kernel is a type of supervised learning algorithm used for classification and regression tasks. The polynomial kernel transforms the input data into a higher-dimensional space, enabling the SVM to capture more complex patterns that are not linearly separable in the original feature space. The kernel function computes the similarity between data points as a polynomial function of their dot product, allowing the model to account for the non-linear relationship between input and the output.

The hyperparameters tuned for SVM Polynomial Kernel are :- C (Cost - trade-off between maximizing the margin and minimizing classification errors), scale (the influence of a single training example) and degree (the flexibility of the decision boundary). The heatmap depicting the RPS values at various points of the hyperparameter grid is shown below. The optimal hyperparameter is the one with the lowest RPS value

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="SVM Polynomial" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/training-tuning_testing/README%20images/SVM_Poly_Heatmap_page-0001%20(1).jpg">
   <p style="text-align: center;"><em>Figure 1: Heatmap of SVM Polynomial kernel.</em></p> 
</div>

3. SVM with RBF Kernel - The RBF kernel maps input data into a higher-dimensional space, where it can draw non-linear decision boundaries. This kernel function measures the similarity between data points based on their distance, with the kernel value decreasing exponentially as the distance increases and is particularly powerful in capturing the non linear relationship between input and output.

The hyperparameters tuned for SVM RBF Kernel are :- C (Cost - trade-off between maximizing the margin and minimizing classification errors),and sigma (the width of the Gaussian function used in the RBF kernel). The heatmap depicting the RPS values at various points of the hyperparameter grid is shown below. The optimal hyperparameter is the one with the lowest RPS value

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="SVM RBF" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/training-tuning_testing/README%20images/SVM_RBF_Heatmap_page-0001%20(1).jpg">
   <p style="text-align: center;"><em>Figure 1: Heatmap of SVM RBF kernel.</em></p> 
</div>

4. Random Forest - Random Forest is an ensemble learning method that builds multiple decision trees during training and merges their outputs to improve accuracy and control overfitting. Each tree in the forest is trained on a random subset of the data, with features also randomly selected at each split point, ensuring diversity among the trees. The final prediction is made by taking a majority vote (in classification) from all the individual trees.

The parameters tuned for Random forest are - mtry (number of features used at split) and ntree (number of trees ). Since `caret` does not provide an inbuilt implementation to tune ntree, a custom implementation was used to tune this hyperparameter. The heatmap depicting the RPS values at various points of the hyperparameter grid is shown below. The optimal hyperparameter is the one with the lowest RPS value

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Random Forest" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/training-tuning_testing/README%20images/RF_Heatmap_page-0001%20(1).jpg">
   <p style="text-align: center;"><em>Figure 1: Heatmap of Random Forest.</em></p> 
</div>

5. Neural networks - neural networks are the deep learning implementation of Machine Learning. They mimic the neurological functioning of the human brain. The structure of the neural networks used in our model is as follows :-

3 layers with 128, 64 and 32 units respectively. 100 epochs with a batch size of 32 and early stopping criterion. Dropout rate and L2 regularization were also used which were tuned. Learning rate was tuned implicitly using Adam optimizer. The heatmap depicting the RPS values at various points of the hyperparameter grid is shown below. The optimal hyperparameter is the one with the lowest RPS value

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Random Forest" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/training-tuning_testing/README%20images/NN_Heatmap_page-0001%20(2).jpg">
   <p style="text-align: center;"><em>Figure 1: Heatmap of Neural Networks.</em></p> 
</div>

This directory contains both .R and .qmd files, and either can be run to train the models. There are 4 subdirectories with nomenclature - **[dataset_name]_images**, which contains the visualizations generated during model training for each dataset. 

The enire model training process took more than $${\color{red}8 \space \color{red}days}$$ to complete on a machine with standard specifications . It is highly recommended to either run the process using parallel computation or run the process on a machine with higher computing power. However, user can bypass the model training process and use the results stored to proceed to the next step - MODEL EVALUATION. The results of model training have been stored in `.rds` files at the time of model development and testing. This directory contains various `.rds` files with the nomenclature - **results_[dataset_name]_[model_name].rds**, since the standalone files for each dataset were more than 100mb in size and could not be pushed. These `.rds` files contain both the final model trained and the performance metrics evaluated for each model of each dataset.  The `.rds` file of SVM Polynomila kernel is subdivided into models and metrics file as a single was still excedding the size limit. The logic of combining the files and using the results has been implemented in the .R/.qmd file of MODEL EVALUATION section.
