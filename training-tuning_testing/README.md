# Model Training and Tuning

This section gives a detailed explanation of the models trained, the different hyperparameters tuned and the metric used for selecting the optimal hyperparameters. The models trained for each of the 4 datasets are :-

- Logistic Regression
- SVM with Polynomial kernel
- SVM with RBF Kernel
- Random Forest
- Nueral Networks

5-fold cross validation procedure with 3 repitions were used to tune the hyperparameters of the models. For creating folds, `creatMultiFolds` procedure was used so as to have the same data folds for each model, so as to ensure the difference in predicitive performance was down only due to the predicitive capibilities of the model, and not because of the data being different. `caret` was used to carry out the training and tuning procedure for the first 4 models and `keras`, in conjuction with `tensorflow` was used to train and tune neural networks. 

### Ranked Probability Score (RPS)

The metric used to select the optimal hyperparameter was **Ranked Probability Score (RPS)** . The Ranked Probability Score (RPS) is a measure of how good forecasts are in matching observed outcomes. For RPS :

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

1. Multinomial Logisitic Regression - 
