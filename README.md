<p align="center"><img width=20.5% src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/Project_Logo.png"></p>
<p align="center"><img width=100% src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/project%20Title.jpg"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![R](https://img.shields.io/badge/R-v4.0+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![GitHub Issues](https://img.shields.io/github/issues/yourusername/yourproject.svg)](https://github.com/yourusername/yourproject/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


This projects aims to use Machine Learning models to predict the outcome of Football Matches using a combination of Feature Engineering and Sentiment Analysis.

### Table of Contents 📑
- [Motivation](#motivation)
- [Setting Up](#setting-up)
- [Directory Structure](#directory-structure)
- [Methodology](#methodology)
  - [Dataset Preparation](#dataset-preparation)
  - [Feature Engineering](#feature-engineering)
  - [EDA and Feature Selection](#eda-and-feature-selection)
  - [Model Training and Tuning](#model-training-and-tuning)
  - [Model Evaluation](#model-evaluation)
  - [Generalized Predictive Performance](#generalized-predictive-performance)
  - [Comparison Against Benchmark Setting](#comparison-against-benchmark-setting)
  - [Forthcoming Research](#forthcoming-research)
- [Poster](https://github.com/ACM40960/project-DeepankarVyas/blob/main/Final%20Project%20Poster.pdf) <em> <- Click Here </em>
- [Contributing to the Project](#contributing-to-the-project)
- [Licenses](#licenses)
- [Data Sources](#data-sources)
- [References](#references)
- [Authors](#authors)


### Motivation

The objective of this project is to develop a model capable of accurately predicting the outcomes of football matches. This model aims to challenge existing betting models by leveraging a unique combination of engineered features, derived from domain-specific knowledge, and sentiment analysis of pre-match reports. By integrating and enhancing previous approaches—some of which focus solely on statistical features while others emphasize text analytics—this project seeks to determine whether a synergistic combination of these methodologies can yield superior predictive results.

### Setting Up

1. Clone the repository:
   ```
   git clone https://github.com/ACM40960/project-DeepankarVyas.git   
   ```
2. After cloning the repository, it will create a dedicated project folder in your local working directory. Entire code of the project is written in **R** language and contains only **.R/.qmd** files. These files can be easily run in RStudio.
2. RStudio is an integrated development environment (IDE) for R programming. It provides a user-friendly interface for writing and running R scripts, visualizing data, and generating reports. Rstudio can be installed using this [link](https://posit.co/products/open-source/rstudio/).
3. Install the required packages :
   ```
   # Vector of required packages
    packages <- c("knitr","rvest","dplyr","stringr","lubridate","tidyr","furrr", "ggplot2", 
                  "data.table","readr", "purrr","rstudioapi","zoo","tidyverse","tidytext","syuzhet",
                  "sentimentr","lexicon","textdata","caret","wordcloud","RColorBrewer","plotly","GGally","gridExtra",
                  "nortest","ggpubr","reshape2","car","Boruta","randomForest","doParallel","grid","e1071","keras","tfruns",
                  "pROC","MASS","ROCR","tensorflow","caret","kableExtra","multiROC","BBmisc") 

    # Function to check if a package is installed, and install it if not
    install_if_missing <- function(packages) {
      for (pkg in packages) {
        if (!requireNamespace(pkg, quietly = TRUE)) {
          install.packages(pkg)
        }
      }
    }

    # Install any missing packages
    install_if_missing(packages)
   ```

### Directory Structure

The entire code base is divided into directories, with each directoriy's content pertaining to a specific task of the project. Eah directory has these 2 files, along with its specific contents :-

1. *.R file - **R** script containg the code carrying out a specific task of the project
2. *.qmd file - **qmd** file containg the code carrying out a specific task of the project.

The code in both the files is exactly the same, with .R files updated with the latest comments. However, any of the files can be run in the RStudio to fetch the desired results.

```
├── dataset-preparation/
│   ├── Bundesliga_2012_2024/                  # Folder containing historical football data files from various seasons.
│   │   ├── D1_1213.csv                        # Match data for the 2012-2013 Bundesliga season.
│   │   ├── D1_1314.csv                        # Match data for the 2013-2014 Bundesliga season.
│   │   ├── D1_1415.csv                        # Match data for the 2014-2015 Bundesliga season.
│   │   ├── D1_1516.csv                        # Match data for the 2015-2016 Bundesliga season.
│   │   ├── D1_1617.csv                        # Match data for the 2016-2017 Bundesliga season.
│   │   ├── D1_1718.csv                        # Match data for the 2017-2018 Bundesliga season.
│   │   ├── D1_1819.csv                        # Match data for the 2018-2019 Bundesliga season.
│   │   ├── D1_1920.csv                        # Match data for the 2019-2020 Bundesliga season.
│   │   ├── D1_2021.csv                        # Match data for the 2020-2021 Bundesliga season.
│   │   ├── D1_2122.csv                        # Match data for the 2021-2022 Bundesliga season.
│   │   ├── D1_2223.csv                        # Match data for the 2022-2023 Bundesliga season.
│   │   └── D1_2324.csv                        # Match data for the 2023-2024 Bundesliga season.
│   ├── dataset_preparation.qmd                # Quarto document for preparing dataset for analysis.
│   └── dataset_preparation.R                  # R script used for preparing dataset for analysis.
├── feature-engineering/
│   ├── feature-engineering.qmd                # Quarto document for performing feature engineering on datasets.
│   └── feature-engineering.R                  # R script containing the code for feature engineering processes.
├── data-cleanup-nlp/
│   ├── Dataset_Cleanup_NLP.qmd                # Quarto document detailing data cleaning and Senitment Analysis procedures.
│   └── Dataset_Cleanup_NLP.R                  # R script for executing data cleaning and Senitment Analysis procedures.
├── eda-splitting/
│   ├── Data_EDA_Splitting.qmd                 # Quarto document for Exploratory Data Analysis (EDA) and dataset splitting.
│   ├── Data_EDA_Splitting.R                   # R script for performing EDA and splitting the data into training/testing sets.
│   └── eda_images/                            # Directory containing visualizations and plots generated during EDA (contents not shown).
├── training-tuning-testing/
│   ├── class_a_images/                        # Directory containing visualizations and plots related to Class A models (contents not shown).
│   ├── class_a_nlp_images/                    # Directory containing visualizations and plots related to Class A_NLP models (contents not shown).
│   ├── class_b_images/                        # Directory containing visualizations and plots related to Class B models (contents not shown).
│   ├── class_b_nlp_images/                    # Directory containing visualizations and plots related to Class B_NLP models (contents not shown).
│   ├── results_class_a_lr.rds                 # Saved results from logistic regression model for Class A.
│   ├── results_class_a_nlp_lr.rds             # Saved results from logistic regression model for Class A_NLP.
│   ├── results_class_a_nlp_nn.rds             # Saved results from NLP neural network model for Class A.
│   ├── results_class_a_nlp_rf.rds             # Saved results from random forest model for Class A_NLP.
│   ├── results_class_a_nlp_svm_poly_metrics.rds # Performance metrics for SVM (polynomial kernel) model for Class A_NLP.
│   ├── results_class_a_nlp_svm_poly_model_part1.rds # Part 1 of the saved SVM (polynomial kernel) model for Class A_NLP.
│   ├── results_class_a_nlp_svm_poly_model_part2.rds # Part 2 of the saved SVM (polynomial kernel) model for Class A_NNLP.
│   ├── results_class_a_nlp_svm_rbf.rds        # Saved results from SVM (RBF kernel) model for Class A_NLP.
│   ├── results_class_a_nlp.rds                # Combined results for all models and metrics for Class A_NLP.
│   ├── results_class_a_nn.rds                 # Saved results from neural network model for Class A.
│   ├── results_class_a_rf.rds                 # Saved results from random forest model for Class A.
│   ├── results_class_a_svm_poly_metrics.rds   # Performance metrics for SVM (polynomial kernel) model for Class A.
│   ├── results_class_a_svm_poly_model_part1.rds # Part 1 of the saved SVM (polynomial kernel) model for Class A.
│   ├── results_class_a_svm_poly_model_part2.rds # Part 2 of the saved SVM (polynomial kernel) model for Class A.
│   ├── results_class_a_svm_rbf.rds            # Saved results from SVM (RBF kernel) model for Class A.
│   ├── results_class_a.rds                    # Combined results for all models for Class A.
│   ├── results_class_b_lr.rds                 # Saved results from logistic regression model for Class B.
│   ├── results_class_b_nlp_lr.rds             # Saved results from logistic regression model for Class B_NLP.
│   ├── results_class_b_nlp_nn.rds             # Saved results from neural network model for Class B_NLP.
│   ├── results_class_b_nlp_rf.rds             # Saved results from random forest model for Class B_NLP.
│   ├── results_class_b_nlp_svm_poly_metrics.rds # Performance metrics for SVM (polynomial kernel) model for Class B_NLP.
│   ├── results_class_b_nlp_svm_poly_model_part1.rds # Part 1 of the saved SVM (polynomial kernel) model for Class B_NLP.
│   ├── results_class_b_nlp_svm_poly_model_part2.rds # Part 2 of the saved SVM (polynomial kernel) model for Class B_NLP.
│   ├── results_class_b_nlp_svm_rbf.rds        # Saved results from SVM (RBF kernel) model for Class B_NLP.
│   ├── results_class_b_nlp.rds                # Combined results for all models for Class B_NLP.
│   ├── results_class_b_nn.rds                 # Saved results from neural network model for Class B.
│   ├── results_class_b_rf.rds                 # Saved results from random forest model for Class B.
│   ├── results_class_b_svm_poly_metrics.rds   # Performance metrics for SVM (polynomial kernel) model for Class B.
│   ├── results_class_b_svm_poly_model_part1.rds # Part 1 of the saved SVM (polynomial kernel) model for Class B.
│   ├── results_class_b_svm_poly_model_part2.rds # Part 2 of the saved SVM (polynomial kernel) model for Class B.
│   ├── results_class_b_svm_rbf.rds            # Saved results from SVM (RBF kernel) model for Class B.
│   ├── results_class_b.rds                    # Combined results for all models for Class B.
│   ├── Training_Tuning_RPS.qmd                # Quarto document detailing the training and tuning process for models.
│   └── Training_Tuning_RPS.R                  # R script for executing model training and tuning processes.
├── model-selection-evaluation/
│   ├── eval_images/                           # Directory containing evaluation visualizations and plots (contents not shown).
│   ├── Model_Selection_Evaluation.qmd         # Quarto document for model selection and evaluation.
│   └── Model_Selection_Evaluation.R           # R script for performing model selection and evaluation.
├── Dataset/
│   ├── class_a_nlp_test.csv                   # Test dataset for Class A_NLP.
│   ├── class_a_nlp_train.csv                  # Training dataset for Class A_NLP.
│   ├── class_a_test.csv                       # Test dataset for Class A models.
│   ├── class_a_train.csv                      # Training dataset for Class A models.
│   ├── class_b_nlp_test.csv                   # Test dataset for Class B_NLP.
│   ├── class_b_nlp_train.csv                  # Training dataset for Class B_NLP.
│   ├── class_b_test.csv                       # Test dataset for Class B models.
│   ├── class_b_train.csv                      # Training dataset for Class B models.
│   ├── Master_Dataset_Before_Preprocessing.csv # Raw master dataset before preprocessing.
│   ├── Master_Dataset_Final.csv               # Final master dataset after preprocessing.
│   └── Master_Dataset.csv                     # Initial master dataset after feature engineering.


```
