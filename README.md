<p align="center"><img width=20.5% src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/Project_Logo.png"></p>
<p align="center"><img width=100% src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/project%20Title.jpg"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![R](https://img.shields.io/badge/R-v4.0+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![GitHub Issues](https://img.shields.io/github/issues/yourusername/yourproject.svg)](https://github.com/yourusername/yourproject/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


This projects aims to use Machine Learning models to predict the outcome of Football Matches using a combination of Feature Engineering and Sentiment Analysis, focused primarily on the **German League (Bundesliga)**.

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
    packages <- c("knitr","rvest","dplyr","stringr","lubridate","tidyr","furrr",
                  "ggplot2", "data.table","readr", "purrr","rstudioapi","zoo",
                  "tidyverse","tidytext","syuzhet","sentimentr","lexicon",
                  "textdata","caret","wordcloud","RColorBrewer","plotly",
                  "GGally","gridExtra","nortest","ggpubr","reshape2","car",
                  "Boruta","randomForest","doParallel","grid","e1071","keras",
                  "tfruns","pROC","MASS","ROCR","tensorflow","caret","kableExtra",
                  "multiROC","BBmisc") 

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

### Methodology
#### Dataset Preparation

The dataset was prepared using Web - Scraping . Base Features and Betting Odds were taken from [Football-Data.co.uk](https://www.football-data.co.uk/germanym.php) [6] , Team Ratings were taken from [FIFA Index](https://www.fifaindex.com/teams/) [5]  and Pre - Match Reports were scraped from  - [WhoScored.com](https://www.whoscored.com/Regions/81/Tournaments/3/Germany-Bundesliga) [8] . Everything was collated to prepare a <code style= "color:Green">**Master Dataset**</code>.

The code to prepare the dataset using Web Scraping can be found here - [Dataset Preparation](https://github.com/ACM40960/project-DeepankarVyas/tree/main/dataset-preparation). This directory contains both the .R and .qmd files, and either can be run to Web Scrap the contents. First, the files for the past 10 Bundesliga seasons is downloaded from **Football-Data.co.uk**. Then, we web scrapped team's ratings from **FIFA Index** website staring from FIFA 12 till FIFA 24. These are the ratings given by FIFA video game to teams and players and are generated periodically, taking into account current form of teams and players, and thus are a good measure of the current state of the team. We focused on the team ratings and web scrapped following components of team ratings-
 1.  Attack Rating
 2.  Defense Rating
 3.  Midfield Rating
 4.  Overall Rating

This was accomplished using `rvest` package, which allows the functionality to download and manipulate xml contents. The ratings were then aggregrated by month so that each team has a defense, attack, midfield and overall rating for each month. Also, a custom team name mapping had to be defined as both the datasets scraped (Base features and betting odds; and the ratings dataset) had same team names worded differently. This was done to ensure uniformity. A sample is provided below :-
```
team_name_mapping <- list(
  "Bayern Munich" = c("Bayern München", "FC Bayern München", "FC Bayern Munich"),
  "Dortmund" = c("Borussia Dortmund"),
  "RB Leipzig" = c("RB Leipzig"),
  "Leverkusen" = c("Bayer 04 Leverkusen", "Leverkusen"),
)

```

If for some month we had missing values, the ratings for that month were imputed using the rating values of the nearby month, so as to ensure there is not much variation in the team's form. 

Next , Web - Scraping was done to extract pre- match reports from **WhoScored.com**. This will give us experts' opinion and will help us understand whether including experts' opinion help complement predictions generated through statistical features alone. The reports scraped contained both Team News and Experts' Pre-Match Predictions. Again , the team names had to be standardized to maintain uniformity. 

We finally merged the contents together and cam up with <span style="color: green;">Master Dataset</span>. Some issues ecountered during this process :-
- Team names - Custom mapping had to be done to maintain uniformity in teams' naming.
- Run time - The enire web scraping process took more than <span style="color: red;"> **2 days**</span> to complete on a machine with standard specifications. it is highly recommended to either run the process using parallel computation or run the process on a machine with higher computing power. However, datasets have already been created and could be used directly for running the forthcoming. The dataset prepared after web scraping and feature engineering ( discussed in length below) is <span style="color: green;"> **Master Dataset**</span>, details of which could be found [here](https://github.com/ACM40960/project-DeepankarVyas/tree/main/Dataset).

#### Feature Engineering

1. Optimal split value derived from MAE.
2. Simulate EPL table using various approaches: 
  - GLM and Poisson-based Monte Carlo: Utilize Poisson regression through R's 'glm' function to simulate standings, calculating attack, defense strengths, and home advantage. 
  - Manual team data reduction for strengths: Compute strength based on average goals, ignoring nuances like opponent quality. 
  - PCA for team strength and table prediction: PCA captures key statistics using two components, representing over 95% variability, and integrates home advantage. Determine home advantage through cross-validation
  - Factor Analysis gave inconclusive results due to inconducive data dimensions.
3. Procrustes Algorithm compares non-metric MDS values from models.
4. Choose best model and predict odds/probabilities.
5. Compare predicted odds with actual organization-provided odds.
6. Translate odds to monetary terms, ensuring the house profits.

### Machine Learning Algorithms

- Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in machine learning to transform high-dimensional data into a lower-dimensional space while preserving as much variance as possible. In the project, PCA has been used to extract meaningful features such as the team strengths, incorporating the home team advantage, from sports statistics data. This has in turn been used to simulate the league standings, and the odds for a team to win a game. It can be seen that the first two principal components reflect more than 95% variability in the data. 

```R
team_data <- as.data.frame(t(Table285[, c("HF", "HA", "AF", "AA")]))# Standardize the data before performing PCA
# Perform PCA
pca_result <- prcomp(scale(team_data), center = TRUE, scale. = TRUE)
```

- Generalized Linear Models (GLMs) can play a significant role in this project by offering a versatile framework for predicting sports outcomes based on various factors. GLMs extend linear regression to accommodate non-normally distributed response variables, making them suitable for modeling binary outcomes like win/loss in sports. In the context of sports statistics, a GLM can be tailored to estimate the probabilities of different match outcomes by considering input features such as team strengths, home advantage, and previous performance. By applying appropriate link functions and distribution assumptions, GLMs can generate outcome probabilities and simulate league standings. Comparing the predicted standings with actual outcomes allows evaluation of the model's performance, aiding in the selection of the most effective predictive method.

```R
  parameters <- glm(formula = Y ~ 0 + XX, family = poisson)
# In the parameters function
```

- In addition, the project incorporates manual functions and the Poisson distribution to compute critical factors such as attack and defense strengths, along with the home advantage for each team. By analyzing historical data, these functions assess the average goals scored and conceded by each team. The attack strength is derived from the difference between average goals scored and conceded, while defense strength stems from the contrary difference. These calculated strengths form the basis for predicting match outcomes using the Poisson distribution. The λ (lambda) parameters, representing expected goal counts, are adjusted to include the home advantage, further enhancing the model's predictive accuracy.
  
```R
# Calculate Attack and Defence Strength for each team
team_data$Attack <- sapply(team_data$Team, function(team) {
  # Calculate average goals scored and conceded for the team
  avg_goals_scored <- mean(c(data_subset$FTHG[data_subset$HomeTeam == team], data_subset$FTAG[data_subset$AwayTeam == team]))
  avg_goals_conceded <- mean(c(data_subset$FTAG[data_subset$HomeTeam == team], data_subset$FTHG[data_subset$AwayTeam == team]))
  return(avg_goals_scored - avg_goals_conceded)
})
team_data$Defence <- sapply(team_data$Team, function(team) {
  # Calculate average goals scored and conceded for the team
  avg_goals_scored <- mean(c(data_subset$FTHG[data_subset$HomeTeam == team], data_subset$FTAG[data_subset$AwayTeam == team]))
  avg_goals_conceded <- mean(c(data_subset$FTAG[data_subset$HomeTeam == team], data_subset$FTHG[data_subset$AwayTeam == team]))
  return(avg_goals_conceded - avg_goals_scored)
})
```

- In addition to Principal Component Analysis (PCA), the project also employs Non-Metric Multidimensional Scaling (MDS) for visualizing teams in a 2D space while preserving their relative ranks. MDS is a technique that aims to represent high-dimensional data in a lower-dimensional space, often for visualization purposes. Non-Metric MDS is utilized to map team data into a 2D space, allowing for an intuitive visualization of team relationships. This technique retains the relative differences between teams while projecting them onto a 2D plane, providing insights into team clusters, similarities, and disparities.
  
```R
# Non-Metric MDS for 2D visualization
library(MASS)
loc = isoMDS(dist(SimTable_actual), k=2, eig=TRUE)
```

### Results

- The project utilizes Procrustes analysis to compare the results obtained from different models. Procrustes analysis is a technique that aligns two sets of data points to best match their structures. In the context of this project, Procrustes analysis is used to align the MDS representations of different models, enabling a quantitative comparison of their predictions and visualizations in the 2D space. The scatter plot below visualizes the results of a Procrustes analysis performed on two sets of Non-metric Multidimensional scaled data of the actual standing and pca standing table. The blue points correspond to the aligned data points from the "Actual" set, while the red dashed line indicates the 1:1 reference line, highlighting how well the alignment matches the original data.

```R
procrustes(loc$points, loc2$points)
```

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="procustes" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/procustes.svg">
   <p><em>Figure 5: Comparing PCA-derived points and actual league table points using Procrustes Analysis</em></p>
</div>

- Furthemore, MAE and MAPE has been employed to compare the different models. MAE calculates the average absolute difference between each team's position in the actual standings and the corresponding position in the simulated standings. This metric provides an overall measure of positional accuracy. Furthermore, MAPE calculates the average percentage difference between each team's position in the actual standings and the corresponding position in the simulated standings. This metric provides insights into the relative accuracy of positional predictions.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="standings" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/team_points_barplot1.svg">
   <p><em>Figure 6: Multiple barchart depicting comparison of team points using different prediction methods </em></p>
</div>

## Model Evaluation Metrics

| Metric          | GLM         | PCA         | Formula     |
|-----------------|-------------|-------------|-------------|
| MAE Score       | 1.9000000   | 3.8000000   | 2.7500000   |
| MAPE Score      | 4.5786319   | 8.0036573   | 5.4346673   |
| Procrustes Score| 3970.1353674| 3086.6280064| 5705.0707390|
| Correlation Score| 0.9992478  | 0.9992478   | 0.9984951   |


- Based on the results table, GLM has been identified as the preferred method to pursue further investigation and development in odds and betting.

### Betting Insights

- Discuss how the betting insights are generated by comparing predicted outcomes with real-time odds.
  The Monte Carlo simulation approach is employed to calculate probabilities for different outcomes (HomeWin, Draw, AwayWin) for various match outcomes between teams, and set into a dataframe. This display acts as a preliminary assessment of the model's accuracy and is particularly useful for gaining insights into its strengths and areas of improvement at a glance.

- The dataframe is then merged with additional home, draw and away winning odds data obtained from betting organizations, like B365H and IW. All these odds are then scaled/standardized, to bring the predicted probabilities and real-time odds to a common scale for accurate correlation calculations. Then correlation coefficients are calculated to quantify the strength and direction of the linear relationship between the predicted probabilities and the real-time odds (by Bet365 and IW) for different outcomes (HomeWin, Draw, AwayWin). 

- Spearman's Rank Correlation Coefficient is useful to compare the two columns as their relationship follows a monotonic pattern, enabling assessment of non-linear connections and ordinal data comparisons. The presented results below indicate a positive correlation (0.8) between our predicted odds and the actual odds, suggesting a favorable alignment between our predictive model and the real-time odds.


| No. | Comparison        | Correlation |
|----:|-------------------|------------:|
|   1 | Bet365 vs Homewin |     0.8109 |
|   2 | Bet365 vs Awaywin |     0.8465 |
|   3 | Average Bet365    |     0.8287 |
|   4 | IW vs Homewin     |     0.8032 |
|   5 | IW vs Awaywin     |     0.8463 |
|   6 | Average IW        |     0.8247 |

The provided correlation values shed light on the alignment between the predicted probabilities and the odds from these betting platforms (ie Bet365 and Interwetten) for different scenarios. Notably, the correlations between Bet365 odds and outcomes for the team to win in home and away ground (Homewin: 0.8109, Awaywin: 0.8465, Average: 0.8287), as well as the correlations between IW odds and outcomes (Homewin: 0.8032, Awaywin: 0.8463, Average: 0.8247), showcase the consistency between our model's predictions and the real-time odds. This observation underscores the potential utility of our approach for informed betting decisions.
  
- Ensuring house profits:

To achieve this, a simulation-based approach is used to analyze the potential earnings and outcomes of a betting strategy applied to football match results. A function is used that generates simulated betting outcomes based on given odds and a specified number of bets. It adds randomness from both normal and uniform distributions to the initial bets and calculates the resulting betting values and the total earnings, which are made to be exponential to the number of bets made. As seen from the plot below, the total house earnings increase as the number of bets made increases.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="moneyplot" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/moneyplot2.svg">
   <p><em>Figure 7: Relationship between the number of bets per match and total house earnings.</em></p>
</div>

### Future Prospects

The current simulation-based approach, inspired by the project, holds exciting prospects for future applications. It can be extended to different sports leagues, seasons, and even esports. Enhancements can involve integrating real-time data for a more realistic house earnings model and comparing a variety of machine learning models like Random Forests and Neural Networks for improved predictions. This expansion could greatly amplify the methodology's versatility, accuracy, and relevance for sports analytics and the betting industry.

### Conclusion

In conclusion, this project demonstrates a holistic approach that combines advanced statistical modeling, machine learning techniques, and real-time odds comparison to accurately predict sports outcomes, enabling informed betting decisions while ensuring the sustainability of house profits, which can be replicated across various sports leagues and seasons.

### Contributing

We welcome contributions to the project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-new-feature`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-new-feature`
5. Open a pull request detailing your changes.

### License

This project is licensed under the MIT License. Please have a look at the [LICENSE.md](https://github.com/ACM40960/project-shubidiwoop/blob/main/LICENSE) for more details.

### References

- James Adam Gardner, Modeling and Simulating Football Results, 2011.
- J. F. C. Kingman. Poisson Processes. Oxford University Press, 1993.
- Rue H, Salvesen O. 2000 Prediction and retrospective analysis of soccer matches in a league. J. R. Stat. Soc. Ser. D (Stat.) 49, 399-418.
- Source: UCD 2023 Spring Multivariate Analysis coursework

### Data Sources

- Historical sports statistics: Source the relevant historical data for the sports leagues of interest. In the case of the English premier league, the data has been sourced [here](https://www.football-data.co.uk/englandm.php)
- Real-time odds data: Integrated with APIs or scraping tools to retrieve real-time odds data from betting platforms such as Bet365, Blue Square, Bet&Win etc. Data structure as been described [here](https://www.football-data.co.uk/notes.txt)

---
### Authors

Feel free to reach out to the project maintainers for any questions or clarifications.

- [Shubham Sharma](https://github.com/shubidiwoop)
- [Vishalkrishna Bhosle](https://github.com/vishalkrishnablaze)

**Disclaimer:** This project is for educational and informational purposes only. Betting and gambling carry risks, and this project does not provide financial advice. Always gamble responsibly.

---

