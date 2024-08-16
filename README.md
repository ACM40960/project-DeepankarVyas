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
  - [Sentiment Analysis and Data Cleanup](#sentiment-analysis-and-data-cleanup)
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

---

### Motivation

The objective of this project is to develop a model capable of accurately predicting the outcomes of football matches. This model aims to challenge existing betting models by leveraging a unique combination of engineered features, derived from domain-specific knowledge, and sentiment analysis of pre-match reports. By integrating and enhancing previous approaches—some of which focus solely on statistical features while others emphasize text analytics—this project seeks to determine whether a synergistic combination of these methodologies can yield superior predictive results.

---

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
---

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
---

### Methodology
#### Dataset Preparation

The dataset was prepared using Web - Scraping . Base Features and Betting Odds were taken from [Football-Data.co.uk](https://www.football-data.co.uk/germanym.php) [6] , Team Ratings were taken from [FIFA Index](https://www.fifaindex.com/teams/) [5]  and Pre - Match Reports were scraped from  - [WhoScored.com](https://www.whoscored.com/Regions/81/Tournaments/3/Germany-Bundesliga) [8] . Everything was collated to prepare a $${\color{green}Master \space \color{green}Dataset}$$.

The code to prepare the dataset using Web Scraping can be found here - [Dataset Preparation](https://github.com/ACM40960/project-DeepankarVyas/tree/main/dataset-preparation). This directory contains the .R file, and  can be run to Web Scrap the contents. First, the files for the past 10 Bundesliga seasons is downloaded from **Football-Data.co.uk**. Then, we web scrapped team's ratings from **FIFA Index** website staring from FIFA 12 till FIFA 24. These are the ratings given by FIFA video game to teams and players and are generated periodically, taking into account current form of teams and players, and thus are a good measure of the current state of the team. We focused on the team ratings and web scrapped following components of team ratings-
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

We finally merged the contents together and cam up with $${\color{green}Master \space \color{green}Dataset}$$. Some issues ecountered during this process :-
- Team names - Custom mapping had to be done to maintain uniformity in teams' naming.
- Run time - The enire web scraping process took more than $${\color{red}5 \space \color{red}days}$$ to complete on a machine with standard specifications. it is highly recommended to either run the process using parallel computation or run the process on a machine with higher computing power. However, datasets have already been created and could be used directly for running the forthcoming. The dataset prepared after web scraping , feature engineering ( discussed in length below) and sentiment analysis ( discussed in length below) is $${\color{green}Master \space \color{green}Dataset}$$, details of which could be found [here](https://github.com/ACM40960/project-DeepankarVyas/tree/main/Dataset).

----

#### Feature Engineering

We have a baseline dataset available , containing information like Goals scored and conceded, Corners, Xg, Shots on Target and Result. However, previous works in this field and our own domain knowledge suggest that the outcome of any sport match is highly unpredictable and various physical, psychological and geographical factors affect the outcome of any match. Therefore, we will engineer most of our features in order to capture as much information as we could to predict the result.

Base Features ,namely - FTHG (Full time Home Team Goals), FTAG (Full time Away Team Goals), HST (Home Team Shots on Target) , AST (Away Team Shots on Target), HC (Home Team Corners) and AC (Away team Corners) are used to engineer the features. Feature engineering was done maily beacuse :-
- To predict a football match, we need to have features available before hand. Therefore, features were engineered so that they depend only on a team's previous matches' statistics.
- Based on Domain knowledge, it was clear that one of the main factors affecting a football match's result is the teams' form. Hence, features were engineered to incorporate this variable.

Wherever applicable, features were engineered so that they depend only on a team's past 5 matches. The number 5 is static and is one of the areas of further research, to make this number dynamic and select it using cross-validation. The process of feature engineering is repeated freash for each season, so that the previous season has no bearing on the current season. Since, we are finding features based on a team's performance in the previous 5 matches, no values were assigned to features of a team for the first 5 matches. The first 5 matches of each team were thus removed from analysis. Since Home and Away match is a simple but an extremely important criterion, all these features were computed for both the home and away team for every match. A full list of features engineered along with the equations used is given below [1,3] :-

1. **HGKPP , AGKPP** - Home and Away teams' past 5 matches Goals.
2. **HSTKPP , ASTKPP** - Home and Away teams' past 5 matches Shots on Targets.
3. **HCKPP , ACKPP** - Home and Away teams' past 5 matches Corners.

<p align="center">
$\mu_{j}^{i}$ = $\left( \sum_{p=j-k}^{j-1} \mu_{p}^{i} \right) / k$ , where $\mu^{i} \in \{\text{Corners, Shots on Target, Goals}\}$
</p>

4. **HSt , ASt, HStWeighted, AStWeighted** - Home and Away teams' Streak and Weighted Streak of the past 5 matches. This feature encapsulates the recent improving/declining trend in the performance of a team. The Streak value for a team is computed by assigning a score to each match result and taking the mean of the previous 5 scores. We also included a temporal dimension to the Streak feature by placing time- dependent weights on the scores of the previous games of a team, obtaining a feature that we refer to as the Weighted Streak (greater weights for recent games, decreasing grad- ually for non-recent games). In the Weighted Streak feature, the weighting scheme is as follows:
- a weight of 1 on the oldest observation (j − k)
- a weight of 5 on the most recent observation (j−1).

<p align="center">
$\text{Streak}(\delta_j) = \left( \sum_{p=j-k}^{j-1} \text{resp}_p \right) / 3k$ <br><br>
$\text{Weighted\_Streak}(\omega_j) = \sum_{p=j-k}^{j-1} \frac{2(p - (j - k - 1) \text{resp}_p)}{3k(k + 1)}$ , where $\text{resp}_p \in \{0, 1, 3\}$
</p>

5. **HForm , AForm** - Home and Away teams' Form. Much like Streak, it is a measure of team's form but it encompasses a team’s performances in individual matches much more intricately. The form values are updated after every match and take into account the quality of the opposition faced, underlining the importance of both time factor and difficulty of the match. Mathematical formulation of Form ensures that a greater coefficient update is provided if a weak team triumphs over a strong team, and vice-versa. In the case of a draw, the Form of a weak team increases while that of a strong team decreases.

<p align="center">
When team `A` beats team `B` :<br><br>
$F_{j}^{A} = F_{(j-1)}^{A} + \gamma F_{(j-1)}^{B}$<br><br>
$F_{j}^{B} = F_{(j-1)}^{B} - \gamma F_{(j-1)}^{B}$<br><br>
​In case of a draw:<br><br>
$F_{j}^{A} = F_{(j-1)}^{A} - \gamma (F_{(j-1)}^{A} - F_{(j-1)}^{B})$<br><br>
$F_{j}^{B} = F_{(j-1)}^{B} - \gamma (F_{(j-1)}^{B} - F_{(j-1)}^{A})$<br><br>
where $\gamma$ is the stealing fraction and is within the range (0,1), with the optimal value being 0.33 .
</p>

6. **HTGD , ATGD** - Home and Away teams' past 5 matches Goal Difference. The Goal Difference (GD) is crucial in football predictive models. It is calculated as the difference between the cumulative goals scored and goals conceded by a team up to a specific match.

<p align="center">
$GD_k = \sum_{j=1}^{k-1} \text{GS}_j - \sum_{j=1}^{k-1} \text{GC}_j$​<br><br>
where $\text{GS} = \text{Goals Scored}, \text{GC} = \text{Goals Conceded}$ .
</p>

7. Other than the engineered features, we also considered the **Team Ratings** of each team's **Attack, Midfield, Defense and also Overall Rating**, which were web scraped in the previous section.
8. The **Day of the Week** when the match was played and the **Season** were included as Psychological factors[7].

The dataset prepared after web scraping , feature engineering and sentiment analysis ( discussed in length below) is $${\color{green}Master \space \color{green}Dataset}$$, details of which could be found [here](https://github.com/ACM40960/project-DeepankarVyas/tree/main/Dataset).

The code to engineer the features can be found here - [Feature Engineering](https://github.com/ACM40960/project-DeepankarVyas/tree/main/feature-engineering). This directory contains both the .R and .qmd files, and  either can be run to engineer the features as outlined above. The code reads the Web scraped content from $${\color{green}Master \space \color{green}Dataset}$$, engineer the features using equtions discussed above and overwrites the same dataset at the end.

---
#### Sentiment Analysis and Data Cleanup

All the work in the field of Sports Analytics/ Predictions have been done either exclusiveluy using statistical features or using exclusively text analytics. One of the motivations of this project is to use a combination of these features and find out whether this results in improved performance [2]. For this purpose, experts' pre - match analysis was used from **WhoScored.com** and sentiment analysis of each match's pre - match report was done to generate a score for both Home and Away team. Due to resource limitation, full fledged Sentiment Analysis using LLMs could not be achieved. However, R provides a pretty handy package `sentimentr`. The `sentimentr` package bolsters sentiment analysis with a lexicon of words that tend to slide sentiment a bit in one direction or the other. These words are known as valence shifters. It takes into account negations or hyperboles that are generally used in everyday speech. 

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Wordcloud" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/eda-splitting/eda_images/Wordcloud.png">
   <p style="text-align: center;"><em>Figure 1: Wordlcoud generated through Sentiment Analysis.</em></p> 
</div>

However, this process was riddled with issues. Sentiment Analysis of a pre-match report is quite different from then Sentiment Analysis of, say, a novel or even a tweet related to the match . Discussed below are the issues and the proposed solutions for the same :-

1. Two subjects, instead of the usual one - One of the main issues encountered during sentiment analysis of a pre-match report is that the report pertains to two teams, not just one team. Therefore, the sentiments could range from extremely positive, to extremely negative , and could be related to different teams. Therefore, it was quite necessary to associate each sentence to one of the teams, for which it was used, and then perform sentiment analysis for each team, using each team's sentences.
2. Varied names for a team - While writing pre - match reports, full name of the team is rarley used. Author often resorts to using short names or even nicknames of the teams. Therefore, it was necessary to have a standardize naming of the theams, or map each name used to a standard name in order to achieve the point discussed above.
3. Custom Lexicon - Lexicon used by the package is of the English vocabulary. Hoever, sports vocabulary could turn out to be quite different from the one used in a general sense. For eg:- "Attack" in general sense conveys negative emotion. But in footballing terms, it could convey positive emotions , suggesting the attacking mindset of the teams. Therefore, in addition to the lexicon used, custom lexicon was designed which was more focused on the terminology used in football and the associated sentiment score.

The code for the Sentiment Analysis could be found here - [Sentiment Analysis](https://github.com/ACM40960/project-DeepankarVyas/tree/main/data-cleanup-nlp). This directory contains both the .R and .qmd files, and  either can be run to generate the sentiment scores . Major functionalities implemented in this file , covering the points above :-

Team mapping to standardize team names -
```
# Corrected team name mapping
team_name_mapping <- list(
  "Bayern Munich" = c("Bayern Munich", "Bayern München", "FC Bayern München", "FC Bayern Munich", "Bayern"),
  "Dortmund" = c("Dortmund", "Borussia Dortmund"),
  "RB Leipzig" = c("RB Leipzig", "Leipzig", "RBL"),
  "Leverkusen" = c("Leverkusen", "Bayer 04 Leverkusen", "Bayer"),
  "Ein Frankfurt" = c("Ein Frankfurt", "Eintracht Frankfurt", "Frankfurt", "Eintracht"),
  "Hoffenheim" = c("Hoffenheim", "TSG 1899 Hoffenheim", "TSG Hoffenheim", "1899 Hoffenheim"),
  "Wolfsburg" = c("Wolfsburg", "VfL Wolfsburg"),
  "Freiburg" = c("Freiburg", "SC Freiburg", "Sport-Club Freiburg"),
)

# Function to expand team names
expand_team_names <- function(mapping) {
  expanded <- tolower(unlist(mapping, use.names = FALSE))
  team_keys <- tolower(rep(names(mapping), lengths(mapping)))
  expanded_mapping <- setNames(team_keys, expanded)
  return(expanded_mapping)
}

# Expanding the team names from the mapping
expanded_mapping <- expand_team_names(team_name_mapping)

# Function to check if a sentence contains any team name from the mapping
contains_team_name <- function(sentence, team_names) {
  pattern <- paste0("\\b(", paste(unique(unname(team_names)), collapse = "|"), ")\\b")
  return(grepl(pattern, sentence))
}


# Function to associate sentences with teams
associate_sentences_with_teams <- function(text, home_team, away_team, expanded_mapping) {
  text_lower <- tolower(text)
  home_team <- tolower(home_team)
  away_team <- tolower(away_team)
  sentences <- unlist(strsplit(text_lower, split = "\\."))
  
  home_team_names <- names(expanded_mapping[expanded_mapping == home_team])
  away_team_names <- names(expanded_mapping[expanded_mapping == away_team])
  
  associated_sentences <- tibble(sentence = sentences, team = NA_character_)
  current_team <- home_team
  
  for (i in seq_along(sentences)) {
    sentence <- sentences[i]
    if (current_team == home_team && contains_team_name(sentence, away_team_names)) {
      current_team <- away_team
    } else if (current_team == away_team && contains_team_name(sentence, home_team_names)) {
      current_team <- home_team
    }
    associated_sentences$team[i] <- current_team
  }
  
  return(associated_sentences)
}

```

This is a sample list used to standardize team names . The pre - match reposrt is checked to see if it contains any of th two team names between whom match is going to be played from the above names. If it does, that particular sentence gets associated with that team. All other team names in the report, if they occur, are ignored. Once the sentence-team mapping is acheived, `sentiment_by()` function of `sentimentr` package is used to calculate the sentiment scores, for both the teams.

```
# Function to calculate sentence-level sentiment scores
calculate_sentence_sentiment <- function(sentences_with_teams) {
  sentiment_scores <- sentiment_by(sentences_with_teams$sentence, by = list(sentences_with_teams$team)) %>%
    select(team = "sentences_with_teams$team", ave_sentiment)
  
  return(sentiment_scores)
}

```

Apart from this, data cleanup is also done to keep only the columns that will be used for further analysis and also the rows which contain NA values among the selected columns are removed.

---
#### EDA and Feature Selection

Once features have been engineered and sentiment analysis done, next step is Exploratory Data Analysis and Feature Selection. Some of the interesting findings of Exploratory Data Analysis is as follows :-

1. Almost none of our features, whether Base features or Engineered features, follow normal distribution.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Home and Away Features" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/eda-splitting/eda_images/a%20(2).png">
   <p style="text-align: center;"><em>Figure 2: Normality violation of Engineered Features.</em></p> 
</div>

To overcome this, we calculated the difference between Home and Away features , called **Differential Features**, which were comparatively closer to normal distribution

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Differential Features" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/eda-splitting/eda_images/b%20(2).png">
   <p style="text-align: center;"><em>Figure 3: Approximately Normal distribution of Differential Features.</em></p> 
</div>

2. In addition to this, feature selection was done using `Boruta` package of R and through correlation matrix. Feature selection helps reduce complexity of the model by feeding only the relevant features and removing
   correlated and redundant features, thereby making model training faster.
   
- The Boruta algorithm is a wrapper built around the random forest classification algorithm. It tries to capture the important features in a dataset with respect to an outcome variable. It compares the importance of original features with that of shadow features, which are created by randomly shuffling the original features. During the process, Boruta duplicates all the features and shuffles them to create these shadow features, then fits a random forest model. It compares the importance scores of the actual features against these shadow features to determine their relevance. If a feature consistently outperforms the shadow features across multiple iterations, it is considered important and retained; otherwise, it is deemed unimportant and rejected. It can work on both numerical or categorical variables, or even a mixture of both.
- Correlation matrix determines how 2 variables are related linearly. If there is a high correlation between 2 variables, that means one of them can be considered redundant as the presence of one explains the effect of the other.

The figure below shows the output of `Boruta` algorithm and heatmap of `Correlation Matrix` :-

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Feature Selection" src="https://github.com/ACM40960/project-DeepankarVyas/blob/main/eda-splitting/eda_images/image%20(2).png">
   <p style="text-align: center;"><em>Figure 4: Borutal Importance plot and Correlation Matrix.</em></p> 
</div>


As shown in Figure 4 depicting the Correlation Matrix and Boruta Feature Selection, it became clear that the variables are highly correlated to their differential features and features such as - Day_Of_Week, Season are not important. These features were removed and based on the analysis done above , it was decided to divide our dataset into 4 different datasets :-

- Class A :- Home and Away features of the teams
- Class A NLP :- Home and Away features of the teams with Sentiment Scores
- Class B :- Differential features of the teams
- Class B NLP :- Differential features of the teams with Sentiment Scores

The code to perform **EDA and Feature Selection** can be found here - [EDA and Feature Selection](https://github.com/ACM40960/project-DeepankarVyas/tree/main/eda-splitting). This directory contains both the .R and .qmd files, and  either can be run to generate the desired analysis. The file reads from $${\color{green}Master \space \color{green}Dataset}$$ and creates separate train and test datasets for all the 4 datasets discussed above (80%- training and 20%- testing). 

A detailed description of all the datasets used can be found here - [Datasets](https://github.com/ACM40960/project-DeepankarVyas/tree/main/Dataset).

---
#### Model Training and Tuning
