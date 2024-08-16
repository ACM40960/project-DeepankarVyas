# Datasets

A detailed description of the datasets can be found below :-

## Master_Dataset 
The dataset created after Web Scraping, Feature engineering and Sentiment Analysis. The columns and a brief description is as follows :-
- Season :- Season in which match was played. (Autumn, Spring, Summer, Winter)
- FTR :- Full time Result (H - Home Win, A - Away Win, D - Draw)
- B36H :- Bet365 odds for home win
- B365A :- Bet365 odds for away win
- B365D :- Bet365 odds for draw
- Home_Attack :- Home team's rating for attack
- Home_Midfield :- Home team's rating for midfield
- Home_Defense :- Home team's rating for defense
- Home_Overall :- Home team's overall rating
- Away_Attack :- Away team's rating for attack
- Away_Midfield :- Away team's rating for midfield
- Away_Defense :- Away team's rating for defense
- Away_Overall :- Away team's overall rating
- Preview :- Pre - match report of the match
- Day_of_Week :- Day of the week on which the match was played
- HGKPP :- Home teams' past 5 matches Goals
- AGKPP :- Away teams' past 5 matches Goals
- HCKPP :- Home teams' past 5 matches Corners
- ACKPP :- Away teams' past 5 matches Corners
- HSTKPP :- Home teams' past 5 matches Shots on Targets
- ASTKPP :- Away teams' past 5 matches Shots on Targets
- GKPP :- HGKPP - AGKPP
- CKPP :- HCKPP - ACKPP
- STKPP :- HSTKPP - ASTKPP
- HForm :- Home teams' Form
- AForm :- Away teams' Form
- Form :- HForm - AForm 
- HSt :- Home teams' Streak of the past 5 matches
- ASt :- Away teams' Streak of the past 5 matches
- HStWeighted :- Home teams' Weighted Streak of the past 5 matches
- AStWeighted :- Away teams' Weighted Streak of the past 5 matches
- Streak :- HSt - ASt
- WeightedStreak :- HStWeighted - AStWeighted
- HTGD :- Home teams' past 5 matches Goal Difference
- ATGD :- Away teams' past 5 matches Goal Difference
- GD :- HTGD - ATGD
- AttDiff :- Home_Attack - Away_Attack
- MidDiff :- Home_Midfield - Away_Midfield
- DefDiff :- Home_Defense - Away_Defense
- OverallDiff :- Home_Overall - Away_Overall
- Home_Score :- Home team's sentiment score based on pre - match report
- Away_Score :- Away team's sentiment score based on pre - match report
- Diff_Score :- Home_Score - Away_Score


## Master_Dataset_Before_Prepocessing

This is the dataset created after Web Scraping and Feature Engineering . This is mainly used for internal purposes. Since websites could block web scraping anytime, this dataset is used to bypass the web scraping process and jump directly to Feature engineering process. It containns the columns outlined above and some additional column, which won't be used for analysis.

## Master_Dataset_Final

This is the dataset created after EDA and Feature Selection, the final dataset before Model training commences. It contains the same columns as `Master_Dataset`, with a few redundant columns removed.

---

Rest of the datasets are pretty self-explanatory. These are the 4 different datasets under consideration 

- Class A :- Home and Away features of the teams
- Class A NLP :- Home and Away features of the teams with Sentiment Scores
- Class B :- Differential features of the teams
- Class B NLP :- Differential features of the teams with Sentiment Scores

divided into train and test datasets (80%-train and 20%- test)
