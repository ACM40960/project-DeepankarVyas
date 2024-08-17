# Feature Engineering

A detailed list of the features engineered, along with the equations used , is described here :-

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

This directory contains both the .R and .qmd files, and  either can be run to engineer the features as outlined above. The code reads the Web scraped content from $${\color{green}Master \space \color{green}Dataset}$$, engineer the features using equations discussed above and overwrites the same dataset at the end.
