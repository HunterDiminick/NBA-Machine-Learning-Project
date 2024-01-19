# NBA Game Outcome Predictive Models in R

This repository contains R scripts that aim to predict NBA game outcomes using various statistical models and machine learning techniques. The predictions are based on historical game data from the 2022-2023 NBA season, specifically utilizing averages from each team's three prior games, either away or at home.

## Data Preparation

The dataset used for training these models was created by scraping game data for all 1230 NBA regular season games of the 2022-2023 season. To ensure data quality and consistency, a Python script was employed to preprocess and aggregate this data.

Notably, during the data preprocessing phase, 136 games from the beginning and end of the season were excluded. These exclusions were made due to irregularities observed during these periods. Teams that had already secured playoff berths might not exert maximum effort, while teams that were out of playoff contention might be strategically "tanking." To eliminate the potential impact of such factors, these games were removed from the dataset.

## Model Descriptions

### Model 1: Basic Classification (Basic_Classification.R)

This model employs a Random Forest classification algorithm to predict whether the home team will win (1) or lose (0) a given NBA game. It is designed to provide a binary outcome prediction.

### Model 2: Margin of Victory Regression (Regression_Margin_Victory.R)

In this model, a new field called "Margin" is introduced during preprocessing. The model utilizes a Random Forest regression approach to predict the margin of victory relative to the home team. By translating these margin predictions into simple win-loss outcomes, the model's accuracy can be evaluated.

### Model 3: Point Total Regression (Regression_Point_Total.R)

This model combines two distinct regression Random Forest models, each dedicated to predicting the total points scored by either the home or away team. After training both models on the same test dataset, they make predictions for home and away team points in each game. These predictions are then used to determine a win-loss category relative to the home team, and the model's accuracy is assessed based on the actual game outcomes.

## Results and Future Considerations

Among the three models evaluated, Model 2 emerged as the top performer overall. The challenge in comparing these models lies in the diverse nature of their tasks; Model 1, a classification model, operates on a different scale than the other two, measured by out-of-bounds RMSE.

When focusing on predicting home or away wins, Model 2 showcased remarkable accuracy. It successfully predicted outcomes with a precision of 71.9% for home team victories and 68.75% for away team victories. This balanced accuracy across both scenarios indicates that the model is not overfit to predict specific outcomes.

In addition to accuracy, exploring metrics like precision, recall, or F1 score could provide a more nuanced understanding of each model's performance. This context is crucial for appreciating the practical implications of their predictive capabilities when deployed against upcoming matchups and unseen scenarios.

Feel free to explore these scripts to gain insights into the predictive capabilities of each model and adapt them to your specific requirements. Keep in mind that accuracies may vary to the stochastic nature of Random Forest models.
