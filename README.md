# NBA-Machine-Learning-Project
Repository with various methods attempting to predict NBA winners in R.
Important to note, the data that I created these models with was averages over each team's three prior games, either away
or at home. For example, if the Lakers were playing away at Detroit, the Lakers stats were their averages over their last 
three away matchups, while the Pistons stats were their averages over their last three home games. Calculating these averages
involved scraping the game data for each game of the 2022-2023 season, and running a python script that produced a new 
dataset. 

Another important thing to note, the NBA regular season has 1230 games. In the preprocessing of the data I deleted 136 games 
from the start and end of the season, given irrecularities found during these periods. At the end of the season, some teams
have already made the playoffs and therefore aren't incentivized to try as hard, while on the other hand some teams are 
out of contention and tanking. I had no way to quantify "effort", so I decided to eliminate games from these two periods. 

I uploaded three different files, each with the same objective but utilizing various methods. The first model, 
Basic_Classification.R, attempts to classify whether the home team will win (1) or lose (0) using a random forest classification
model. The second model, labeled Regression_Margin_Victory.R, creates a new field in preprocessing called Margin, which is the 
response variable that the model makes predictions on. The random forest model created makes predictions on the margin of victory
relative to the home team, from which we can translate into a simple winner, and determine the model accuracy. The last file, 
Regression_Point_Total.R, is the aggregation of two different regression Random Forest models, each of which attempt to predict
the total points for the home or away team. After training each of the models on the same set of test data, the model makes 
predictions on the home and away points for each game. Using these predictions, I then created a simple win-loss category
relative to the home team, and compared it to who won each matchup to determine the accuracy of the model. 
