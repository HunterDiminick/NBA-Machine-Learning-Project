#score predictions
library(ggplot2)
library(corrplot)
library(readr)
library(dplyr)
library(fastDummies)
library(caret)
library(ranger)
library(rsample)
library(randomForest)
library(tidyverse)

#This file presents a method of building multiple regression Random Forest 
#models in order to predict the home and away point totals in NBA matchups. 
#Combining the point-total predictions made by each model enables us to predict
#which team will win each game with an accuracy around 70%. 


#nba games from October 29 2022 through April 2, 2023
data1 <- read_csv("NBA_2223_Data.csv")
colnames(data1) <- gsub(" ", "_", colnames(data1))
data1 <- na.omit(data1)

set.seed(123)
n_rows<- nrow(data1)
train_idx <- sample(seq_len(n_rows),size = floor(.75*n_rows), replace = FALSE)
train_data <- data1[train_idx, ]
test_data <- data1[-train_idx, ]


#awaypointsdata <- data1 %>% select(-Home_Team,-Away_Team,-Home_Pts,-Home_Win)

#homepointsdata <- data1 %>% select(-Home_Team,-Away_Team,-Visitor_Pts,-Home_Win)
#data1[, -which(names(data1) == "Home_Win")] <- scale(data1[, -which(names(data1) == "Home_Win")])



away_train <- train_data %>% select(-Home_Team,-Away_Team,-Home_Pts,-Home_Win)
away_test <- test_data %>% select(-Home_Team,-Away_Team,-Home_Pts,-Home_Win)

"set.seed(123)
away_split <- initial_split(awaypointsdata,prop = .7)
away_train <- training(away_split)
away_test <- testing(away_split)"


#set seed for reproducibility
set.seed(123)

#baseline random forest
m1 <- randomForest(formula = Visitor_Pts~.,
                   data = away_train)
#plot of ntree versus mse
plot(m1)

#calculate minimum mse for each value of ntree
which.min(m1$mse) #290

sqrt(m1$mse[which.min(m1$mse)]) #average margin error of 11.87124


#create training and validation set 
set.seed(123)

#training data
away_train_v2 <- away_train

#validation data
away_valid <- away_test
x_test <- away_valid[setdiff(names(away_valid), "Visitor_Pts")]
y_test <- away_valid$Visitor_Pts

#build model
set.seed(123)
rf_oob_comp <- randomForest(
  formula = Visitor_Pts~.,
  data = away_train_v2,
  xtest = x_test,
  ytest = y_test
)

#get OOB and validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous() +
  xlab("Number of trees")

#tuning random forest
features <- setdiff(names(away_train), "Visitor_Pts")

set.seed(123)
m2 <- tuneRF(
  x = away_train[features],
  y = away_train$Visitor_Pts,
  ntreeTry = 500,
  mtryStart = 5,
  stepFactor = 1.5,
  improve = 0.005,
  trace = FALSE
)
#optimal mtry = 10

set.seed(123)
system.time(
  away_ranger <- ranger(
    formula = Visitor_Pts~.,
    data = away_train,
    num.trees = 500,
    mtry = 5,
    importance = 'impurity'
  )
)
#plot of ranger model's variable importances

away_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 Most Important Variables in Predicting Away Points")+
  xlab("Metric") + ylab("Importance")

#parameters to search over, mtry, ntree, node size, sample size
hyper_grid <- expand.grid(
  mtry = seq(5,20,by=3),
  ntree = seq(400,500,by=50),
  node_size = seq(3,9,by=2),
  sample_size = c(.55, .632, .70, .80),
  OOB_RMSE = 0
)

for(i in 1:nrow(hyper_grid)){
  model <- ranger(
    formula = Visitor_Pts~.,
    data = away_train,
    num.trees = hyper_grid$ntree[i],
    mtry = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sample_size[i],
    seed            = 123
  )
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>%
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

plot(hyper_grid$mtry,hyper_grid$OOB_RMSE)
#plugging in the optimal tuning parameters that we found 
#to create our optimal ranger model

set.seed(123)
OOB_RMSE <- vector(mode = "numeric", length = 100)
for(i in seq_along(OOB_RMSE)){
  optimal_away_ranger <- ranger(
    formula = Visitor_Pts~.,
    data = away_train, 
    num.trees = 400,
    mtry = 20,
    min.node.size = 9,
    sample.fraction = .632,
    importance = 'impurity'
  )
  OOB_RMSE[i] <- sqrt(optimal_away_ranger$prediction.error)
}

#This is a historgram of OOB_RMSE, where we can get a better feel for where
#our actual OOB_RMSE will fall
hist(OOB_RMSE,breaks = 20)

#variable importances from our best ranger model

optimal_away_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 Most Important Variables in Predicting Away Points") + 
  xlab("Metric") + ylab("Importance (Unscaled)")

set.seed(123)
away_test_preds <- predict(away_ranger,data = away_test)
away_test_preds <- away_test_preds$predictions

actual_away_pts <- away_test$Visitor_Pts

away_vals <- tibble(away_test_preds,actual_away_pts)
away_differences <- away_vals$away_test_preds - away_vals$actual_away_pts

summary(away_differences)


#-----------------------------------------------
#build home points model

#set seed for reproducibility
home_train <- train_data %>% select(-Home_Team,-Away_Team,-Visitor_Pts,-Home_Win)
home_test <- test_data %>% select(-Home_Team,-Away_Team,-Visitor_Pts,-Home_Win)
set.seed(123)

#baseline random forest
m1 <- randomForest(formula = Home_Pts~.,
                   data = home_train)
#plot of ntree versus mse
plot(m1)

#calculate minimum mse for each value of ntree
which.min(m1$mse) #290

sqrt(m1$mse[which.min(m1$mse)]) #average margin error of 11.87124


#create training and validation set 


#training data
home_train_v2 <- home_train

#validation data
home_valid <- home_test
x_test <- home_valid[setdiff(names(home_valid), "Home_Pts")]
y_test <- home_valid$Home_Pts

#build model
set.seed(123)
rf_oob_comp <- randomForest(
  formula = Home_Pts~.,
  data = home_train_v2,
  xtest = x_test,
  ytest = y_test
)

#get OOB and validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous() +
  xlab("Number of trees")

#tuning random forest
features <- setdiff(names(home_train), "Home_Pts")

set.seed(123)

m2 <- tuneRF(
  x = home_train[features],
  y = home_train$Home_Pts,
  ntreeTry = 500,
  mtryStart = 10,
  stepFactor = 1.5,
  improve = 0.005,
  trace = FALSE
)
#optimal mtry = 15

set.seed(123)
system.time(
  home_ranger <- ranger(
    formula = Home_Pts~.,
    data = home_train,
    num.trees = 500,
    mtry = 7,
    importance = 'impurity'
  )
)
#plot of ranger model's variable importances

home_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 Most Important Variables in Predicting Home Points") + 
  xlab("Metric") + ylab("Importance (Unscaled)")

#parameters to search over, mtry, ntree, node size, sample size
hyper_grid <- expand.grid(
  mtry = seq(10,22,by=3),
  ntree = seq(400,500,by=50),
  node_size = seq(3,9,by=2),
  sample_size = c(.55, .632, .70, .80),
  OOB_RMSE = 0
)


for(i in 1:nrow(hyper_grid)){
  model <- ranger(
    formula = Home_Pts~.,
    data = home_train,
    num.trees = hyper_grid$ntree[i],
    mtry = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sample_size[i],
    seed            = 123
  )
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>%
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

plot(hyper_grid$mtry,hyper_grid$OOB_RMSE)
#plugging in the optimal tuning parameters that we found 
#to create our optimal ranger model

set.seed(123)
OOB_RMSE <- vector(mode = "numeric", length = 100)
for(i in seq_along(OOB_RMSE)){
  optimal_home_ranger <- ranger(
    formula = Home_Pts~.,
    data = home_train, 
    num.trees = 450,
    mtry = 10,
    min.node.size = 9,
    sample.fraction = .632,
    importance = 'impurity'
  )
  OOB_RMSE[i] <- sqrt(optimal_home_ranger$prediction.error)
}

#This is a historgram of OOB_RMSE, where we can get a better feel for where
#our actual OOB_RMSE will fall
hist(OOB_RMSE,breaks = 20)

#variable importances from our best ranger model
optimal_home_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 Most Important Variables in Predicting Home Points") + 
  xlab("Metric") + ylab("Importance (Unscaled)")

#Making predictions for the home team's number of points
test_preds <- predict(home_ranger,data = home_test)
home_test_preds <- test_preds$predictions

#actual points scored by the home team
actual_home_pts <- home_test$Home_Pts

#seeing how far off the model was in predicting the home team's points 
#for each game. 
home_vals <- tibble(home_test_preds,actual_home_pts)
home_differences <- home_vals$home_test_preds - home_vals$actual_home_pts
summary(home_differences)


#this is the concatenation of the two models. By predicting both the home and 
#away team's points, we can make a prediction for which team will win the 
#game. 
frame <- tibble(home_test_preds,away_test_preds)
differences <- frame$home_test_preds - frame$away_test_preds
pred_winners <- ifelse(differences>0,1,0)
act_winners <- test_data$Home_Win

win_tab <- tibble(pred_winners,act_winners)
table <- table(win_tab$pred_winners,win_tab$act_winners)
#69.34 overall
#
#-------------------------------------------------------------
#sampling the same random rows, and making predictions on each

diff_pts <- data1$Home_Pts - data1$Visitor_Pts
diff_elo <- data1$Home_ELO - data1$Away_ELO

# Create a data frame with the differences
diff_df <- data.frame(Diff_Pts = diff_pts, Diff_ELO = diff_elo)
df2_neg <- diff_df %>% filter(Diff_ELO<0)
df2_pos <- diff_df %>% filter(Diff_ELO >=0)

# Scatter plot with regression line for df2_neg
ggplot(df2_neg, aes(x = Diff_ELO, y = Diff_Pts)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(x = "Home ELO - Away ELO", y = "Home Pts - Away Pts") +
  ggtitle("Difference in ELO vs Scoring Margin (Away ELO > Home ELO)")

# Scatter plot with regression line for df2_pos
ggplot(df2_pos, aes(x = Diff_ELO, y = Diff_Pts)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(x = "Home ELO - Away ELO", y = "Home Pts - Away Pts") +
  ggtitle("Difference in ELO vs Scoring Margin (Away ELO < Home ELO)")

#This plot demonstrates how ELO can be seen as an effective predictor for the 
#results of NBA games. 

ggplot(diff_df, aes(x = Diff_ELO, y = Diff_Pts)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(x = "Home ELO - Away ELO", y = "Home Pts - Away Pts") +
  ggtitle("Difference in ELO vs Scoring Margin")
