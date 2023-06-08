library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)
library(readr)
library(tidyverse)
library(dplyr)

#nba games from October 29 2022 through April 2, 2023


#organizing data, creating margin category
data <- read_csv("NBA_2223_Data.csv")
colnames(data) <- gsub(" ", "_", colnames(data))
margin <- data$Home_Pts - data$Visitor_Pts
margin <-tibble(margin)
final_data <- tibble(margin,data)
final_data <- final_data %>% select(-Home_Pts,-Visitor_Pts,-Home_Win)


#create dummy columns for the home and away team, results in additional 60 columns

#drop home and away team columns


#clean data
clean_data <- na.omit(final_data)

#creating test and training splits
game_split <- initial_split(clean_data,prop = .8)
game_train <- training(game_split)
game_test <- testing(game_split)

#set seed for reproducibility
set.seed(123)

#baseline random forest
m1 <- randomForest(formula = margin~.,
                   data = game_train)
#plot of ntree versus mse
plot(m1)

#calculate minimum mse for each value of ntree
which.min(m1$mse) #298

sqrt(m1$mse[which.min(m1$mse)]) #average margin error of 12.15


#create training and validation set 
set.seed(123)

#training data
game_train_v2 <- game_train

#validation data
game_valid <- game_test
x_test <- game_valid[setdiff(names(game_valid), "margin")]
y_test <- game_valid$margin

#build model
rf_oob_comp <- randomForest(
  formula = margin~.,
  data = game_train_v2,
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
features <- setdiff(names(game_train), "margin")

set.seed(123)

m2 <- tuneRF(
  x = game_train[features],
  y = game_train$margin,
  ntreeTry = 300,
  mtryStart = 25,
  stepFactor = 1.5,
  improve = 0.01,
  trace = FALSE
)
#best mtry is 25

system.time(
  game_randomForest <- randomForest(
    formula = margin~.,
    data = game_train,
    ntree = 300,
    mtry = 25
    #mtry = floor(length(features) / 3
  )
)


#demonstration of speed of the ranger package as opposed to random_forest 
#default package

system.time(
  game_ranger <- ranger(
    formula = margin~.,
    data = game_train,
    num.trees = 300,
    mtry = 25,
    importance = 'impurity'
  )
)

#plot of ranger model's variable importances
game_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 Most Important Variables") + 
  xlab("Feature") + ylab("Importance (Unscaled)")

#parameters to search over, mtry, ntree, node size, sample size
hyper_grid <- expand.grid(
  mtry = seq(25,35,by=2),
  ntree = seq(300,500,by=50),
  node_size = seq(3,9,by=2),
  #sample_size = .75,
  OOB_RMSE = 0
)

set.seed(123)
for(i in 1:nrow(hyper_grid)){
  model <- ranger(
    formula = margin~.,
    data = game_train,
    num.trees = hyper_grid$ntree[i],
    mtry = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = .75,
    seed            = 123
  )
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>%
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

#plugging in the optimal tuning parameters that we found 
#to create our optimal ranger model

set.seed(123)
OOB_RMSE <- vector(mode = "numeric", length = 100)
for(i in seq_along(OOB_RMSE)){
  optimal_ranger <- ranger(
    formula = margin~.,
    data = game_train, 
    num.trees = 400,
    mtry = 27,
    min.node.size = 7,
    sample.fraction = .75,
    importance = 'impurity'
  )
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

#This is a historgram of OOB_RMSE, repeated over 100 different trials, 
#where we can get a better feel for where our actual OOB_RMSE will fall
hist(OOB_RMSE,breaks = 20)

#variable importances from our best ranger model
optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 Most Important Variables") + 
  xlab("Feature") + ylab("Importance(Unscaled)")


#Making predictions on the testing set with all three models
set.seed(123)
pred_randomForest <- predict(game_randomForest,game_test)
randomForestPrediction <- pred_randomForest

set.seed(123)
pred_ranger <- predict(optimal_ranger, game_test)
rangerPrediction <- pred_ranger$predictions
predictions_all <- tibble(randomForestPrediction,rangerPrediction)


actual <- game_test$margin
actual <- ifelse(actual>0,1,0)

rangerWin <- ifelse(predictions_all$rangerPrediction>0,1,0)
forestWin <- ifelse(predictions_all$randomForestPrediction>0,1,0)

#creation of a confusion matrix with the actual results and model predictions
ranger_table <- tibble(rangerWin,actual)
forest_table <- tibble(forestWin,actual)

#viewing of confusion matrix
ranger_results <- table(ranger_table)
#ranger: 64.38% accurate. Not great considering the home team wins in about 
#60% of all home games. 
forest_results <- table(forest_table)
#forest: 67% accurate. Again, not great. 
