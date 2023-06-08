library(ggplot2)
library(corrplot)
library(readr)
library(fastDummies)
library(tidyverse)
library(ranger)
library(caret)

#This file presents a method for creating a classification model using 
#random forest to predict the winner of NBA regular season matchups. 

#nba games from October 29 2022 through April 2, 2023
data1 <- read_csv("NBA_2223_Data.csv")
colnames(data1) <- gsub(" ", "_", colnames(data1))

data1 <- data1 %>% select(-Home_Team,-Away_Team,-Home_Pts,-Visitor_Pts)
#data1[, -which(names(data1) == "Home_Win")] <- scale(data1[, -which(names(data1) == "Home_Win")])
data1 <- na.omit(data1)


classificationModel <- data1
classificationModel$Home_Win <- as.factor(classificationModel$Home_Win)

set.seed(123)
repeat_cv <- trainControl(method = 'repeatedcv', number=10)

set.seed(123)
train_index <- createDataPartition(y=classificationModel$Home_Win,p=.8,list=FALSE)


training_set <- classificationModel[train_index,]
testing_set <- classificationModel[-train_index,]

set.seed(123)
class_forest <- train(
  Home_Win~.,
  data = training_set,
  method = 'rf',
  trControl = repeat_cv,
  metric = 'mse',
  ntree = 150
)

#Determining optimal ntree
ntree_max <- 600
ntree_range <- seq(100, ntree_max, by = 10)

accuracy <- numeric(length(ntree_range))

for (i in seq_along(ntree_range)){
  model <- randomForest(Home_Win~.,data = training_set, ntree = ntree_range[i])
  predictions <- predict(model, newdata = testing_set)
  accuracy[i] <- mean(predictions == testing_set$Home_Win)
}
plot(ntree_range, accuracy, type = "b", xlab = "Number of Trees", ylab = "Accuracy")
tibble(ntree_range, accuracy) %>% arrange(desc(accuracy))

#--------------------------------------
#variable importances
predictions <- predict(class_forest, newdata = testing_set)
prob_predictions<- predict(class_forest,newdata=testing_set,type="prob")

prediction_named = tibble(Predictions = predictions, Actual = testing_set$Home_Win)
confusion_matrix<- table(prediction_named$Predictions,prediction_named$Actual)

importances <- varImp(class_forest,scale=FALSE)
importance_vector <- importances$importance
feature_names <- names(testing_set)[-1]

# barplot(importance_vector$Overall, 
#         names.arg = feature_names, 
#         xlab = "Features", 
#         ylab = "Importance",
#         main = "Feature Importances",
#         las=2)

tibble(feature = feature_names, importance = importance_vector$Overall) %>%
  mutate(feature = fct_reorder(feature, importance)) %>%
  ggplot( aes(x=feature, y=importance)) +
  geom_bar(stat="identity", fill="#f68060", alpha=.6, width=.4) +
  coord_flip() +
  labs(title = "Built-In", x = "Feature", y="Importance") +
  theme_bw()

  





# data5 <- data1


y_test <- testing_set$Home_Win
x_test <- testing_set %>% select(-Home_Win)

# logLoss <- function(pred,actual){
#   prob_pred <- pred[,2]
#   -mean(actual * log(prob_pred) + (1 - actual) * log(1 - prob_pred))
# }

pred1 <- predict(class_forest, newdata = x_test, type="prob")

# loss <- logLoss(pred1,y_test)
# 
# M <- 10
# loss_diffs <- list()
# for (var in names(x_test)){
#   losses <- numeric(M)
#   for (i in 1:M){
#     X_test_perm <- x_test
#     X_test_perm[[var]] <- sample(X_test_perm[[var]])
#     y_pred_perm <- predict(class_forest,X_test_perm,type="prob")
#     loss_perm <- logLoss(y_pred_perm,y_test)
#     losses[i] <- loss - loss_perm
#   }
#   loss_diffs[[var]] <- losses
# }
# loss_diffs

#found again that the two teams playing are not very significant
# boxplot(loss_diffs, main = "Permutation (After Fitting) Feature Importance", xlab = "Variable", ylab = "Loss Difference")


log_loss <- function(y, p_hat){
  -mean(ifelse(y == 1, log(p_hat), log(1-p_hat))) 
}

loss_base = log_loss(y = y_test, p_hat = pred1[,2])

perm_feat_imp_2 <- function(model, data, var, M=1, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  loss = numeric(M)
  for(i in 1:M){
    data[[var]] = sample(data[[var]])  
    loss[i] = log_loss(y = data$Home_Win, p_hat = predict(model, data, type="prob")[,2])
  }
  tibble(loss, iter = 1:M, var)
}

perm_feat_imp_2(model = class_forest, data = testing_set, var = "Home_ELO", M = 2)

vars = feature_names  # vector of all variables
M = 10                             # number of replications
importance_2 = map_df(vars, \(j) perm_feat_imp_2(model = class_forest, data = testing_set, var = j, M = M, seed = 2023))

importance_2 %>% 
  mutate(importance = loss - loss_base) %>% 
  ggplot(aes(reorder(var, -importance, decreasing=T), importance)) + 
  geom_boxplot(alpha=.6, width=.75) +
  coord_flip() + 
  theme_bw() + 
  labs(title = "Permutation After Fitting",x = "Feature", y="Importance")



# stack(loss_diffs) %>%
#   ggplot(aes(x=ind, y=values)) + geom_boxplot()
# 
# stack(loss_diffs) %>%
#   mutate(values = values*(-1)) %>%
#   mutate(feature = fct_reorder(ind, values, .fun='median')) %>%
#   filter(feature != "Home_ELO" & feature != "Away_ELO") %>%
#   ggplot(aes(x=feature, y=values)) +
#     geom_boxplot(alpha=.6, width=.75) +
#     coord_flip() + 
#     theme_bw() +
#     labs(x = "Feature", y="Importance") + 
#     ylim(0, 0.03)
# 
# stack(loss_diffs) %>%
#   mutate(values = values*(-1)) %>%
#   mutate(feature = fct_reorder(ind, values, .fun='median')) %>%
#   filter(feature == "Home_ELO" | feature == "Away_ELO") %>%
#   ggplot(aes(x=feature, y=values)) +
#   geom_boxplot(alpha=.6, width=.75) +
#   coord_flip() + 
#   theme_bw() +
#   labs(title = "Permutation (After Fitting) Feature Importance", x = "Feature", y="Importance") + 
#   ylim(0.15, 0.3)



# M<- 10
# loss_diffs2 <- list()
# for (var in names(data5)[-1]){
#   losses <- numeric(M)
#   for (i in 1:M){
#     data <- data5
#     unchanged_x <- data %>% select(-Home_Win)
#     orig_y <- data$Home_Win
#     data[[var]] <- sample(data[[var]])
#     data$Home_Win <- factor(data5$Home_Win, levels = c(0,1))
#     model <- randomForest(Home_Win~.,data = data)
#     y_pred_perm <- predict(model,unchanged_x,type="prob")
#     loss_perm <- logLoss(y_pred_perm,orig_y)
#     losses[i] <- loss - loss_perm
#   }
#   loss_diffs2[[var]] <- losses
# }
# 
# boxplot(loss_diffs2, main = "Permutation (Before Fitting) Feature Importance", xlab = "Variable", ylab = "Loss Difference")
# 
# 
# 
# stack(loss_diffs2) %>%
#   mutate(feature = fct_reorder(ind, values, .fun='median')) %>%
#   # filter(feature != "Home_ELO" & feature != "Away_ELO") %>%
#   ggplot(aes(x=feature, y=values)) +
#   geom_boxplot(alpha=.6, width=.75) +
#   coord_flip() + 
#   theme_bw() +
#   labs(title = "Permutation (Before Fitting) Feature Importance", x = "Feature", y="Importance")
# 
# stack(loss_diffs2) %>%
#   mutate(feature = fct_reorder(ind, values, .fun='median')) %>%
#   filter(feature == "Home_ELO" | feature == "Away_ELO") %>%
#   ggplot(aes(x=feature, y=values)) +
#   geom_boxplot(alpha=.6, width=.75) +
#   coord_flip() + 
#   theme_bw() +
#   labs(title = "Permutation (Before Fitting) Feature Importance", x = "Feature", y="Importance") + 
#   ylim(0.15, 0.3)


fit <- function(data) {
  ranger(
    Home_Win~.,
    data = data,
    mtry = 25,
    num.trees = 150,
    min.node.size = 1,
    splitrule = "gini",
    probability = TRUE,
    importance = "impurity"
  )
}

perm_feat_imp_3 <- function(fit, data_train, data_test, var, M=1, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  loss = numeric(M)
  for(i in 1:M){
    data_train[[var]] = sample(data_train[[var]])  
    RF = fit(data_train)
    loss[i] = loss[i] = log_loss(y = data_test$Home_Win, p_hat = predict(RF, data_test)$predictions[,"1"])
  }
  
  tibble(loss, iter = 1:M, var)
}

perm_feat_imp_3(fit, data_train = training_set, data_test = testing_set, var = "Home_ELO", M = 2)

vars = feature_names  # vector of all variables
M = 10                             # number of replications
importance_3 = map_df(vars, \(j) perm_feat_imp_3(fit, data_train = training_set, data_test = testing_set, var = j, M = M, seed = 2023))

importance_3 %>% 
  mutate(importance = loss - loss_base) %>% 
  ggplot(aes(reorder(var, -importance, decreasing=T), importance)) + 
  geom_boxplot(alpha=.6, width=.75) +
  coord_flip() + 
  theme_bw() + 
  labs(title = "Permutation Before Fitting", x = "Feature", y= "Importance")


class_ranger <- ranger(
  Home_Win~.,
  data = training_set,
  mtry = 25,
  num.trees = 150,
  min.node.size = 1,
  splitrule = "gini",
  probability = FALSE,
  importance = "impurity"
)

class_ranger <- ranger(
  Home_Win~.,
  data = training_set,
  mtry = 25,
  num.trees = 150,
  min.node.size = 1,
  splitrule = "gini",
  probability = TRUE,
  importance = "impurity"
)

probs_ranger <- predict(class_ranger, testing_set)$predictions[,"1"]

log_loss(y_test, rep(0.627, 218))

testing_set %>% View()

length(y_test)


preds_ranger <- predict(class_ranger, testing_set)$predictions
tibble(preds_ranger, y_test) %>% table()


new_data <- data5%>% select(-Home_Inactive,-Home_Travel,-Home_Covers,-Away_Inactive,-Away_Cover)
new_data$Home_Win <- factor(new_data$Home_Win, levels = c(0,1))
new_model <- randomForest(Home_Win~., data = new_data)
new_model$importance


data3 <- classificationModel
data3[, -which(names(data3) == "Home_Win")] <- scale(data3[, -which(names(data3) == "Home_Win")])

am.data = glm(Home_Win ~., data = data3, family = binomial)
print(summary(am.data))

am2 = glm(formula = Home_Win ~ Home_ELO + Away_ELO + Away_Inactive, data = classificationModel,family = binomial)
print(summary(am2))

X_t <- data3 %>% select(-Home_Win)
probs <- am.data %>% predict(X_t, type = "response")
pred.classes <- ifelse(probs >.5, 1,0)

actual <- data3 %>% select(Home_Win)

df3 <- cbind(pred.classes,actual)
confusion_matrix_logreg <- table(df3)

ELO_diff <- X_t$Home_ELO - X_t$Away_ELO
ELO_diff2 <- tibble(ELO_diff)

elodata = cbind(actual,ELO_diff2)
  
elodiffmodel <- glm(formula = actual ~ ELO_diff2, data=elodata,family = binomial)
probabilities <- elodiffmodel %>% predict(X_t, type = "response")
preds2 <- ifslese(probabilities > .5,1,0)
conf_table <- cbine(preds2,actual)

confusion_matrix_2 <- table(conf_table)
