#Load the necessary libraries for data manipulation and model training
library(tidyverse)         # For data manipulation and visualization
library(caret)             # For machine learning modeling
library(e1071)             # For SVM and other utilities
library(class)             # For k-NN
library(nnet)              # For neural networks
library(plyr)              # For data manipulation
library(caretEnsemble)     # For creating model ensembles
library(DMwR)              # For SMOTE (Synthetic Minority Over-sampling Technique)
library(randomForest)      # For random forest
library(C50)               # For decision tree models
library(ROSE)              # For dealing with imbalanced data
library(doParallel)        # For parallel processing
library(randomGLM)         # For random GLM

#Check current working directory
getwd()

#Read the raw data from CSV file
data_raw <- read.csv("Raw_Data/train.csv", sep="|")

#Display the first few rows of the data
head(data_raw)

#Show the structure of the dataset
str(data_raw)

#Count the number of missing values in each column
colSums(is.na(data_raw))

#Convert categorical variables to dummy variables
dmy <- dummyVars("~.", data=data_raw, fullRank = TRUE)

#Create a new dataframe with dummy variables
data <- data.frame(predict(dmy, newdata=data_raw))

#Check the structure of the new dataframe
str(data)

#Convert the 'fraud' column to a factor
data$fraud <- as.factor(data$fraud)

#Rename factor levels for clarity
levels(data$fraud) <- c("No_fraud", "fraud")

#Check the structure of the updated dataframe
str(data)

#Partition the data into training and testing sets
train_index <- createDataPartition(y=data$fraud, p=0.50, list=FALSE)

#Display the indices of training data
train_index

#Check the structure of the training indices
str(train_index)

#Create the training dataset
train_data_raw <- data[train_index,]

#Create the testing dataset
test_data <- data[-train_index,]

#Check the structure of the training data
str(train_data_raw)

#Display the distribution of the target variable in the training data
table(train_data_raw$fraud)

#Display the distribution of the target variable in the testing data
table(test_data$fraud)

#Resampling: Apply SMOTE to handle class imbalance
train_data <- SMOTE(fraud~., train_data_raw, perc.over=1500, perc.under = 115)

#Check the distribution of the target variable after SMOTE
table(train_data$fraud)

#Check the structure of the resampled training data
str(train_data)

#Feature selection using recursive feature elimination (RFE)
#Using repeated cross-validation (10-folds)
control_rfe <- rfeControl(functions=rfFuncs, method="repeatedcv", repeats=5, verbose=TRUE)
outcomeName <- 'fraud'
predictors <- names(train_data)[!names(train_data) %in% outcomeName]

#Run RFE to select important features
train_data_rfe <- rfe(train_data[,predictors], train_data[,outcomeName], rfeControl=control_rfe)

#Summarize the results of RFE
train_data_rfe

#Retrieve the predictors selected by RFE
predictors <- predictors(train_data_rfe)

#Display the selected predictors
predictors

#Model stacking
#Define the algorithms to be included in the stacking
stacked_algorithm <- c("glm", "knn", "lda","nb", "rpart")

#Set up cross-validation
control <- trainControl(method="repeatedcv", number=10, repeats=10, index = createFolds(train_data$fraud,10), savePredictions="final", classProbs=TRUE, summaryFunction=twoClassSummary)

#Set seed for reproducibility
set.seed(100)

#Train multiple models using caretList
model_list <- caretList(train_data[,predictors], train_data[,outcomeName], trControl = control, metric="Sens", methodList = stacked_algorithm)

#Summarize the resampling results
result <- resamples(model_list)

#Display the summary of resampling results
summary(result)

#Box plots to compare model performance
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(result, scales=scales)
dotplot(result)

#Check correlation between models to ensure they are uncorrelated and can be ensembled
modelCor(result)
splom(result)

#Stack models using logistic regression
set.seed(101)
glm_stack <- caretStack(model_list, method="glm", metric="Sens", trControl=control)

#Print the summary of the stacked model
print(glm_stack)
summary(glm_stack)
plot(glm_stack)

#Make predictions using the stacked model
glm_stack_classes <- predict(glm_stack, newdata=test_data[,predictors], type="raw")

#Evaluate the performance of the stacked model
glm_cm <- confusionMatrix(glm_stack_classes, test_data$fraud)
glm_cm

#Custom function to calculate monetary payoff (user-defined function assumed to be included in the script)
monetary_payoff(glm_cm)

#Stack models using Gradient Boosting Machine (GBM)
set.seed(101)
gbm_stack <- caretStack(model_list, method="gbm", metric="Sens", trControl = control)

#Print the summary of the stacked model
print(gbm_stack)
summary(gbm_stack)
plot(gbm_stack)

#Make predictions using the GBM stacked model
gbm_stack_classes <- predict(gbm_stack, newdata=test_data[,predictors], type="raw")

#Evaluate the performance of the GBM stacked model
gbm_cm <- confusionMatrix(gbm_stack_classes, test_data$fraud)
gbm_cm

#Custom function to calculate monetary payoff (user-defined function assumed to be included in the script)
monetary_payoff(gbm_cm)

#Generate predictions from each model in the list and add predictions from the ensemble models
test_preds <- data.frame(sapply(model_list, predict, newdata = test_data[,predictors]))
test_preds$ensemble_glm <- predict(glm_stack, newdata = test_data[,predictors])
test_preds$ensemble_gbm <- predict(gbm_stack, newdata = test_data[,predictors])
test_preds