rm(list=ls())

#applying knn to heroin data
#Load necessary libraries 
library(caret) 
library(class)

# Load the dataset  
data_subset <- read.csv("C:/Users/YENISI/Downloads/drug+consumption+quantified/drug_consumption.data", sep=",", header = F)[, -c(1,4,5, 14:23, 25:32)] 

colnames(data_subset) <- c("age","gender","education","neuroticism",        
"extraversion", "openness_to_experience", "agreeableness", 
"conscientiousness", "impulsivity","sensation_seeking", "heroin")

data_subset$heroin = as.factor(data_subset$heroin)
levels(data_subset$heroin)=c("CL0","CL1","CL2","CL3","CL4","CL4","CL4")

# Partition data into training (70%) and testing (30%) sets 
set.seed(123) 
# for reproducibility 
train_index <- createDataPartition(data_subset$heroin, p = 0.7, list = FALSE) 
train_data <- data_subset[train_index, ] 
test_data <- data_subset[-train_index, ]

# Fit KNN model on training data 
# Example range from 1 to 20 with step 2
k_values <- seq(1, 20, by = 2)
# 10-fold cross-validation
kfold <- trainControl(method = "cv", number = 9)  

knn_model <- train(heroin ~ ., data = train_data, method = "knn", 
trControl = kfold, tuneGrid = data.frame(k = k_values)) 
knn_model 

# Predict on test data 
predictions <- predict(knn_model, newdata = test_data) 

# Evaluate model performance 
confusion_matrix <- table(predictions, test_data$heroin) 
confusion_matrix 
accuracy <- sum(diag(confusion_matrix)) /sum(confusion_matrix) 
print(paste("Accuracy:", accuracy)) 


#applying weighted knn to heroin data
library(kknn)
library(doParallel)
library(caret) 
library(class)

# Load the dataset  
data_subset <- read.csv("C:/Users/YENISI/Downloads/drug+consumption+quantified/drug_consumption.data", sep=",", header = F)[, -c(1,4,5, 14:23, 25:32)] 

colnames(data_subset) <- c("age","gender","education","neuroticism",        
"extraversion", "openness_to_experience", "agreeableness", 
"conscientiousness", "impulsivity","sensation_seeking", "heroin")

data_subset$heroin = as.factor(data_subset$heroin)
levels(data_subset$heroin)=c("CL0","CL1","CL2","CL3","CL4","CL4","CL4")

# Partition data into training (70%) and testing (30%) sets 
set.seed(123) 
# for reproducibility 
train_index <- createDataPartition(data_subset$heroin, p = 0.7, list = FALSE) 
train_data <- data_subset[train_index, ] 
test_data <- data_subset[-train_index, ]

# Define a smaller tuning grid for kknn
grid <- expand.grid(kmax = 1:10, distance = 1:3, kernel = "optimal")

# Set up parallel processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Define cross-validation method
kfold <- trainControl(method = "cv", number = 10)

# Fit weighted kNN model on training data using parallel processing
weighted_knn_model <- train(heroin ~ ., data = train_data, method = "kknn", trControl = kfold, tuneGrid = grid)

# Stop parallel processing
stopCluster(cl)

# Print the model
print(weighted_knn_model)

# Predict on test data
prediction <- predict(weighted_knn_model, newdata = test_data)

# Evaluate model performance
confusion_matrix <- table(prediction, test_data$heroin)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))
print(confusion_matrix)

#Plot heatmap for weighted kNN model
confusion_df <- as.data.frame(as.table(confusion_matrix))
names(confusion_df) <- c("Predicted_Class", "True_Class", "Frequency")

#Plot heatmap for weighted kNN model using ggplot with dark green color palette
ggplot(confusion_df, aes(x = Predicted_Class, y = True_Class, fill = Frequency)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c(option = "D", direction = -1) +  # Use dark green color palette
  labs(title = "Confusion Matrix Heatmap",
       x = "Predicted Class",
       y = "True Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  
# Rotate x-axis labels
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 16, hjust = 0.5),
        legend.title = element_blank(),   # Remove legend title
        legend.position = "right",        # Adjust legend position
        legend.text = element_text(size = 10))

library(reshape2)

# Reshape the train data
train_data_melted <- melt(train_data, id.vars = c("heroin"))

# Plot heatmap for train data using ggplot with dark green color palette
ggplot(train_data_melted, aes(x = variable, y = heroin, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c(option = "D", direction = -1) +  # Use dark green color palette
  labs(title = "Heatmap for Train Data",
       x = "Variable",
       y = "heroin") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  # Rotate x-axis labels
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 16, hjust = 0.5))

#-------------------------------------------------------------------------
#weighted k-nn to cannabis
library(kknn)
library(doParallel)
library(caret)

# Load the dataset  
data_subset <- read.csv("C:/Users/YENISI/Downloads/drug+consumption+quantified/drug_consumption.data", sep=",", header = F)[, -c(1,4,5, 14:18, 20:32)] 

colnames(data_subset) <- c("age","gender","education","neuroticism",        
"extraversion", "openness_to_experience", "agreeableness", 
"conscientiousness", "impulsivity","sensation_seeking", "cannabis")

data_subset$cannabis <- as.factor(data_subset$cannabis)
levels(data_subset$cannabis) <- c("CL0", "CL1", "CL2", "CL3", "CL4", "CL4", "CL4")

# Partition data into training (70%) and testing (30%) sets
set.seed(123) 
train_index <- createDataPartition(data_subset$cannabis, p = 0.7, list = FALSE)
train_data <- data_subset[train_index, ]
test_data <- data_subset[-train_index, ]

# Define a smaller tuning grid for kknn
grid <- expand.grid(kmax = 1:10, distance = 1:3, kernel = "optimal")

# Set up parallel processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Define cross-validation method
kfold <- trainControl(method = "cv", number = 10)

# Fit weighted kNN model on training data using parallel processing
weighted_knn_model <- train(cannabis ~ ., data = train_data, method = "kknn", trControl = kfold, tuneGrid = grid)

# Stop parallel processing
stopCluster(cl)

# Print the model
print(weighted_knn_model)

# Predict on test data
predictions <- predict(weighted_knn_model, newdata = test_data)

# Evaluate model performance
confusion_matrix <- table(predictions, test_data$cannabis)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))
print(confusion_matrix)

#Plot heatmap for weighted kNN model
confusion_df <- as.data.frame(as.table(confusion_matrix))
names(confusion_df) <- c("Predicted_Class", "True_Class", "Frequency")

#Plot heatmap for weighted kNN model using ggplot with dark green color palette
ggplot(confusion_df, aes(x = Predicted_Class, y = True_Class, fill = Frequency)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c(option = "D", direction = -1) +  # Use dark green color palette
  labs(title = "Confusion Matrix Heatmap",
       x = "Predicted Class",
       y = "True Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  
# Rotate x-axis labels
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 16, hjust = 0.5),
        legend.title = element_blank(),   # Remove legend title
        legend.position = "right",        # Adjust legend position
        legend.text = element_text(size = 10))

library(reshape2)

# Reshape the train data
train_data_melted <- melt(train_data, id.vars = c("cannabis"))

# Plot heatmap for train data using ggplot with dark green color palette
ggplot(train_data_melted, aes(x = variable, y = cannabis, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c(option = "D", direction = -1) +  # Use dark green color palette
  labs(title = "Heatmap for Train Data",
       x = "Variable",
       y = "Cannabis") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  # Rotate x-axis labels
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 16, hjust = 0.5))
#------------
#shapiro wilks test
# Load required libraries
library(dplyr)
library(caret) 
library(mvnormtest)

# Load your dataset
data <- read.csv("C:/Users/YENISI/Downloads/drug+consumption+quantified/drug_consumption.data", sep=",", header = F)[, -c(1,4,5, 14:18, 20:32)] 

colnames(data) <- c("age","gender","education","neuroticism",        
"extraversion", "openness_to_experience", "agreeableness", 
"conscientiousness", "impulsivity","sensation_seeking", "cannabis")

# Convert 'cannabis' column to factor with meaningful levels
data$cannabis <- as.factor(data$cannabis)
levels(data$cannabis) <- c("CL0", "CL1", "CL2", "CL3", "CL4", "CL4", "CL4")

# Split the data into training (70%) and testing (30%) sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(data$cannabis, p = 0.7, list = FALSE)
train_data <- data[train_index, ]

# Split training data into separate data frames based on the 'cannabis' classes
train_data_split <- split(train_data, train_data$cannabis)

# Extract numeric attributes
numeric_attributes <- train_data[, 1:10]

# Transpose the numeric attributes data frame
transposed_numeric <- t(numeric_attributes)

# Perform multivariate Shapiro-Wilk test
multivariate_shapiro_test <- lapply(train_data_split, function(df) {
  mshapiro.test(t(df[, 1:10]))
})

# Display the test results
multivariate_shapiro_test
#-------------------------------
#box m
# Load required libraries
library(dplyr)
library(caret) 
library(mvnormtest)

# Load your dataset
data <- read.table("C:/Users/YENISI/Downloads/drug+consumption+quantified/drug_consumption.data", sep=",", header = F)[, -c(1,4,5, 14:18, 20:32)] 

colnames(data) <- c("age","gender","education","neuroticism",        
"extraversion", "openness_to_experience", "agreeableness", 
"conscientiousness", "impulsivity","sensation_seeking", "cannabis")

# Convert 'cannabis' column to factor with meaningful levels
data$cannabis <- as.factor(data$cannabis)
levels(data$cannabis) <- c("CL0", "CL1", "CL2", "CL3", "CL4", "CL4", "CL4")

# Split the data into training (70%) and testing (30%) sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(data$cannabis, p = 0.7, list = FALSE)
train_data <- data[train_index, ]

# Split training data into separate data frames based on the 'cannabis' classes
train_data_split <- split(train_data, train_data$cannabis)

# Extract numeric attributes
numeric_attributes <- train_data[, 1:10]

# Transpose the numeric attributes data frame
transposed_numeric <- t(numeric_attributes)


require("heplots") 
library(mvnormtest)
attach(train_data) 

#box_m
box_m_test_result <- boxM(cbind(age,gender,education,neuroticism,
extraversion,openness_to_experience,                                
agreeableness, conscientiousness, impulsivity,                               
sensation_seeking) ~ cannabis, data = train_data)
#------------------
#qda
library(kknn)
library(doParallel)
library(caret)
library(MASS) 

# Load the dataset  
data_subset <- read.table("C:/Users/YENISI/Downloads/drug+consumption+quantified/drug_consumption.data", sep=",", header = F)[, -c(1,4,5, 14:18, 20:32)] 

colnames(data_subset) <- c("age","gender","education","neuroticism",        
"extraversion", "openness_to_experience", "agreeableness", 
"conscientiousness", "impulsivity","sensation_seeking", "cannabis")

data_subset$cannabis <- as.factor(data_subset$cannabis)
levels(data_subset$cannabis) <- c("CL0", "CL1", "CL2", "CL3", "CL4", "CL4", "CL4")

# Partition data into training (70%) and testing (30%) sets
set.seed(123) 
train_index <- createDataPartition(data_subset$cannabis, p = 0.7, list = FALSE)
train_data <- data_subset[train_index, ]
test_data <- data_subset[-train_index, ]

# Fit QDA model on the updated training data 
model <- qda(cannabis ~ ., data = train_data) 
model
# Evaluate the model on test data 
predicted <- predict(model, test_data) 
accuracy <- mean(predicted$class == test_data$cannabis) 
accuracy