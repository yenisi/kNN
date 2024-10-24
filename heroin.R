rm(list=ls())
# Load necessary libraries
library(ggplot2)
library(gridExtra)

#---HEROIN
#----"neuroticism", "extraversion"

data_subset <- read.csv("C:/Users/srijani/Desktop/knn project/drug_data.csv", header = FALSE)[, -c(1,4,5, 14:23, 25:32)]

colnames(data_subset) <- c("age", "gender", "education",
                           "neuroticism", "extraversion", "openness_to_experience",
                           "impulsivity", "conscientiousness", "agreeableness",
                           "sensation_seeking", "heroin")
data_subset$heroin <- as.factor(data_subset$heroin)
levels(data_subset$heroin) <- c("CL0", "CL1", "CL2", "CL3", "CL4", "CL4", "CL4")

# Partition data into training (70%) and testing (30%) sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(data_subset$heroin, p = 0.7, list = FALSE)
train_data <- data_subset[train_index, ]
test_data <- data_subset[-train_index, ]

# Create scatterplot with color coded points and legend
scatterplot <- ggplot(train_data, aes(x = extraversion, y = neuroticism, color = heroin)) +
  geom_point() +
  labs(title = "Scatterplot of Extraversion vs. Neuroticism",
       x = "Extraversion", y = "Neuroticism", color = "Heroin") +
  scale_color_manual(values = c("CL0" = "blue", "CL1" = "green", "CL2" = "red", "CL3" = "orange", "CL4" = "purple")) +
  theme_minimal()

# Train k-NN classifier
k <- 10  # Number of neighbors
knn_model <- knn(train_data[, c("extraversion", "neuroticism")], test_data[, c("extraversion", "neuroticism")], train_data$heroin, k)

# Generate a grid of points
x_range <- range(train_data$extraversion)
y_range <- range(train_data$neuroticism)
x_grid <- seq(x_range[1], x_range[2], length.out = 100)
y_grid <- seq(y_range[1], y_range[2], length.out = 100)
grid <- expand.grid(extraversion = x_grid, neuroticism = y_grid)

# Predict class labels for each point in the grid
grid$predicted <- knn(train_data[, c("extraversion", "neuroticism")], grid, train_data$heroin, k)

# Plot decision boundaries
decision_boundary <- ggplot(grid, aes(x = extraversion, y = neuroticism, color = predicted)) +
  geom_point(size = 1) +
  scale_color_manual(values = c("CL0" = "lightblue", "CL1" = "green", "CL2" = "red", "CL3" = "orange", "CL4" = "purple")) +
  geom_point(data = train_data, aes(x = extraversion, y = neuroticism, color = heroin), size = 3, shape = 19, alpha = 0.7) +
  labs(title = "Decision Boundaries for k-NN Classification",
       x = "Extraversion", y = "Neuroticism", color = "Predicted Heroin Class") +
  theme_minimal()

# Arrange plots side by side
grid.arrange(scatterplot, decision_boundary, ncol = 2)

#-----"agreeableness","openness_to_experience"

# Load necessary libraries
library(ggplot2)
library(class)
library(caret)

# Read the dataset
data_subset <- read.csv("C:/Users/srijani/Desktop/knn project/drug_data.csv", header = FALSE)[, -c(1,4,5, 14:23, 25:32)]

colnames(data_subset) <- c("age", "gender", "education",
                           "neuroticism", "extraversion", "openness_to_experience",
                           "agreeableness", "conscientiousness", "impulsivity",
                           "sensation_seeking", "heroin")
data_subset$heroin <- as.factor(data_subset$heroin)
levels(data_subset$heroin) <- c("CL0", "CL1", "CL2", "CL3", "CL4", "CL4", "CL4")


# Partition data into training (70%) and testing (30%) sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(data_subset$heroin, p = 0.7, list = FALSE)
train_data <- data_subset[train_index, ]
test_data <- data_subset[-train_index, ]

# Create scatterplot with color coded points and legend
scatterplot <- ggplot(train_data, aes(x = openness_to_experience, y = agreeableness, color = heroin)) +
  geom_point() +
  labs(title = "Scatterplot of openness_to_experience vs. agreeableness",
       x = "openness_to_experience", y = "agreeableness", color = "Heroin") +
  scale_color_manual(values = c("CL0" = "blue", "CL1" = "green", "CL2" = "red", "CL3" = "orange", "CL4" = "purple")) +
  theme_minimal()

# Train k-NN classifier
k <- 10  # Number of neighbors
knn_model <- knn(train_data[, c("openness_to_experience", "agreeableness")], test_data[, c("openness_to_experience", "agreeableness")], train_data$heroin, k)

# Generate a grid of points
x_range <- range(train_data$openness_to_experience)
y_range <- range(train_data$agreeableness)
x_grid <- seq(x_range[1], x_range[2], length.out = 100)
y_grid <- seq(y_range[1], y_range[2], length.out = 100)
grid <- expand.grid(openness_to_experience = x_grid, agreeableness = y_grid)

# Predict class labels for each point in the grid
grid$predicted <- knn(train_data[, c("openness_to_experience", "agreeableness")], grid, train_data$heroin, k)

# Plot decision boundaries
decision_boundary <- ggplot(grid, aes(x = openness_to_experience, y = agreeableness, color = predicted)) +
  geom_point(size = 1) +
  scale_color_manual(values = c("CL0" = "lightblue", "CL1" = "green", "CL2" = "red", "CL3" = "orange", "CL4" = "purple")) +
  geom_point(data = train_data, aes(x = openness_to_experience, y = agreeableness, color = heroin), size = 3, shape = 19, alpha = 0.7) +
  labs(title = "Decision Boundaries for k-NN Classification",
       x = "openness_to_experience", y = "agreeableness", color = "Predicted Heroin Class") +
  theme_minimal()

# Arrange plots side by side
grid.arrange(scatterplot, decision_boundary, ncol = 2)


