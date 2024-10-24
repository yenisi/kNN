rm(list=ls())
library(ggplot2)
library(gridExtra)

#---cannabis
#----"neuroticism", "conscientiousness"

data_subset <- read.csv("C:/Users/srijani/Desktop/knn project/drug_data.csv", header = FALSE)[, -c(1,4,5, 14:18, 20:32)]

colnames(data_subset) <- c("age", "gender", "education",
                           "neuroticism", "extraversion", "openness_to_experience",
                           "agreeableness", "conscientiousness", "impulsivity",
                           "sensation_seeking", "cannabis")
data_subset$cannabis <- as.factor(data_subset$cannabis)
levels(data_subset$cannabis) <- c("CL0", "CL1", "CL2", "CL3", "CL4", "CL4", "CL4")

# Partition data into training (70%) and testing (30%) sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(data_subset$cannabis, p = 0.7, list = FALSE)
train_data <- data_subset[train_index, ]
test_data <- data_subset[-train_index, ]

# Create scatterplot with color coded points and legend
scatterplot <- ggplot(train_data, aes(x = conscientiousness, y = neuroticism, color = cannabis)) +
  geom_point() +
  labs(title = "Scatterplot of conscientiousness vs. Neuroticism",
       x = "conscientiousness", y = "Neuroticism", color = "cannabis") +
  scale_color_manual(values = c("CL0" = "blue", "CL1" = "green", "CL2" = "red", "CL3" = "orange", "CL4" = "purple")) +
  theme_minimal()

# Train k-NN classifier
k <- 10  # Number of neighbors
knn_model <- knn(train_data[, c("conscientiousness", "neuroticism")], test_data[, c("conscientiousness", "neuroticism")], train_data$cannabis, k)

# Generate a grid of points
x_range <- range(train_data$conscientiousness)
y_range <- range(train_data$neuroticism)
x_grid <- seq(x_range[1], x_range[2], length.out = 100)
y_grid <- seq(y_range[1], y_range[2], length.out = 100)
grid <- expand.grid(conscientiousness = x_grid, neuroticism = y_grid)

# Predict class labels for each point in the grid
grid$predicted <- knn(train_data[, c("conscientiousness", "neuroticism")], grid, train_data$cannabis, k)

# Plot decision boundaries
# Create decision boundary plot
decision_boundary <- ggplot(grid, aes(x = conscientiousness, y = neuroticism, color = predicted)) +
  geom_point(size = 1) +
  scale_color_manual(values = c("CL0" = "lightblue", "CL1" = "green", "CL2" = "red", "CL3" = "orange", "CL4" = "purple")) +
  labs(title = "Decision Boundaries for k-NN Classification",
       x = "conscientiousness", y = "Neuroticism", color = "Predicted cannabis Class") +
  theme_minimal()

# Arrange plots side by side
grid.arrange(scatterplot, decision_boundary, ncol = 2)