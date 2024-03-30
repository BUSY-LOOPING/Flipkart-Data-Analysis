# Analysis Overview

This repository contains an overview of the data analysis process for predicting price. The analysis is structured into several steps, each employing various data analysis techniques and machine learning algorithms.

## Importing Dependencies

The analysis begins with importing necessary dependencies, ensuring a smooth workflow.

## Reading Data

The dataset is read into the environment to kickstart the analysis.

## Data Preprocessing

1. **Removing Redundant Columns**: Unnecessary columns are removed to streamline the dataset.

2. **Checking for Null Values**: The dataset is inspected for missing values.

3. **Filling Null Values**: Techniques are applied to handle missing data effectively.

4. **Feature Transformation**: Certain features are transformed to improve model performance.

5. **Feature Generation**: New features are created to enhance the predictive power of the model.

6. **Binary Encoding**: Categorical variables are encoded to prepare them for machine learning algorithms.

7. **Correlation Analysis**: The correlation between variables is analyzed to understand their relationships.

8. **Data Visualizations**: Various visualizations are employed to gain insights into the data distribution and patterns.

## Predicting Price

The main objective of the analysis is to predict price using machine learning algorithms.

### Data Preparation

- **Separating Independent and Dependent Features**: The dataset is split into independent features (predictors) and the dependent feature (target variable).

- **Train-Test Split**: The dataset is divided into training and testing sets to evaluate model performance.

### Model Building

1. **Linear Regression**: A linear regression model is trained on the data to predict prices.

2. **Random Forest Regression**: A random forest regression model is employed to capture non-linear relationships in the data.

3. **Gradient Boosting Regression**: Gradient boosting regression is utilized to create an ensemble of weak learners for improved prediction accuracy.

### Evaluation

- **Training**: The models are trained on the training dataset.

- **Evaluating Predictions**: Model performance is evaluated using appropriate evaluation metrics.

- **Visualizing Predictions**: Visualizations are created to compare predicted prices with actual prices and assess model accuracy.
