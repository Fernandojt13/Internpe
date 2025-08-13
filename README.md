Machine Learning Projects
This repository contains four machine learning projects, each focused on a different problem domain. These projects utilize various Python libraries such as pandas, scikit-learn, matplotlib, and tensorflow to perform data processing, model training, and evaluation.

1. Breast Cancer Detection 🩺
This project aims to classify tumors as malignant (cancerous) or benign (non-cancerous) based on a dataset of medical features. It uses a Logistic Regression model, which is a powerful and interpretable algorithm for binary classification tasks.

Key Features
Data Preprocessing: Handles missing values and scales numerical features using StandardScaler to improve model performance.

Model Training: Employs a LogisticRegression model from scikit-learn to predict the diagnosis.

Evaluation: The model's performance is evaluated using accuracy_score, confusion_matrix, and classification_report.

Visualization: A scatter plot is generated to visually compare the actual and predicted labels for the test data.

Dependencies
numpy

pandas

scikit-learn

matplotlib

2. Car Price Prediction 🚗
This project predicts the selling price of a used car based on several features like company, year, and kilometers driven. It uses a Linear Regression model, which is a suitable algorithm for predicting a continuous numerical value.

Key Features
Data Cleaning: The script cleans and preprocesses raw data, converting columns like 'year' and 'Price' into appropriate numerical formats.

Feature Engineering: Uses OneHotEncoder via a ColumnTransformer to handle categorical features like company and fuel_type, making them suitable for the regression model.

Exploratory Data Analysis (EDA): Uses seaborn and matplotlib to visualize relationships between features and the target price, helping to understand the data better.

Model Training: A LinearRegression model from scikit-learn is trained on the preprocessed data.

Evaluation: The model's performance is assessed using the r2_score to measure how well the predictions match the actual prices.

Dependencies
pandas

matplotlib

seaborn

scikit-learn

3. Diabetes Prediction 🩸
This project predicts whether a person has diabetes or not based on diagnostic measurements included in a dataset. It uses a Support Vector Machine (SVM) classifier, a robust model for complex classification problems.

Key Features
Data Preprocessing: Standardizes the features using StandardScaler to ensure all variables contribute equally to the model.

Model Training: An SVC (Support Vector Classifier) with a linear kernel is used for training.

Evaluation: The model's accuracy is calculated on both training and test datasets to check for overfitting and generalization.

Interactive Prediction: A function is included to take user input for a new set of health metrics and predict the diabetes status in real-time.

Dependencies
numpy

pandas

scikit-learn

4. IPL Match Win Probability Prediction 🏏
This project predicts the win probability of the team batting in the second innings of an Indian Premier League (IPL) match. It uses a Logistic Regression model within a Pipeline to handle both data transformation and modeling.

Key Features
Data Preparation: Merges and cleans data from two CSV files (matches.csv and deliveries.csv) to create a comprehensive dataset for analysis.

Feature Engineering: Creates relevant features like runs_left, balls_left, crr (current run rate), and rrr (required run rate) to capture the state of the match.

Pipeline: Uses a Pipeline with a ColumnTransformer to streamline the preprocessing of categorical features and model training.

Model Training: A LogisticRegression model is used to predict the probability of winning.

Evaluation & Visualization: The model's performance is evaluated using accuracy_score and confusion_matrix. A detailed visualization is created to show how the win and lose probabilities change over the course of a match.

Dependencies
numpy

pandas

matplotlib

seaborn

scikit-learn

pickle
