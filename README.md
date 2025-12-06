# customer_churn_prediction
Customer Churn Prediction using Machine Learning:

This project builds a machine learning model to predict whether a customer is likely to churn (leave a service).
The objective is to help businesses identify at-risk customers and take proactive actions to retain them.

This project was developed as part of an internship assignment and demonstrates an end-to-end ML workflow including preprocessing, handling imbalanced data, model training, comparison, and evaluation.

 Project Highlights:

-Cleaned and preprocessed the Telco Customer Churn dataset

-Converted categorical values using One-Hot Encoding

-Handled missing values (especially in TotalCharges)

-Balanced the dataset using SMOTE

Trained and evaluated:

✔ Logistic Regression

✔ Decision Tree Classifier

✔ Neural Network (MLPClassifier)

Achieved good recall and F1-scores for predicting churn

Project executed using VS Code + Python + scikit-learn + imbalanced-learn

Folder Structure:

customer_churn_prediction/
│── churn_model.py
│── README.md
│── requirements.txt
│── data/
│     └── customer_churn.csv
│── .venv/     (virtual environment - not uploaded)

 Understanding the Problem:

Churn occurs when a customer stops using a service.
Telecom companies, banks, subscription platforms, and e-commerce companies heavily rely on churn prediction to prevent customer loss.

Technologies Used:

-Python

-Pandas / NumPy

-Scikit-Learn

-SMOTE (Imbalanced-Learn)

-Matplotlib / Seaborn

-VS Code

 -Data Preprocessing Steps

-Removed unique identifier customerID

-Converted Churn into numerical labels (Yes → 1, No → 0)

-Fixed datatype issues in TotalCharges

-Filled missing values using median

-Applied One-Hot Encoding to all categorical columns

-Performed train-test split with stratification

-Scaled numerical features using StandardScaler

-Applied SMOTE to balance minority class

Models Trained:

1️. Logistic Regression

Simple and fast baseline model

Performs well with balanced data after SMOTE

Good recall for churn prediction

2️. Decision Tree Classifier

Captures nonlinear patterns

Easy to interpret

Depth limited to prevent overfitting

3️. Neural Network (MLPClassifier)

Two hidden layers (64, 32)

ReLU activation, Adam optimizer

Achieves competitive performance

 Model Evaluation Metrics:

Each model was evaluated using,

-Accuracy

-Precision

-Recall

-F1-Score

Example output:

| Class             | Precision | Recall | F1-Score | Support |
| ----------------- | --------- | ------ | -------- | ------- |
| **0** (Not Churn) | 0.83      | 0.80   | 0.81     | XXXX    |
| **1** (Churn)     | 0.49      | 0.53   | 0.51     | YYYY    |



Churn class = 1 → more important to catch, so recall is valuable.

Results Summary:

| Model                | Best Metric               | Notes                             |
| -------------------- | ------------------------- | --------------------------------- |
| Logistic Regression  | Good recall & F1          | Stable after SMOTE                |
| Decision Tree        | Good interpretability     | Captures patterns                 |
| Neural Network (MLP) | Best balanced performance | Needs more iterations to converge |

How to Run the Project:

1. Clone the repository

git clone https://github.com/AfreenJenifer/customer_churn_prediction.git

2. Create virtual environment

python -m venv .venv
.\.venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the model

python churn_model.py

 Future Enhancements:

Try Random Forest, XGBoost, LightGBM

Hyperparameter tuning using GridSearchCV

Add confusion matrix, ROC curve

Deploy model using Flask / FastAPI

Add Streamlit web UI

 Author:

Afreen Jenifer A

Intern | Data Analysis Enthusiast

GitHub: https://github.com/AfreenJenifer
