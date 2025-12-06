import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

# 1. LOAD DATA
df = pd.read_csv("data/customer_churn.csv")
print("Data shape:", df.shape)
print(df.head())

# 2. CLEANING

# Drop ID column if present
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Convert target to 0/1
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Handle TotalCharges special case (Telco dataset)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fill NaN with median value
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

print("\nMissing values per column after cleaning:")
print(df.isna().sum())

# 3. SPLIT X, y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode all categorical features
X = pd.get_dummies(X, drop_first=True)
print("\nAfter encoding:", X.shape)

# 4. TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. SMOTE ON TRAINING DATA (NO NaNs NOW)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nClass balance before SMOTE:")
print(y_train.value_counts())
print("\nClass balance after SMOTE:")
print(y_train_res.value_counts())

# 6. SCALE FEATURES
scaler = StandardScaler()
X_train_res_s = scaler.fit_transform(X_train_res)
X_test_s = scaler.transform(X_test)

# 7. TRAIN MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res_s, y_train_res)

# 8. EVALUATE
y_pred = model.predict(X_test_s)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

print("\n====================")
print("Decision Tree Model")
print("====================")

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_res_s, y_train_res)
y_pred_dt = dt.predict(X_test_s)
print(classification_report(y_test, y_pred_dt))

print("\n====================")
print("Neural Network (MLP) Model")
print("====================")

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
mlp.fit(X_train_res_s, y_train_res)
y_pred_mlp = mlp.predict(X_test_s)
print(classification_report(y_test, y_pred_mlp))
