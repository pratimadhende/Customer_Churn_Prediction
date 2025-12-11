""""
Customer Churn Prediction for a Telecom / Service Company + Data-Driven Insights
Author: Pratima Dhende
Objectives: To develop an accurate churn prediction model and use explainability techniques 
to identify the key factors driving customer churn

"""
# -------------------
# Import Libraries 
# -------------------

import pandas as pd
import numpy as np
import seaborn as sns 
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier

# print("Loading Dataset....")
# df=pd.read_csv("Telco_Churn.csv")
csv_file = "Telco_Churn.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, csv_file)


print("Loading data from:", file_path)
df = pd.read_csv(file_path)
print("Dataset loaded. Shape:", df.shape)
print(df.head())


# print("Dataset Loaded.")
# print(df.head())

# --------------------------------
# Data Cleaning and Preprocessing
# --------------------------------
print("Data Cleaning....")
df=df.replace(" ",np.nan)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df["TotalCharges"].median(), inplace=True)

df=df.drop("customerID", axis=1)
# OR We use instead of this
# df.drop("CustomerID", axis=1, inplace=True)

for col in df.columns:
    df[col].dtype=="object"
    df[col]=LabelEncoder().fit_transform(df[col])

print("Cleaning Completed")

# ------
# Split
# ------
X=df.drop("Churn",axis=1)
y=df["Churn"]

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

# ----------------
# Model Trainning
# ----------------
print("Model Tranning...")

model=RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Train Completed")

# ----------
# Evalution
# ----------
Prediction=model.predict(X_test)

print("Accurate Score: ",accuracy_score(y_test,Prediction))
print("Classification Report: \n",classification_report(y_test,Prediction))

cm=confusion_matrix(y_test,Prediction)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------
# Feature Importance
# -------------------
importance=pd.Series(model.feature_importances_, index=X.columns)
importance=importance.sort_values(ascending=False)
plt.figure(figsize=(10,5))
importance[ :10].plot(kind="bar")
plt.title("Top 10 Feature Importance")
plt.ylabel("feature score")
plt.show()

print("'n Top 10 factors influencing churn: ")
print(importance[:10])

# --------------------
# SHAP EXPLAINABILITY
# --------------------
print("\nRunning SHAP explainability...")

# Sample for faster computation
sample_data = X_train.sample(200, random_state=42)

# Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# GLOBAL SUMMARY PLOT
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values[1], sample_data)

# GLOBAL BAR PLOT
shap.summary_plot(shap_values[1], sample_data, plot_type="bar")

# LOCAL EXPLANATION
print("\nGenerating SHAP force plot for a sample customer...")
sample_customer = X_test.iloc[0:1]

shap.initjs()
shap.force_plot(
    explainer.expected_value[1],
    explainer.shap_values(sample_customer)[1],
    sample_customer
)