#!/usr/bin/python3

# Title: "Student Performance Prediction with XAI (LIME & SHAP)"

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lime.lime_tabular
import shap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import joblib

# ----------------------------
# STEP-1: Load and preprocess dataset
# ----------------------------

df = pd.read_csv("Student_data/student-mat.csv", sep=";")

# Convert G3 to binary classification
df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Drop duplicates
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)

# Features and label
X = df.drop(columns=["G3"])
y = df["G3"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns.tolist()

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaled versions with column names
X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test = pd.DataFrame(X_test_scaled, columns=feature_names)

# ----------------------------
# Train models
# ----------------------------

def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        if name == "Random Forest":
            joblib.dump(model, 'model.pkl')

    # Save necessary artifacts
    joblib.dump(feature_names, "feature_names.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return trained

# ----------------------------
# Streamlit App
# ----------------------------

def predict():
    st.title("🎓 Student Pass/Fail Prediction")
    st.markdown("🔢 Enter **numeric, comma-separated** values matching the model's features.")
    st.caption("You need to enter **exactly 41 values**, in the same order used in training.")

    input_feature = st.text_input("📥 Input Features:")
    with st.expander("🧪 Need a sample input?"):
        st.code("18,1,1,0,1,0,1,1,1,0,0,0,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0", language="text")
        st.caption("🔎 Paste this into the input box above to test a valid prediction.")

    if st.button("🚀 Predict"):
        try:
            # Load artifacts
            model = joblib.load("model.pkl")
            feature_names = joblib.load("feature_names.pkl")
            scaler = joblib.load("scaler.pkl")

            # Parse input and check feature count
            features = list(map(float, input_feature.strip().split(",")))
            if len(features) != len(feature_names):
                st.error(f"❌ Expected {len(feature_names)} features, but got {len(features)}.")
                return

            # Convert to DataFrame and scale
            input_df = pd.DataFrame([features], columns=feature_names)
            input_scaled = scaler.transform(input_df)

            # Predict
            prediction = model.predict(input_scaled)
            result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"
            st.success(f"Prediction: {result}")

        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("⚠️ Make sure the input is properly formatted and complete.")

# ----------------------------
# Model Evaluation (console only)
# ----------------------------

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# ----------------------------
# Main Entry
# ----------------------------

if __name__ == "__main__":
    print("📊 Training models on student-mat.csv...")

    trained_models = train_models()

    for name, model in trained_models.items():
        evaluate_model(model, name)

    # Launch Streamlit app
    predict()
