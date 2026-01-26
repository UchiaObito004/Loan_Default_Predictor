import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

st.title("Loan Approval Prediction App")


@st.cache_data
def load_data():
    df = pd.read_csv("data/loan_approval_dataset.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop(columns=[" loan_status", "loan_id"])
y = df[" loan_status"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include="object").columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

st.success(f"Model Accuracy: **{accuracy:.2f}%**")


st.subheader(" Predict Loan Approval")

user_input = {}

for col in X.columns:
    if col in num_cols:
        user_input[col] = st.number_input(col, float(X[col].min()), float(X[col].max()))
    else:
        user_input[col] = st.selectbox(col, X[col].unique())

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = pipeline.predict(input_df)
    result = le.inverse_transform(prediction)[0]

    if result.strip().lower() == "approved":
        st.success(" Loan Approved")
    else:
        st.error(" Loan Rejected")
