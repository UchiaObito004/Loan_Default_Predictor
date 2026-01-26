# test_main.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def test_pipeline_creation():
    """Test ML pipeline builds correctly"""

    X = pd.DataFrame({
        "age": [25, 40, 35],
        "income": [50000, 80000, 60000],
        "gender": ["Male", "Female", "Male"]
    })

    y = ["Approved", "Rejected", "Approved"]

    num_cols = ["age", "income"]
    cat_cols = ["gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    pipeline.fit(X, y_encoded)

    preds = pipeline.predict(X)

    assert len(preds) == len(X)


def test_label_encoder_inverse():
    """Ensure label encoding works correctly"""

    le = LabelEncoder()
    y = ["Approved", "Rejected"]

    encoded = le.fit_transform(y)
    decoded = le.inverse_transform(encoded)

    assert list(decoded) == y
