import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import json

MODEL_PATH = "models/ensemble_models.pkl"
USER_PROFILE_PATH = "models/user_profiles.json"  # store user self-assessment

# -----------------------------
# Load / Save User Profiles
# -----------------------------
def load_user_profiles():
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_user_profiles(profiles):
    os.makedirs("models", exist_ok=True)
    with open(USER_PROFILE_PATH, "w") as f:
        json.dump(profiles, f, indent=2)

# -----------------------------
# Data & Model Functions
# -----------------------------
def load_data():
    df = pd.read_csv("dataset.csv")
    X = df.drop("role", axis=1)
    y = df["role"]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X, y_encoded, encoder

def train_models():
    print("Training models... (first run only)")
    X, y, encoder = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()

    models = {
        "logistic": Pipeline([
            ("scaler", scaler),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "svm": Pipeline([
            ("scaler", scaler),
            ("model", SVC(probability=True))
        ]),
        "knn": Pipeline([
            ("scaler", scaler),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "naive_bayes": GaussianNB(),
        "ann": Pipeline([
            ("scaler", scaler),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500
            ))
        ])
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"trained: {name}")

    os.makedirs("models", exist_ok=True)
    joblib.dump((trained_models, encoder), MODEL_PATH)
    print("Models saved!")
    return trained_models, encoder

def load_models():
    if os.path.exists(MODEL_PATH):
        print("Loading trained models...")
        return joblib.load(MODEL_PATH)
    else:
        return train_models()

models, encoder = load_models()
user_profiles = load_user_profiles()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(input_data: dict, user_id: str = None):
    """
    Predict top 3 careers based on input_data.
    Also optionally stores input_data as user's self-assessment.
    """
    df = pd.DataFrame([input_data])

    combined_probs = None
    model_count = 0

    for name, model in models.items():
        try:
            probs = model.predict_proba(df)[0]
            if combined_probs is None:
                combined_probs = probs
            else:
                combined_probs += probs
            model_count += 1
        except Exception as e:
            print(f"{name} skipped:", e)

    if combined_probs is None:
        raise Exception("Input features do not match dataset columns")

    combined_probs = combined_probs / model_count
    top3_idx = np.argsort(combined_probs)[-3:][::-1]
    top3_roles = encoder.inverse_transform(top3_idx)
    top3_scores = combined_probs[top3_idx]

    recommendations = []
    for role, score in zip(top3_roles, top3_scores):
        recommendations.append({
            "career": role,
            "confidence": float(round(score, 3))
        })

    # Store self-assessment for personalization if user_id provided
    if user_id:
        user_profiles[user_id] = input_data
        save_user_profiles(user_profiles)

    return {"recommendations": recommendations}

# -----------------------------
# Get User Profile for AI Mentor
# -----------------------------
def get_user_profile(user_id: str):
    """
    Returns self-assessment scores for a given user_id
    """
    return user_profiles.get(user_id, None)

# -----------------------------
# Testing
# -----------------------------
if __name__ == "__main__":
    sample = {
        "logic": 7,
        "math": 8,
        "creativity": 5,
        "coding": 9,
        "communication": 6,
        "patience": 5,
        "curiosity": 8,
        "attention": 7,
        "risk_taking": 6,
        "visualization": 4
    }
    print(predict(sample, user_id="test_user"))
    print(get_user_profile("test_user"))
