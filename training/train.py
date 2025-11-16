import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import load_data, build_preprocess_pipeline

# Correct paths inside Docker container
DATA_PATH = "/app/data/telco_churn.csv"
MODEL_PATH = "/app/model/model.pkl"
PREPROCESSOR_PATH = "/app/model/preprocess.pkl"

def main():
    print("🔹 Loading data...")
    df = load_data(DATA_PATH)

    # Convert Churn column to binary
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    print("🔹 Building preprocessing pipeline...")
    preprocessor = build_preprocess_pipeline()

    print("🔹 Transforming training data...")
    X_transformed = preprocessor.fit_transform(X)

    print("🔹 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )

    print("🔹 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 MODEL METRICS:")
    print(f"  • Accuracy : {accuracy:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall   : {recall:.4f}")
    print(f"  • F1 Score : {f1:.4f}\n")

    # Save model and preprocessor
    print(f"🔹 Saving model to: {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"🔹 Saving preprocessor to: {PREPROCESSOR_PATH}")
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
