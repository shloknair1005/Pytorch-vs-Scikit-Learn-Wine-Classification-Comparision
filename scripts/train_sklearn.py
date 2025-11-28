import json
from sklearn.datasets import load_wine
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from models.sklearn_model import create_pipeline

# Loading dataset
data = load_wine()

# Making data into dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)

# Assigning target column
df['target'] = data.target

# Checking out the data
print(df.head())
df.info()

# Converting data into X and y
X = df.drop(columns=["target"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = create_pipeline()
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

results = {
    "model": "Scikit-Learn Random Forest Classifier",
    "accuracy": float(accuracy),
    "n_estimators": 100
}

with open("../results/sklearn_results.json", "w") as f:
    json.dump(results, f, indent=4)

joblib.dump(pipeline, "../saved_models/sklearn_model.pkl")





