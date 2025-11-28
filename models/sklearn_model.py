from sklearn.datasets import load_wine
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Loading dataset
data = load_wine()

# Making data into dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)

# Assigning target column
df['target'] = data.target

# Checking out the data
print("-"*100)
print(df.head())
df.info()


# Converting data into X and y
X = df.drop(columns=["target"], axis=1)
y = df["target"]


def create_pipeline():
    # Imputing and standardizing values from the data
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing the data
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, X.columns)
    ])

    # Joining the pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

    return pipeline
