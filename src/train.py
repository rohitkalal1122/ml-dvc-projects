import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier


# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

n_estimators = params["model"]["n_estimators"]
max_depth = params["model"]["max_depth"]


# Load training data
train_data = pd.read_csv("data/train.csv")

X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=params["train"]["random_state"]
)


# Train model
model.fit(X_train, y_train)


# Save model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training completed successfully.")
print(f"n_estimators: {n_estimators}")
print(f"max_depth: {max_depth}")