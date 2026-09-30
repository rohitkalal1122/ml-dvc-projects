import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load test data
test_data = pd.read_csv("data/test.csv")

X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]


# Load trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)


# Make predictions
y_pred = model.predict(X_test)


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)


# Create metrics dictionary
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}


# Save metrics
with open("metrics/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# Display metrics
print("Evaluation completed successfully.")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")