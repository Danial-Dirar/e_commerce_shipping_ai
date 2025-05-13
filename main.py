# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Step 1: Load Dataset
df = pd.read_csv("E-commerce Shipping Dataset.csv")
# Show class imbalance before preprocessing
plt.figure(figsize=(6, 4))
sns.countplot(x="Reached.on.Time_Y.N", data=df)
plt.title("Class Distribution (Before Preprocessing)")
plt.xlabel("Reached on Time (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.show()

# Step 2: Drop ID (non-predictive)
df.drop(columns=["ID"], inplace=True)


# Step 3: Encode Categorical Variables
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Step 5: Split Data
X = df.drop("Reached.on.Time_Y.N", axis=1)
y = df["Reached.on.Time_Y.N"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Step 6: Define Models
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Neural Network": MLPClassifier(max_iter=1000)
}

# Step 7: Train and Evaluate Models
results = {}
for name, model in models.items():
    scaler = MinMaxScaler() if name == "KNN" else StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "fpr": roc_curve(y_test, y_prob)[0],
        "tpr": roc_curve(y_test, y_prob)[1]
    }

# Step 8: Summary Table
summary_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "ROC AUC": res["roc_auc"]
    }
    for name, res in results.items()
])
print("\n=== Model Evaluation Summary ===")
print(summary_df)

# Step 9: Accuracy Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=summary_df)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.show()

# Step 10: Precision & Recall Bar Chart
melted_df = pd.melt(summary_df, id_vars="Model", value_vars=["Precision", "Recall"])
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="value", hue="variable", data=melted_df)
plt.title("Precision vs Recall Comparison")
plt.tight_layout()
plt.show()

# Step 11: Confusion Matrices
for name, res in results.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Step 12: ROC Curve Comparison
plt.figure(figsize=(10, 8))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()
