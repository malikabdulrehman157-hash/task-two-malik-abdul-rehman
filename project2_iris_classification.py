"""
DecodeLabs AI Internship - Project 2: Data Classification Using AI
Supervised learning pipeline on the Iris benchmark (IPO framework).

INPUT   : Load and understand the Iris dataset (150 samples, 3 classes, 4 features)
PROCESS : Shuffled stratified 80/20 split -> StandardScaler -> K-Nearest Neighbors
          (K chosen with the elbow method using cross-validation on the training set only)
OUTPUT  : Accuracy, confusion matrix, precision / recall / F1 score, prediction on new data
"""

import matplotlib
matplotlib.use("Agg")  # save plots to files; remove this line to show windows in PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------- 1. INPUT
line("1. LOAD AND UNDERSTAND THE DATA")
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print(f"Shape: {df.shape[0]} samples x {len(iris.feature_names)} features")
print(f"Classes: {list(iris.target_names)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nClass balance:")
print(df["species"].value_counts())
print("\nMissing values:", int(df.isna().sum().sum()))
print("\nFeature statistics (note the different ranges -> scaling needed):")
print(df[iris.feature_names].describe().round(2))

X = iris.data
y = iris.target

# ---------------------------------------------------------------- 2. SPLIT
line("2. TRAIN / TEST SPLIT (80/20, shuffled, stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples (locked away until final evaluation)")

# ---------------------------------------------------------------- 3. CHOOSE K
line("3. TUNING THE ENGINE: CHOOSING K (elbow method)")
# Scaler sits inside the pipeline so it is fitted only on training folds (no data leakage).
k_values = range(1, 31)
cv_errors = []
for k in k_values:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    cv_errors.append(1 - scores.mean())

best_k = list(k_values)[int(np.argmin(cv_errors))]
print(f"Best K by 5-fold cross-validation: K = {best_k} "
      f"(CV error = {min(cv_errors):.3f})")

plt.figure(figsize=(8, 4.5))
plt.plot(list(k_values), cv_errors, marker="o")
plt.axvline(best_k, color="tab:orange", linestyle="--", label=f"Chosen K = {best_k}")
plt.xlabel("K (number of neighbours)")
plt.ylabel("Cross-validation error rate")
plt.title("Choosing K: Elbow Method")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_k.png", dpi=150)
plt.close()
print("Saved plot: elbow_k.png")

# ---------------------------------------------------------------- 4. TRAIN
line("4. TRAIN THE MODEL (instantiate -> fit -> predict)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on training data only
X_test_scaled = scaler.transform(X_test)        # reuse training mean/std

model = KNeighborsClassifier(n_neighbors=best_k)  # INSTANTIATE
model.fit(X_train_scaled, y_train)                # FIT
predictions = model.predict(X_test_scaled)        # PREDICT
print(f"KNN trained with K = {best_k}")

# ---------------------------------------------------------------- 5. OUTPUT
line("5. OUTPUT VALIDATION")
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
print(f"Accuracy:        {acc:.3f}")
print(f"Macro F1 score:  {f1:.3f}")

print("\nConfusion matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, predictions)
print(pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names))

print("\nClassification report (precision, recall, F1 per class):")
print(classification_report(y_test, predictions, target_names=iris.target_names))

disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title(f"KNN (K={best_k}) Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved plot: confusion_matrix.png")

# ---------------------------------------------------------------- 6. EXTRA
line("6. EXPERIMENT: COMPARE WITH OTHER ALGORITHMS")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

others = {
    f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
}
for name, clf in others.items():
    clf.fit(X_train_scaled, y_train)
    p = clf.predict(X_test_scaled)
    print(f"{name:22s} accuracy = {accuracy_score(y_test, p):.3f}   "
          f"macro F1 = {f1_score(y_test, p, average='macro'):.3f}")

line("7. PREDICT A COMPLETELY NEW FLOWER")
new_flowers = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2],   # looks like setosa
     [6.0, 2.9, 4.5, 1.5],   # looks like versicolor
     [6.9, 3.1, 5.8, 2.3]],  # looks like virginica
    columns=iris.feature_names,
)
new_pred = model.predict(scaler.transform(new_flowers))
for row, label in zip(new_flowers.values, new_pred):
    print(f"{row} -> {iris.target_names[label]}")

print("\nDone.")
