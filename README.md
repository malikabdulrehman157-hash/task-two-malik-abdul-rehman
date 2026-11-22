# Project 2 – Data Classification Using AI

DecodeLabs Artificial Intelligence Internship (Batch 2026)

## Goal
Build a basic supervised classification model on a small dataset (Iris: 150 samples, 3 classes, 4 features).

## Pipeline (Input → Process → Output)
1. **Load & understand** the Iris dataset: shape, class balance, missing values, feature ranges.
2. **Split** 80/20 into training and test sets, shuffled and stratified.
3. **Scale** with `StandardScaler`, fitted on the training data only.
4. **Choose K** for K-Nearest Neighbors with the elbow method (5-fold cross-validation on the training set).
5. **Train & predict**: `KNeighborsClassifier` → `fit` → `predict`.
6. **Validate**: accuracy, confusion matrix, precision, recall and F1 score.
7. **Experiment**: compare KNN with Logistic Regression and a Decision Tree, and classify new flowers.

## Results (random_state = 42)
| Metric | Value |
|---|---|
| Chosen K | 5 |
| Test accuracy | 93.3% |
| Macro F1 | 0.93 |

The only mistakes were 2 virginica flowers predicted as versicolor. Setosa was classified perfectly.

## How to run
```bash
pip install scikit-learn pandas matplotlib
python project2_iris_classification.py
```
Outputs `elbow_k.png` and `confusion_matrix.png`.
