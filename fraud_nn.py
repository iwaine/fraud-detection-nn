# Full fraud model pipeline that matches the script:
# 1) Load + quick EDA checks
# 2) Prepare features/target
# 3) Scale Time/Amount
# 4) Stratified split
# 5) Class weights (handle imbalance)
# 6) Build + compile neural network
# 7) Train w/ early stopping
# 8) Evaluate on NEVER-seen test set
# 9) Make probability predictions + threshold tradeoff

# DATA LINK: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, precision_recall_curve
)

import tensorflow as tf
from tensorflow import keras

# ----------------------------
# 1) Load dataset
# ----------------------------
df = pd.read_csv("creditcard.csv")  # or "creditcard.csv"
print("Shape:", df.shape)
print(df.head())

# ----------------------------
# 2) Quick EDA (high-value checks)
# ----------------------------
# Class imbalance
class_counts = df["Class"].value_counts()
print("\nClass counts:\n", class_counts)
print("\nClass %:\n", df["Class"].value_counts(normalize=True))

plt.figure()
class_counts.plot(kind="bar")
plt.title("Class Distribution (0 = Not Fraud, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Missing values
missing = df.isna().sum()
print("\nMissing values (top 10):\n", missing.sort_values(ascending=False).head(10))

# Amount distribution (skew)
plt.figure()
df["Amount"].hist(bins=50)
plt.title("Amount Distribution (Raw)")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.yscale("log")  # makes the skew visible
plt.show()

# ----------------------------
# 3) Target + features
# ----------------------------
y = df["Class"].astype(int).values
X = df.drop(columns=["Class"]).copy()

# ----------------------------
# 4) Scale Time + Amount
# ----------------------------
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# Visual: scaling doesn't change shape, just units
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Amount"], bins=50)
axes[0].set_title("Amount (Before Scaling)")
axes[0].set_yscale("log")

axes[1].hist(X["Amount"], bins=50)
axes[1].set_title("Amount (After Scaling)")
axes[1].set_yscale("log")
plt.show()

# Convert to numpy for TensorFlow
X = X.values.astype(np.float32)

# ----------------------------
# 5) Stratified train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Visual: fraud rate stays similar
train_rate = y_train.mean()
test_rate = y_test.mean()
plt.figure()
plt.bar(["Train", "Test"], [train_rate, test_rate])
plt.title("Fraud Rate: Train vs Test (Stratified)")
plt.ylabel("Fraud rate")
plt.show()

# ----------------------------
# 6) Class weights for imbalance
# ----------------------------
neg, pos = np.bincount(y_train)
class_weight = {0: 1.0, 1: neg / max(pos, 1)}
print("\nTrain fraud rate:", pos / (neg + pos))
print("Class weights:", class_weight)

plt.figure()
plt.bar(["Not Fraud (0)", "Fraud (1)"], [class_weight[0], class_weight[1]])
plt.title("Class Weights (Fraud is upweighted)")
plt.ylabel("Weight")
plt.show()

# ----------------------------
# 7) Build neural network (simple MLP)
# ----------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.summary()

# ----------------------------
# 8) Compile (metrics that matter for fraud)
# ----------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(curve="ROC", name="roc_auc"),
        keras.metrics.AUC(curve="PR", name="pr_auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    )
]

# ----------------------------
# 9) Train (watch validation metrics)
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=2048,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Visual: training progress (PR-AUC)
plt.figure()
plt.plot(history.history["pr_auc"], label="Train PR-AUC")
plt.plot(history.history["val_pr_auc"], label="Val PR-AUC")
plt.title("Training Progress (PR-AUC)")
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.legend()
plt.show()

# ----------------------------
# 10) Evaluate on NEVER-seen test set
# ----------------------------
y_prob = model.predict(X_test, verbose=0).ravel()

test_pr_auc = average_precision_score(y_test, y_prob)
print("\nTest PR-AUC:", test_pr_auc)


# Default threshold
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

print("\n=== Evaluation @ threshold = 0.5 ===")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Visual: probability separation + threshold line
plt.figure()
plt.hist(y_prob[y_test == 0], bins=50, alpha=0.6, label="Not Fraud (true 0)")
plt.hist(y_prob[y_test == 1], bins=50, alpha=0.6, label="Fraud (true 1)")
plt.axvline(threshold)
plt.title("Test Probabilities by True Class + Threshold")
plt.xlabel("P(fraud)")
plt.ylabel("Count")
plt.legend()
plt.show()

# ----------------------------
# 11) Tune threshold (tradeoff: catch more fraud vs fewer false alarms)
# ----------------------------
prec, rec, thr = precision_recall_curve(y_test, y_prob)

target_recall = 0.90
idxs = np.where(rec >= target_recall)[0]
if len(idxs) > 0:
    idx = idxs[-1]
    best_thr = thr[idx] if idx < len(thr) else 0.5
    print(f"\nThreshold for ~{int(target_recall*100)}% recall:", best_thr)

    y_pred_tuned = (y_prob >= best_thr).astype(int)
    print(f"\n=== Evaluation @ tuned threshold = {best_thr:.4f} ===")
    print(classification_report(y_test, y_pred_tuned, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))
else:
    print("\nCould not find a threshold achieving target recall with this model.")

# ----------------------------
# 12) “Make predictions” (show highest-risk transactions)
# ----------------------------
results = pd.DataFrame({
    "prob_fraud": y_prob,
    "true_class": y_test
}).sort_values("prob_fraud", ascending=False)

print("\nTop 10 highest-risk transactions (test set):")
print(results.head(10))
