# Credit Card Fraud Detection with a Neural Network (learning project)

Binary classifier detecting fraudulent transactions on a highly imbalanced dataset (about 0.17% fraud), with a focus on metrics suited to class imbalance and on the precision/recall trade-off.

> **Attribution.** This project is adapted from a public GitHub gist used as a learning base.
> Dataset: [Credit Card Fraud Detection, ULB Machine Learning Group (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Pipeline

1. Exploratory checks: class distribution, missing values, skew of transaction amounts
2. Standardisation of `Time` and `Amount` (PCA features V1 to V28 already scaled)
3. Stratified train/test split (80/20)
4. Class weighting to compensate for imbalance
5. Multilayer perceptron (Keras/TensorFlow): 64 and 32 ReLU units with dropout, sigmoid output
6. Training with early stopping on validation PR-AUC
7. Evaluation on the held-out test set: PR-AUC, classification report, confusion matrix
8. Decision-threshold tuning to reach a target recall (90%)
9. Ranking of the highest-risk transactions

## Why PR-AUC rather than accuracy

With less than 0.2% positives, a model predicting "no fraud" everywhere reaches over 99.8% accuracy. PR-AUC, precision and recall reflect performance on the minority class.

## Usage

```bash
pip install -r requirements.txt
# download creditcard.csv from Kaggle into src/
cd src
python fraud_nn.py
```

## My adaptations

<!-- List here only what you actually changed. -->

## Limitations

Single dataset with anonymised features, no temporal validation, no model persistence. Educational use only.
