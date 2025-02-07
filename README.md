# Whale Classification with XGBoost

## Overview

This project involves classifying two types of whales (Gervais and Cuvier) using Gradient Boosting with the XGBoost library. The dataset consists of echo-location clicks of these whales, with preprocessing done using PySpark on a 10-node cluster. The preprocessed dataset is a NumPy array containing:

- Columns 0-9: Projections on the first 10 eigen vectors

- Column 10: RMSE

- Column 11: Peak2peak

- Column 12: Label (0 if species is Gervais, 1 if Cuvier)

The focus is on implementing XGBoost for classification, tuning parameters, computing confidence intervals with bootstrapped models, and analyzing tokenized data from tweets related to the classification task.

## Dataset & Preprocessing

### Data Processing

- Data preprocessing was handled in Data_Processing_Whales.ipynb.

- The dataset consists of 4,175 rows.

- Data is stored as NumPy arrays.

- The relevant files are:

  - X_train.pkl, X_test.pkl

  - y_train.pkl, y_test.pkl

## XGBoost - Theory

- XGBoost is a gradient boosting algorithm that optimizes decision trees in sequential iterations. Key documentation references:

  - XGBoost Model Guide

  - XGBoost Python API

### Key XGBoost Functions

1. xgboost.train - Training API

  - plst: Parameter list for XGBoost.

- dtrain: Training data.

- num_round: Number of boosting iterations.

- eval_list: Evaluation dataset.

- verbose_eval: Controls logging output.

2. bst.predict - Prediction API

- dtest: Test dataset.

- ntree_limit: Limit on the number of trees.

- output_margin: Outputs untransformed raw margin scores.



## Bootstrapped Predictions

Steps for Bootstrapping:

1. Sample bootstrap_size indices from the training set with replacement.

2. Create train and test data matrices using xgb.DMatrix.

3. Initialize evallist as [(dtrain, 'train'), (dtest, 'eval')].

4. Train the model using XGBoost.

5. Normalize output margin scores:
```
normalized = margin_scores / (max(scores) - min(scores))
```
6. Compute confidence intervals for each example.
```
def bootstrap_pred(X_train, X_test, y_train, y_test, n_bootstrap, minR, maxR, bootstrap_size, num_round, plst):
    all_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(len(X_train)), size=bootstrap_size, replace=True)
        X_sample = X_train[indices]
        y_sample = y_train[indices]
        dtrain = xgb.DMatrix(X_sample, label=y_sample)
        dtest = xgb.DMatrix(X_test, label=y_test)
        bst = xgb.train(plst, dtrain, num_round)
        margin_scores = bst.predict(dtest, output_margin=True)
        normalized = (margin_scores - np.min(margin_scores)) / (np.max(margin_scores) - np.min(margin_scores))
        all_scores.append(normalized)
    return np.array(all_scores)
```
## Calculating Confidence Intervals
```
def calc_stats(margin_scores):
    mean = np.mean(margin_scores, axis=1)
    std = np.std(margin_scores, axis=1)
    min_scr = np.round(mean - 3 * std, 2)
    max_scr = np.round(mean + 3 * std, 2)
    return min_scr, max_scr
```
## Making Predictions
```
def predict(min_scr, max_scr):
    predictions = np.zeros(min_scr.shape, dtype=int)
    predictions[(min_scr < 0) & (max_scr < 0)] = -1  # Cuvier’s whale
    predictions[(min_scr < 0) & (max_scr > 0)] = 0  # Unsure
    predictions[(min_scr > 0) & (max_scr > 0)] = 1  # Gervais whale
    return predictions
```
## Tokenizing Tweets for Feature Extraction

Steps:

1. Tokenize tweets using the Tokenizer class.

2. Count the number of times each token appears.

3. Compute token popularity and relative frequency.
```
def sum_values(values):
    total = 0
    for value in values:
        total += value
    return total
```
Filtering tokens that are mentioned by at least 100 users:
```
freq_tokens = overall_tokens.filter(lambda tok_count: tok_count[1] >= 100).cache()
num_freq_tokens = freq_tokens.count()
top20 = freq_tokens.sortBy(lambda tok_count: tok_count[1], ascending=False).collect()[:20]
```
## Runtime Performance Metrics

### Total execution time breakdown:
```
total time: 00:39
set up Spark context: 00:02
read data: 00:04
count unique users: 00:05
count tweets per user partition: 00:04
count all unique tokens: 00:04
count most popular tokens: 00:01
print popular tokens in each group: 00:20
```
## Conclusion

- Used XGBoost for classification.

- Implemented bootstrapped margin scores.

- Extracted popular tokens from tweets.

- Computed confidence intervals for predictions.

This project showcases machine learning and data science techniques for text-based and numerical classification using big data tools (PySpark) and boosting models (XGBoost).

