# Supervised Learning Cheat Sheet

> Quick reference for writing pseudocode. Symbols, shapes, and update rules are stated explicitly so you can translate directly into code.

---

## Table of Contents

0. [Model Quick Reference](#model-quick-reference)
1. [Life Cycle of Supervised Models](#1-life-cycle-of-supervised-models)
   - [Metrics Overview](#metrics-overview)
2. [Supervised Models](#2-supervised-models)
   - [Linear Regression](#21-linear-regression)
   - [Logistic Regression](#22-logistic-regression)
   - [Decision Tree](#23-decision-tree)
   - [Random Forest](#24-random-forest)
   - [Support Vector Machine (SVM)](#25-support-vector-machine-svm)
   - [K-Nearest Neighbors (KNN)](#26-k-nearest-neighbors-knn)
   - [Naive Bayes](#27-naive-bayes)
   - [Neural Network (Feedforward)](#28-neural-network-feedforward)
   - [Gradient Boosting](#29-gradient-boosting)
3. [Dictionary](#3-dictionary) — [A–Z index](#accuracy)

---

## Model Quick Reference

> **Quick Reference** — Pick a model by scenario. Start with **Best choice**, fall back to **Alternatives** if constraints apply.

### By Task Type

| Task | Best choice | Alternatives |
|------|-------------|--------------|
| **Regression** — linear relationship, interpretability | [Linear Regression](#21-linear-regression) | [Ridge / Lasso](#l2-ridge) ([regularization](#regularization)) |
| **Regression** — non-linear, tabular data | [Gradient Boosting](#29-gradient-boosting) | [Random Forest](#24-random-forest), [Decision Tree](#23-decision-tree) |
| **Regression** — small dataset, local patterns | [KNN](#26-k-nearest-neighbors-knn) | [Linear Regression](#21-linear-regression) |
| **Classification** — need probabilities, linear boundary | [Logistic Regression](#22-logistic-regression) | [SVM](#25-support-vector-machine-svm) (Platt scaling for probs) |
| **Classification** — non-linear, tabular, high accuracy | [Gradient Boosting](#29-gradient-boosting) | [Random Forest](#24-random-forest), [Neural Network](#28-neural-network-feedforward) |
| **Classification** — text / sparse high-dimensional | [Naive Bayes](#27-naive-bayes) | [Logistic Regression](#22-logistic-regression), [Neural Network](#28-neural-network-feedforward) |
| **Classification** — images, sequences, complex patterns | [Neural Network](#28-neural-network-feedforward) | [Gradient Boosting](#29-gradient-boosting) (tabular only) |

### By Scenario

| Scenario | Best choice | Why |
|----------|-------------|-----|
| Need **interpretable rules** (if-then) | [Decision Tree](#23-decision-tree) | Human-readable splits; single tree |
| Need **interpretable coefficients** | [Linear](#21-linear-regression) / [Logistic Regression](#22-logistic-regression) | Weight per feature |
| **Tabular data** — robust default | [Random Forest](#24-random-forest) | Handles mixed types, resistant to [overfitting](#overfitting) |
| **Tabular data** — max accuracy | [Gradient Boosting](#29-gradient-boosting) | State-of-the-art on structured data |
| **Small dataset** (< few thousand rows) | [KNN](#26-k-nearest-neighbors-knn), [SVM](#25-support-vector-machine-svm) | Work with limited samples |
| **High-dimensional** data (many features) | [SVM](#25-support-vector-machine-svm), [Naive Bayes](#27-naive-bayes) | Kernel trick / sparse text |
| **Large dataset** (millions of rows) | [Linear](#21-linear-regression) / [Logistic Regression](#22-logistic-regression), [Gradient Boosting](#29-gradient-boosting) | Scalable; avoid KNN |
| **No training time** (lazy inference) | [KNN](#26-k-nearest-neighbors-knn) | Store data only at fit |
| **Fast baseline / prototype** | [Logistic Regression](#22-logistic-regression), [Naive Bayes](#27-naive-bayes) | Quick to train and tune |
| **Class imbalance** | [Gradient Boosting](#29-gradient-boosting), [Random Forest](#24-random-forest) | Use class weights; optimize [F1-score](#f1-score) / [ROC-AUC](#roc-auc) |
| **Feature importance** needed | [Random Forest](#24-random-forest), [Gradient Boosting](#29-gradient-boosting) | Built-in importance scores |
| **Mixed feature types** (numeric + categorical) | [Decision Tree](#23-decision-tree), [Random Forest](#24-random-forest) | No encoding required for trees |
| **Clear margin** between classes | [SVM](#25-support-vector-machine-svm) | Maximizes separation |
| **Free internal validation** (no holdout) | [Random Forest](#24-random-forest) | [OOB](#oob-out-of-bag) error estimate |

### Decision Flow

```text
START
├─ Target continuous?
│   ├─ Linear? → Linear Regression
│   ├─ Tabular, need accuracy? → Gradient Boosting → Random Forest
│   └─ Very small n? → KNN
│
└─ Target categorical?
    ├─ Need interpretability? → Decision Tree → Logistic Regression
    ├─ Text / sparse features? → Naive Bayes → Logistic Regression
    ├─ Tabular, need accuracy? → Gradient Boosting → Random Forest
    ├─ Images / deep patterns? → Neural Network
    ├─ Small n, high-dim? → SVM
    └─ Quick baseline? → Logistic Regression
```

---

## 1. Life Cycle of Supervised Models

### Life Cycle

#### 1. Data Collection and Preparation

| Step | Pseudocode skeleton |
|------|---------------------|
| **Data Cleaning** | `remove_duplicates(X, y)` → `handle_missing(X, strategy)` → `clip_outliers(X, bounds)` → `encode_categorical(X)` |
| **[Feature Engineering](#feature-engineering)** | `scale(X)` → `create_polynomial_features(X)` → `select_features(X, y, method)` |

```text
FUNCTION prepare_data(raw_X, raw_y):
    X, y ← clean(raw_X, raw_y)
    X ← engineer_features(X)
    RETURN X, y
```

**Common cleaning strategies**

| Issue | Strategy |
|-------|----------|
| Missing values | [imputation](#imputation): mean/median (numeric), mode impute (categorical), drop row |
| Outliers | IQR clip, z-score threshold, winsorize |
| Duplicates | drop or aggregate |
| Categorical | [one-hot encode](#one-hot-encoding), label encode, target encode |

**Common feature transforms**

| Transform | When to use |
|-----------|-------------|
| StandardScaler | features on different scales (SVM, KNN, NN) |
| MinMaxScaler | bounded input needed (NN, some distance metrics) |
| Log / Box-Cox | right-skewed distributions |
| Polynomial features | linear models need non-linearity |

---

#### 2. Data Split

```text
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25, random_state=42)
# Training Set: 3/4 of the data
# Testing Set:  1/4 of the data
```

**[Cross-validation](#cross-validation) techniques**

| Method | Pseudocode idea | Use when |
|--------|-----------------|----------|
| **[k-fold](#k-fold)** | Split into k folds; train on k−1, validate on 1; rotate k times; average metric | Stable estimate of generalization; moderate data size |
| **[Stratified k-fold](#k-fold)** | [k-fold](#k-fold) but each fold preserves class proportions | Classification with imbalanced classes |
| **[Bootstrapping](#bootstrapping)** | Sample n points with replacement → train; out-of-bag ([OOB](#oob-out-of-bag)) points → validate; repeat B times | [Bagging](#bagging) ensembles; uncertainty estimation |
| **Leave-one-out (LOO)** | k = n (special case of [k-fold](#k-fold)) | Very small datasets |

```text
FUNCTION k_fold_cv(X, y, model, k=5):
    scores ← []
    FOR each fold (train_idx, val_idx) in k_folds(X, y, k):
        model.fit(X[train_idx], y[train_idx])
        scores.append(metric(y[val_idx], model.predict(X[val_idx])))
    RETURN mean(scores), std(scores)
```

---

#### 3. Model Training

```text
FUNCTION train(X_train, y_train, algorithm, hyperparams):
    model ← algorithm(**hyperparams)
    model.fit(X_train, y_train)
    RETURN model
```

**[Hyperparameter](#hyperparameter) Tuning** — optimizing *external* parameters (not learned from data)

| Method | Pseudocode idea | Pros / Cons |
|--------|-----------------|-------------|
| **[Grid Search](#grid-search)** | Try every combination in a predefined grid | Exhaustive, slow on large grids |
| **[Random Search](#random-search)** | Sample random combinations from distributions | Faster; often finds good params with fewer trials |

```text
FUNCTION grid_search(X, y, model, param_grid, cv=5):
    best_score, best_params ← −∞, None
    FOR params IN cartesian_product(param_grid):
        score ← k_fold_cv(X, y, model.set_params(**params), cv)
        IF score > best_score:
            best_score, best_params ← score, params
    RETURN best_params

FUNCTION random_search(X, y, model, param_distributions, n_iter, cv=5):
    best_score, best_params ← −∞, None
    FOR i IN 1..n_iter:
        params ← sample(param_distributions)
        score ← k_fold_cv(X, y, model.set_params(**params), cv)
        IF score > best_score:
            best_score, best_params ← score, params
    RETURN best_params
```

---

#### 4. Model Testing and Evaluation

```text
FUNCTION evaluate(model, X_test, y_test, task):
    y_pred ← model.predict(X_test)
    RETURN compute_metrics(y_test, y_pred, task)
```

##### Metrics Overview

A [metric](#metric) is a single number that compares model predictions $\hat{y}$ (or $\hat{p}$) against ground truth $y$. Choose the metric that matches your task and business goal — see definitions in the [Dictionary](#3-dictionary).

**Regression metrics** measure how far continuous predictions are from true values:

**[MSE](#mse)**

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**[RMSE](#rmse)**

$$\text{RMSE} = \sqrt{\text{MSE}}$$

**[MAE](#mae)**

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**[R²](#r2)**

$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

**Classification metrics** measure label correctness. For each class $c$, define true positives $\text{TP}_c$, false positives $\text{FP}_c$, and false negatives $\text{FN}_c$ from the [confusion matrix](#confusion-matrix):

**[Accuracy](#accuracy)**

$$\text{Accuracy} = \frac{N_{\text{correct}}}{n}$$

**[Precision](#precision)**

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**[Recall](#recall)**

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**[F1-score](#f1-score)**

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**[ROC-AUC](#roc-auc)**

Use **macro** averaging (unweighted mean across classes) when every class matters equally; use **micro** averaging (pool all TP/FP/FN) when overall sample counts dominate.

---

#### 5. Model Deployment

```text
FUNCTION deploy(model, artifacts):
    save(model, "model.pkl")
    save(scaler, "scaler.pkl")          # preprocessing pipeline
    save(feature_names, "features.json")

# Serving via API
FUNCTION predict_endpoint(request):
    X ← preprocess(request.features, scaler)
    RETURN {"prediction": model.predict(X), "probability": model.predict_proba(X)}
```

| Channel | Pattern |
|---------|---------|
| REST API | Flask / FastAPI endpoint wrapping `model.predict` |
| Batch | Scheduled job reads input table → writes predictions |
| Cloud | AWS SageMaker, GCP Vertex AI, Azure ML — upload [artifact](#artifact) + endpoint config |

---

## 2. Supervised Models

### 2.1 Linear Regression

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Continuous target; roughly linear relationship; interpretability needed |
| **Cons** | Sensitive to outliers; assumes linearity; multicollinearity hurts coefficient interpretation |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | feature matrix | (n, d) |
| y | target vector | (n,) |
| w | weight vector (incl. bias as w₀ with x₀=1) | (d+1,) or (d,) + b |
| ŷ | prediction | (n,) |
| n | number of samples | scalar |
| d | number of features | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Assumption** — linear model with homoscedastic Gaussian noise  
   $y = Xw + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0,\sigma^2)$

2. **Prediction**  
   $\hat{y} = Xw \quad\text{or}\quad \hat{y} = w^\top x + b$

3. **Loss (<a href="#mse">MSE</a>)**  
   $\mathcal{L}(w) = \dfrac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \dfrac{1}{n}\|y - Xw\|^2$

4. **Objective**  
   $w^* = \arg\min_w \mathcal{L}(w)$

5. **Closed form** (when $X^\top X$ is invertible)  
   $w^* = (X^\top X)^{-1} X^\top y$

</details>

<details>
<summary><strong>d. Update Rules</strong> · <a href="#gradient-descent">gradient descent</a></summary>

1. Initialize weights $w$

2. **Forward** — compute predictions  
   $\hat{y} = Xw$

3. **Gradient** of loss  
   $\nabla_w \mathcal{L} = -\dfrac{2}{n}\, X^\top (y - \hat{y})$

4. **Update** with <a href="#learning-rate">learning rate</a> $\eta$  
   $w \leftarrow w - \eta \nabla_w \mathcal{L}$

5. Repeat steps 2–4 for each <a href="#epoch">epoch</a> until convergence

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| learning_rate | 1e-4 – 1.0 | too large → diverge |
| <a href="#regularization">regularization</a> (Ridge λ, Lasso λ) | 1e-4 – 1e2 | controls <a href="#overfitting">overfitting</a> |
| max_epochs / tol | — | <a href="#early-stopping">early stopping</a> on validation loss |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#r2">R²</a> (goodness of fit), <a href="#mse">MSE</a> / <a href="#rmse">RMSE</a> (penalizes large errors), <a href="#mae">MAE</a> (robust to outliers). See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split (see Life Cycle §1–2)
(X_train, X_test, y_train, y_test) ← split(scale(X), y, test_size=0.25)

# Train
model ← LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
R2 ← model.score(X_test, y_test)
MSE ← mean((y_test - y_pred)^2)
```

</details>

---

### 2.2 Logistic Regression

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Binary or multiclass classification; need probabilities; linear decision boundary |
| **Cons** | Cannot model complex non-linear boundaries without <a href="#feature-engineering">feature engineering</a> |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | features | (n, d) |
| y | labels {0,1} or one-hot | (n,) or (n, C) |
| w | weights | (d,) + b |
| z | linear score | (n,) |
| ŷ | predicted probability | (n,) |
| C | number of classes | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Assumption** — log-odds are linear in features  
   $\mathbb{P}(y{=}1 \mid x) = \sigma(w^\top x + b)$

2. **<a href="#sigmoid">Sigmoid</a>**  
   $\sigma(z) = \dfrac{1}{1 + e^{-z}} \in (0,1)$

3. **Prediction** — class 1 if $\hat{p} \geq 0.5$

4. **Loss (<a href="#log-loss">cross-entropy</a>)**  
   $\mathcal{L}(w,b) = -\dfrac{1}{n}\sum_{i=1}^{n}\bigl[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\bigr]$

5. **Objective** (with optional <a href="#l2-ridge">L2</a> penalty)  
   $w^*, b^* = \arg\min_{w,b}\; \mathcal{L}(w,b) + \lambda\|w\|^2$

</details>

<details>
<summary><strong>d. Update Rules</strong> · <a href="#gradient-descent">gradient descent</a></summary>

1. **Linear score**  
   $z = Xw + b$

2. **Probability**  
   $\hat{y} = \sigma(z)$

3. **Gradients**  
   $\nabla_w \mathcal{L} = \dfrac{1}{n}\, X^\top (\hat{y} - y)$  
   $\dfrac{\partial \mathcal{L}}{\partial b} = \dfrac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$

4. **Update**  
   $w \leftarrow w - \eta(\nabla_w \mathcal{L} + \lambda w)$  
   $b \leftarrow b - \eta \dfrac{\partial \mathcal{L}}{\partial b}$

5. Repeat until convergence

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range |
|----------------|---------------|
| C (= 1/λ) | 1e-3 – 1e3 |
| solver | lbfgs, saga, liblinear |
| class_weight | None, balanced |
| max_iter | 100 – 1000 |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#accuracy">Accuracy</a>, <a href="#precision">Precision</a>, <a href="#recall">Recall</a>, <a href="#f1-score">F1-score</a>, <a href="#roc-auc">ROC-AUC</a>, and <a href="#log-loss">Log Loss</a> (directly matches the training objective). See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split
(X_train, X_test, y_train, y_test) ← split(scale(X), y, test_size=0.25)

# Train
model ← LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
y_proba ← model.predict_proba(X_test)[:, 1]
accuracy ← metric(y_test, y_pred)
```

</details>

---

### 2.3 Decision Tree

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Mixed feature types; non-linear boundaries; interpretability via rules |
| **Cons** | High <a href="#variance">variance</a>; overfits easily; unstable to small data changes |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| X | (n, d) feature matrix |
| y | (n,) targets |
| node | subset of samples S ⊆ {1..n} |
| feature j, threshold t | split candidate |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Assumption** — piecewise constant prediction in each leaf

2. **Regression impurity (<a href="#mse">MSE</a> at node $S$)**  
   $\text{MSE}(S) = \dfrac{1}{|S|}\sum_{i\in S}(y_i - \bar{y}_S)^2$

3. **Classification impurity (<a href="#gini">Gini</a>)**  
   $G = 1 - \sum_k p_k^2$

4. **Classification impurity (<a href="#entropy">Entropy</a>)**  
   $H = -\sum_k p_k \log_2 p_k$

5. **Split criterion (<a href="#information-gain">information gain</a>)**  
   $\text{IG} = I(S) - \dfrac{|S_L|}{|S|}I(S_L) - \dfrac{|S_R|}{|S|}I(S_R)$  
   where $S_L = \{i \in S : x_{ij} \leq t\}$, $S_R$ is the complement

6. **Leaf prediction**  
   $\hat{y} = \bar{y}_S$ (regression) or majority class (classification)

</details>

<details>
<summary><strong>d. Update Rules</strong> (recursive split)</summary>

1. **Input** — sample set $S$ at current node

2. **Stop?** — if max depth, too few samples, or pure node → return **Leaf**$(\text{aggregate}(S))$

3. **Best split**  
   $(j^*, t^*) = \arg\max_{j,t}\; \text{IG}(S, j, t)$

4. **Partition**  
   $S_L \leftarrow \{i \in S : x_{i,j^*} \leq t^*\}$  
   $S_R \leftarrow S \setminus S_L$

5. **Recurse**  
   $\text{Node}(j^*, t^*,\; \text{build}(S_L),\; \text{build}(S_R))$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Effect |
|----------------|--------|
| max_depth | limits tree height |
| min_samples_split | min samples to split a node |
| min_samples_leaf | min samples per leaf |
| max_features | random subset of features per split (→ Random Forest behavior) |
| criterion | gini, <a href="#entropy">entropy</a> (classification); mse, mae (regression) |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

MSE / <a href="#r2">R²</a> for regression; <a href="#accuracy">Accuracy</a> / <a href="#f1-score">F1-score</a> for classification. See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25)

# Train
model ← DecisionTreeClassifier(max_depth=5, criterion="gini", min_samples_leaf=5)
# model ← DecisionTreeRegressor(max_depth=5, criterion="mse")
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
rules ← export_text(model, feature_names)
```

</details>

---

### 2.4 Random Forest

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Tabular data; need robustness; non-linear patterns; feature importance |
| **Cons** | Less interpretable than single tree; slower <a href="#inference">inference</a> than linear models; memory heavy |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| B | number of trees |
| m | features sampled per split (typically √d or d/3) |
| tree_b | b-th decision tree |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Assumption** — <a href="#ensemble">ensemble</a> of decorrelated trees reduces <a href="#variance">variance</a>

2. **Bootstrap training** — tree $b$ fits on sample $(X_b, y_b)$ drawn with replacement

3. **Feature subsampling** — at each split, consider random subset of $m$ features

4. **Regression prediction**  
   $\hat{y} = \dfrac{1}{B}\sum_{b=1}^{B} \text{tree}_b(x)$

5. **Classification prediction** — majority vote or average class probabilities

6. **<a href="#oob-out-of-bag">OOB</a> validation** — samples not in bootstrap provide free error estimate

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Initialize empty forest $\mathcal{F} = \{\}$

2. **For** $b = 1, \ldots, B$:

   - Draw <a href="#bootstrap">bootstrap</a> sample $(X_b, y_b)$
   - Build tree$_b$ on $(X_b, y_b)$ with $m$-feature subsampling
   - $\mathcal{F} \leftarrow \mathcal{F} \cup \{\text{tree}_b\}$

3. **Predict**$(x)$ — aggregate $\{\text{tree}_b(x)\}$: mean (regression) or vote (classification)

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range |
|----------------|---------------|
| n_estimators (B) | 100 – 500 |
| max_depth | 5 – 30 or None |
| max_features (m) | sqrt(d), log2(d) |
| min_samples_leaf | 1 – 10 |
| <a href="#bootstrap">bootstrap</a> | True |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Same as the underlying task, plus <a href="#oob-out-of-bag">OOB</a> score for internal validation. See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25)

# Train
model ← RandomForestClassifier(n_estimators=200, max_features="sqrt", oob_score=True)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
importance ← model.feature_importances_
oob ← model.oob_score_
```

</details>

---

### 2.5 Support Vector Machine (SVM)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | High-dimensional data; clear margin separation; small-to-medium n |
| **Cons** | Slow on large n; sensitive to feature scaling; kernel choice matters |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| xᵢ | sample | (d,) |
| w | weight vector | (d,) |
| b | bias | scalar |
| αᵢ | dual coefficients | (n,) |
| C | <a href="#regularization">regularization</a> | scalar |
| ξᵢ | slack variables | (n,) |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Goal** — hyperplane with maximum margin between classes

2. **Hard-margin objective**  
   $\min \dfrac{1}{2}\|w\|^2 \quad\text{s.t.}\quad y_i(w^\top x_i + b) \geq 1$

3. **Soft-margin** (slack $\xi_i \geq 0$, penalty $C$)  
   $\min_{w,b,\xi}\; \dfrac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(w^\top x_i + b) \geq 1 - \xi_i$

4. **Hinge loss equivalent**  
   $\mathcal{L} = \sum_i \max(0,\; 1 - y_i(w^\top x_i + b))$

5. **<a href="#kernel-trick">Kernel trick</a>** — replace $x_i^\top x_j$ with $K(x_i, x_j)$

6. **Decision function**  
   $\text{sign}\!\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$

</details>

<details>
<summary><strong>d. Update Rules</strong> (primal SGD)</summary>

1. **For each** sample $(x_i, y_i)$:

2. **Margin check**  
   $m_i = y_i(w^\top x_i + b)$

3. **If** $m_i < 1$ (hinge active):  
   $w \leftarrow w - \eta(w - C\, y_i x_i)$

4. **Else** (hinge inactive):  
   $w \leftarrow w - \eta w$

5. Dual solvers (SMO) adjust $\alpha_i$ until KKT conditions hold

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Notes |
|----------------|-------|
| C | trade-off margin vs. misclassification |
| kernel | linear, rbf, poly |
| gamma (RBF) | 1e-3 – 1e1; inverse influence radius |
| degree (poly) | 2 – 5 |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#accuracy">Accuracy</a> and <a href="#f1-score">F1-score</a> for classification; SVR (regression variant) uses $\varepsilon$-insensitive loss instead. See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split — scale features
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25)
X_train ← scaler.fit_transform(X_train)
X_test ← scaler.transform(X_test)

# Train
model ← SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
```

</details>

---

### 2.6 K-Nearest Neighbors (KNN)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Small datasets; local patterns; no training phase needed |
| **Cons** | Slow prediction O(nd) per query; curse of dimensionality; requires feature scaling |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| X_train | (n, d) stored training set |
| k | number of neighbors |
| N_k(x) | set of k closest training points to x |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Assumption** — local smoothness: similar inputs → similar outputs

2. **Neighbor set** — $N_k(x)$ = $k$ closest training points to query $x$

3. **Euclidean distance**  
   $d(x,x') = \sqrt{\sum_j (x_j - x'_j)^2}$

4. **Manhattan distance**  
   $d(x,x') = \sum_j |x_j - x'_j|$

5. **Minkowski distance**  
   $d(x,x') = \left(\sum_j |x_j - x'_j|^p\right)^{1/p}$

6. **Regression prediction**  
   $\hat{y} = \dfrac{1}{k}\sum_{i \in N_k(x)} y_i$

7. **Classification prediction** — majority vote in $N_k(x)$

8. **Distance weighting** — $w_i = 1 / d(x, x_i)$

</details>

<details>
<summary><strong>d. Update Rules</strong> · <a href="#lazy-learner">lazy learner</a></summary>

1. **Fit** — store $(X_{\text{train}}, y_{\text{train}})$; no parameter update

2. **For query** $x$ — compute $d(x, x_i)$ for all training points

3. **Select** $k$ neighbors with smallest distance

4. **Aggregate** labels: mean / vote / weighted average

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range |
|----------------|---------------|
| k | 1 – 20 (odd for binary classification to avoid ties) |
| metric | euclidean, manhattan, minkowski |
| weights | uniform, distance |
| algorithm | brute, kd_tree, ball_tree |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Same as task type (MSE / <a href="#r2">R²</a> or <a href="#accuracy">Accuracy</a> / <a href="#f1-score">F1-score</a>). See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split — scale features
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25)
X_train ← scaler.fit_transform(X_train)
X_test ← scaler.transform(X_test)

# Train (stores data only)
model ← KNeighborsClassifier(n_neighbors=5, weights="distance")
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
```

</details>

---

### 2.7 Naive Bayes

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Text classification; high-dimensional sparse data; fast baseline |
| **Cons** | Independence assumption often violated; poor probability calibration |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| x | feature vector (d,) |
| y | class label |
| P(y) | class prior |
| P(xⱼ\|y) | likelihood of feature j given class |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Bayes rule**  
   $P(y \mid x) = \dfrac{P(x \mid y)\, P(y)}{P(x)}$

2. **Naive assumption** (features independent given class)  
   $P(x \mid y) = \prod_j P(x_j \mid y)$

3. **Decision rule**  
   $\hat{y} = \arg\max_y\; P(y) \prod_j P(x_j \mid y)$

4. **Gaussian NB**  
   $P(x_j \mid y) = \mathcal{N}(\mu_{jy},\, \sigma_{jy}^2)$

5. **Multinomial NB**  
   $P(x_j \mid y) \propto \theta_{jy}^{x_j}$

6. **Bernoulli NB**  
   $P(x_j \mid y) = \theta_{jy}^{x_j}(1-\theta_{jy})^{1-x_j}$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. **Estimate priors**  
   $P(y{=}c) = \dfrac{\text{count}(y{=}c)}{n}$

2. **Estimate likelihoods** per class $c$ and feature $j$ (e.g. $\mu_{jc}$, $\sigma_{jc}$ for Gaussian)

3. **Predict** — log-score for numerical stability:  
   $\text{score}(c) = \log P(y{=}c) + \sum_j \log P(x_j \mid y{=}c)$

4. **Output**  
   $\hat{y} = \arg\max_c\; \text{score}(c)$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Notes |
|----------------|-------|
| var_smoothing | additive smoothing for Gaussian (1e-9 – 1e-5) |
| alpha | Laplace smoothing for Multinomial/Bernoulli |
| fit_prior | learn P(y) or uniform |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#accuracy">Accuracy</a> and <a href="#f1-score">F1-score</a> (macro-F1 preferred for imbalanced text classes). See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare — X is word counts or TF-IDF
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25)

# Train
model ← MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
y_proba ← model.predict_proba(X_test)
```

</details>

---

### 2.8 Neural Network (Feedforward)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Complex non-linear patterns; images, text, sequences (with extensions) |
| **Cons** | Needs lots of data; black box; tuning-heavy; GPU helps |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | input batch | (n, d₀) |
| Wˡ | weight matrix layer l | (dₗ₋₁, dₗ) |
| bˡ | bias layer l | (dₗ,) |
| zˡ | pre-activation | (n, dₗ) |
| aˡ | activation | (n, dₗ) |
| L | number of layers | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Layer forward** ($a^0 = X$)  
   $z^l = a^{l-1} W^l + b^l$  
   $a^l = g(z^l)$

2. **Activations** — <a href="#relu">ReLU</a>: $g(z)=\max(0,z)$; <a href="#sigmoid">Sigmoid</a>: $\sigma(z)$; <a href="#softmax">Softmax</a> at output

3. **Regression loss (<a href="#mse">MSE</a>)**  
   $\mathcal{L} = \dfrac{1}{n}\sum_i (y_i - \hat{y}_i)^2$

4. **Classification loss (<a href="#log-loss">cross-entropy</a>)**  
   $\mathcal{L} = -\dfrac{1}{n}\sum_i \sum_k y_{ik}\log\hat{y}_{ik}$

5. **Objective**  
   $\theta^* = \arg\min_\theta \mathcal{L}(\theta)$ over all $W^l, b^l$

</details>

<details>
<summary><strong>d. Update Rules</strong> · <a href="#gradient-descent">backpropagation</a></summary>

1. **Forward pass** — compute $z^l, a^l$ for $l = 1,\ldots,L$; evaluate $\mathcal{L}$

2. **Output error**  
   $\delta^L = \dfrac{\partial \mathcal{L}}{\partial a^L} \odot g'(z^L)$

3. **Backward** for $l = L{-}1, \ldots, 1$:  
   $\dfrac{\partial \mathcal{L}}{\partial W^l} = \dfrac{1}{n}(a^{l-1})^\top \delta^l$  
   $\dfrac{\partial \mathcal{L}}{\partial b^l} = \dfrac{1}{n}\sum_i \delta_i^l$  
   $\delta^{l-1} = \delta^l (W^l)^\top \odot g'(z^{l-1})$

4. **Update** with <a href="#learning-rate">learning rate</a> $\eta$:  
   $W^l \leftarrow W^l - \eta \dfrac{\partial \mathcal{L}}{\partial W^l}$  
   $b^l \leftarrow b^l - \eta \dfrac{\partial \mathcal{L}}{\partial b^l}$

5. Repeat for each <a href="#epoch">epoch</a> / mini-<a href="#batch">batch</a>

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range |
|----------------|---------------|
| hidden layers / units | 1–3 layers; 32–512 units |
| learning_rate | 1e-5 – 1e-1 (with scheduler) |
| batch_size | 16 – 256 |
| <a href="#dropout">dropout</a> | 0.1 – 0.5 |
| optimizer | Adam, SGD+momentum |
| <a href="#early-stopping">early stopping</a> | monitor validation loss |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Task-dependent (<a href="#mse">MSE</a> / <a href="#r2">R²</a> or <a href="#accuracy">Accuracy</a> / <a href="#f1-score">F1-score</a>); monitor training vs. validation loss to detect <a href="#overfitting">overfitting</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split — scale features
(X_train, X_test, y_train, y_test) ← split(scale(X), y, test_size=0.25)

# Train
model ← MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      learning_rate_init=1e-3, max_iter=200, early_stopping=True)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
loss_curve ← model.loss_curve_
```

</details>

---

### 2.9 Gradient Boosting

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Tabular data; state-of-the-art accuracy on structured problems |
| **Cons** | Slow to train; many <a href="#hyperparameter">hyperparameters</a>; can <a href="#overfitting">overfit</a> if not regularized |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| M | number of <a href="#boosting">boosting</a> rounds (trees) |
| η | <a href="#learning-rate">learning rate</a> (shrinkage) |
| h_m | <a href="#weak-learner">weak learner</a> (shallow tree) at round m |
| F_m | <a href="#ensemble">ensemble</a> model after m rounds |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Assumption** — additive <a href="#ensemble">ensemble</a> of <a href="#weak-learner">weak learners</a> corrects residuals

2. **Initialize**  
   $F_0(x) = \arg\min_\gamma \sum_i \mathcal{L}(y_i, \gamma)$

3. **Additive update** at round $m$  
   $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$

4. **Pseudo-residuals** (negative gradient)  
   $r_i = -\dfrac{\partial \mathcal{L}}{\partial F}(x_i)$

5. **Squared-error residual**  
   $r_i = y_i - F_{m-1}(x_i)$

6. **Logistic residual**  
   $r_i = y_i - \sigma(F_{m-1}(x_i))$

7. **Final prediction**  
   $\hat{y} = F_M(x)$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. **Initialize** $F$ ← constant prediction (e.g. $\bar{y}$)

2. **For** $m = 1, \ldots, M$:

   - Compute <a href="#boosting">pseudo-residuals</a>: $r_i = -\dfrac{\partial \mathcal{L}}{\partial F}(x_i)$
   - Fit shallow tree $h_m$ to $(X, r)$
   - $F \leftarrow F + \eta \cdot h_m$

3. **Predict**$(x)$ — return $F_M(x)$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range |
|----------------|---------------|
| n_estimators (M) | 100 – 1000 |
| learning_rate (η) | 0.01 – 0.3 |
| max_depth | 3 – 8 |
| subsample | 0.5 – 1.0 |
| colsample_bytree | 0.5 – 1.0 |
| reg_lambda, reg_alpha | L2/L1 on leaf weights |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Same as task type; use a validation set or <a href="#early-stopping">early stopping</a> on a chosen <a href="#metric">metric</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>



<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare & split
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25)

# Train
model ← GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred ← model.predict(X_test)
y_proba ← model.predict_proba(X_test)
```

</details>

---

## 3. Dictionary

Click any term below, or follow links throughout the cheat sheet.

**A–C:** [Accuracy](#accuracy) · [Artifact](#artifact) · [Bagging](#bagging) · [Batch](#batch) · [Bias](#bias) · [Bias-Variance Tradeoff](#bias-variance-tradeoff) · [Boosting](#boosting) · [Bootstrap](#bootstrap) · [Bootstrapping](#bootstrapping) · [Class Imbalance](#class-imbalance) · [Confusion Matrix](#confusion-matrix) · [Cross-Validation](#cross-validation)

**D–G:** [Dropout](#dropout) · [Early Stopping](#early-stopping) · [Ensemble](#ensemble) · [Entropy](#entropy) · [Epoch](#epoch) · [F1-Score](#f1-score) · [Feature](#feature) · [Feature Engineering](#feature-engineering) · [Gini](#gini) · [Gradient](#gradient) · [Gradient Descent](#gradient-descent) · [Grid Search](#grid-search)

**H–O:** [Hyperparameter](#hyperparameter) · [Imputation](#imputation) · [Inference](#inference) · [Information Gain](#information-gain) · [Instance](#instance) · [K-Fold](#k-fold) · [Kernel Trick](#kernel-trick) · [L1 (Lasso)](#l1-lasso) · [L2 (Ridge)](#l2-ridge) · [Label](#label) · [Lazy Learner](#lazy-learner) · [Learning Rate](#learning-rate) · [Log Loss](#log-loss) · [MAE](#mae) · [Metric](#metric) · [MSE](#mse) · [OOB (Out-of-Bag)](#oob-out-of-bag) · [One-Hot Encoding](#one-hot-encoding) · [Overfitting](#overfitting)

**P–Z:** [Parameter](#parameter) · [Pipeline](#pipeline) · [Precision](#precision) · [R²](#r2) · [Random Search](#random-search) · [Recall](#recall) · [Regularization](#regularization) · [ReLU](#relu) · [RMSE](#rmse) · [ROC-AUC](#roc-auc) · [Sigmoid](#sigmoid) · [Softmax](#softmax) · [Support Vector](#support-vector) · [Underfitting](#underfitting) · [Variance](#variance)

---

<a id="accuracy"></a>
### Accuracy

Overall fraction of predictions that match the true label. Misleading when [classes are imbalanced](#class-imbalance) — a model predicting only the majority class can score high accuracy while failing on rare classes.

$$\text{Accuracy} = \frac{N_{\text{correct}}}{n}$$

<a id="artifact"></a>
### Artifact

A saved model file and its associated preprocessors (scaler, encoder, feature list) packaged for [deployment](#inference).

<a id="bagging"></a>
### Bagging

**B**ootstrap **agg**regating. Train multiple models on different [bootstrap](#bootstrap) samples and aggregate their predictions. Reduces [variance](#variance). [Random Forest](#24-random-forest) is a bagging method over decision trees.

<a id="batch"></a>
### Batch / Mini-batch

A subset of the training set used in one [gradient](#gradient) update step. Smaller batches add noise but can improve generalization; larger batches give more stable gradients.

<a id="bias"></a>
### Bias

Error from an overly simple model that cannot capture the true relationship in the data ([underfitting](#underfitting)). High bias means systematic prediction errors regardless of training set size.

<a id="bias-variance-tradeoff"></a>
### Bias-Variance Tradeoff

Increasing model complexity reduces [bias](#bias) but increases [variance](#variance). The best model balances both to minimize total generalization error.

<a id="boosting"></a>
### Boosting

Sequentially train [weak learners](#weak-learner), each correcting the errors of the previous ensemble. Reduces [bias](#bias). [Gradient Boosting](#29-gradient-boosting) fits each new tree to pseudo-residuals.

<a id="bootstrap"></a>
### Bootstrap

Sample $n$ points from the training set **with replacement**, creating a dataset of the same size where some points appear multiple times and others not at all.

<a id="bootstrapping"></a>
### Bootstrapping

Repeatedly drawing [bootstrap](#bootstrap) samples to train models or estimate uncertainty. In [Random Forest](#24-random-forest), samples not selected in a bootstrap draw become [OOB](#oob-out-of-bag) validation points.

<a id="class-imbalance"></a>
### Class Imbalance

Unequal class frequencies in the training data. May require class weighting, resampling, or metrics like [F1-score](#f1-score) and [ROC-AUC](#roc-auc) instead of [accuracy](#accuracy).

<a id="confusion-matrix"></a>
### Confusion Matrix

A table counting true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) per class. Foundation for [precision](#precision), [recall](#recall), and [F1-score](#f1-score).

<a id="cross-validation"></a>
### Cross-Validation

Resampling technique to estimate model performance on unseen data without holding out a separate test set during tuning. See also [k-fold](#k-fold).

<a id="dropout"></a>
### Dropout

[Regularization](#regularization) technique for neural networks: randomly zero out neurons during training so no single neuron becomes indispensable. Typical rate: 0.1–0.5.

<a id="early-stopping"></a>
### Early Stopping

Halt training when a validation [metric](#metric) stops improving, preventing [overfitting](#overfitting). Commonly used with [neural networks](#28-neural-network-feedforward) and [gradient boosting](#29-gradient-boosting).

<a id="ensemble"></a>
### Ensemble

Combine multiple models ([bagging](#bagging), [boosting](#boosting), or voting) to achieve better performance than any single model.

<a id="entropy"></a>
### Entropy

Classification impurity measure for [decision trees](#23-decision-tree): $H = -\sum_k p_k \log_2 p_k$, where $p_k$ is the fraction of class $k$ in a node. Pure nodes have entropy 0.

<a id="epoch"></a>
### Epoch

One complete pass through the entire training set during iterative optimization ([gradient descent](#gradient-descent)).

<a id="f1-score"></a>
### F1-Score

Harmonic mean of [precision](#precision) and [recall](#recall). Balances both when you care about positive-class performance.

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

<a id="feature"></a>
### Feature

An input variable (one column of the feature matrix $X$). Also called a predictor or independent variable.

<a id="feature-engineering"></a>
### Feature Engineering

Creating, transforming, or selecting [features](#feature) to improve model performance — e.g., scaling, polynomial terms, log transforms, or encoding categoricals.

<a id="gini"></a>
### Gini

Classification impurity for [decision trees](#23-decision-tree): $G = 1 - \sum_k p_k^2$. Ranges from 0 (pure node) to near 1 (uniformly mixed classes).

<a id="gradient"></a>
### Gradient

The vector of partial derivatives of the loss with respect to model [parameters](#parameter). Points in the direction of steepest increase; optimization moves in the opposite direction to minimize loss.

<a id="gradient-descent"></a>
### Gradient Descent

Iterative optimization: update parameters as $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$, where $\eta$ is the [learning rate](#learning-rate). Used in [linear regression](#21-linear-regression), [logistic regression](#22-logistic-regression), [neural networks](#28-neural-network-feedforward), and more.

<a id="grid-search"></a>
### Grid Search

[Hyperparameter](#hyperparameter) tuning by exhaustively trying every combination in a predefined grid. Guaranteed to search the grid but slow when the grid is large.

<a id="hyperparameter"></a>
### Hyperparameter

A configuration value set **before** training (not learned from data) — e.g., learning rate, tree depth, number of neighbors $k$. Tuned via [grid search](#grid-search) or [random search](#random-search).

<a id="imputation"></a>
### Imputation

Filling missing values in the dataset — e.g., mean/median for numeric [features](#feature), mode for categorical, or dropping rows.

<a id="inference"></a>
### Inference / Prediction

Using a trained model on new, unseen data to produce outputs. Also called prediction or scoring. Contrast with training.

<a id="information-gain"></a>
### Information Gain

Reduction in node impurity ([Gini](#gini) or [Entropy](#entropy)) after a [decision tree](#23-decision-tree) split. The split maximizing information gain is chosen at each node.

$$\text{IG} = I(S) - \frac{|S_L|}{|S|}I(S_L) - \frac{|S_R|}{|S|}I(S_R)$$

<a id="instance"></a>
### Instance / Sample

One row of data: a single input-output pair $(x_i, y_i)$.

<a id="k-fold"></a>
### K-Fold

[Cross-validation](#cross-validation) method: split data into $k$ folds, train on $k{-}1$ folds and validate on the remaining one, rotate $k$ times, and average the [metric](#metric). **Stratified** k-fold preserves class proportions in each fold.

<a id="kernel-trick"></a>
### Kernel Trick

Replace dot products $x_i^\top x_j$ with a kernel function $K(x_i, x_j)$ to implicitly map data into a higher-dimensional space without computing coordinates. Used in [SVM](#25-support-vector-machine-svm).

<a id="l1-lasso"></a>
### L1 (Lasso)

[L2 regularization](#l2-ridge) variant: penalty $\lambda \sum_j |w_j|$. Encourages sparsity — many weights become exactly zero, performing implicit feature selection.

<a id="l2-ridge"></a>
### L2 (Ridge)

[Regularization](#regularization) penalty $\lambda \|w\|^2 = \lambda \sum_j w_j^2$. Shrinks weights toward zero without eliminating them. Also called Ridge regression when applied to linear models.

<a id="label"></a>
### Label / Target

The output variable $y$ the model is trained to predict. Also called the response or dependent variable.

<a id="lazy-learner"></a>
### Lazy Learner

A model that stores training data at fit time and defers all computation to inference — e.g., [KNN](#26-k-nearest-neighbors-knn). No explicit training-phase parameter updates.

<a id="learning-rate"></a>
### Learning Rate

Step size $\eta$ in [gradient descent](#gradient-descent). Too large causes divergence; too small slows convergence. Often scheduled or decayed during training.

<a id="log-loss"></a>
### Log Loss

Cross-entropy loss for classification. Directly matches the training objective of [logistic regression](#22-logistic-regression). Penalizes confident wrong predictions heavily.

<a id="mae"></a>
### MAE

**M**ean **A**bsolute **E**rror. Average absolute difference between prediction and truth. More robust to [outliers](#outliers) than [MSE](#mse).

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

<a id="metric"></a>
### Metric

A single number quantifying how well model predictions match ground truth. Choose the [metric](#metric) that aligns with your task and business goal — optimizing the wrong metric yields models that look good on paper but fail in practice.

<a id="mse"></a>
### MSE

**M**ean **S**quared **E**rror. Average squared difference between prediction and truth. Penalizes large errors more than [MAE](#mae).

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

<a id="oob-out-of-bag"></a>
### OOB (Out-of-Bag)

Samples not included in a tree's [bootstrap](#bootstrap) draw during [Random Forest](#24-random-forest) training. Used as a free internal validation set without a separate holdout.

<a id="one-hot-encoding"></a>
### One-Hot Encoding

Convert a categorical [feature](#feature) into a binary vector with a single 1 for the active category and 0 elsewhere.

<a id="overfitting"></a>
### Overfitting

Model memorizes training data (low training error) but fails to generalize (high test error). Caused by excessive model complexity relative to data size. Combated with [regularization](#regularization), [early stopping](#early-stopping), or simpler models.

<a id="parameter"></a>
### Parameter

Values **learned** during training — weights, biases, tree splits, class probabilities. Contrast with [hyperparameters](#hyperparameter).

<a id="pipeline"></a>
### Pipeline

Chain preprocessing steps and a model into a single reproducible object, ensuring the same transforms are applied at training and [inference](#inference) time.

<a id="precision"></a>
### Precision

Of all positive predictions, how many are actually positive — "when the model says yes, how often is it right?"

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

<a id="r2"></a>
### R²

Coefficient of determination. Fraction of variance in $y$ explained by the model. $R^2 = 1$ is perfect; $R^2 = 0$ means the model is no better than predicting the mean.

$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

<a id="random-search"></a>
### Random Search

[Hyperparameter](#hyperparameter) tuning by sampling random combinations from specified distributions. Often finds good settings faster than [grid search](#grid-search) on large search spaces.

<a id="recall"></a>
### Recall

Of all actual positives, how many the model finds — "did we catch the important cases?" Also called sensitivity or true positive rate.

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

<a id="regularization"></a>
### Regularization

Penalty on model complexity to prevent [overfitting](#overfitting). Common forms: [L1 (Lasso)](#l1-lasso), [L2 (Ridge)](#l2-ridge), [dropout](#dropout), tree pruning.

<a id="relu"></a>
### ReLU

**Re**ctified **L**inear **U**nit: $g(z) = \max(0, z)$. Default hidden-layer activation in [neural networks](#28-neural-network-feedforward). Simple and avoids vanishing gradients.

<a id="rmse"></a>
### RMSE

**R**oot **M**ean **S**quared **E**rror. Square root of [MSE](#mse); has the same units as the target $y$, making it easier to interpret.

$$\text{RMSE} = \sqrt{\text{MSE}}$$

<a id="roc-auc"></a>
### ROC-AUC

Area under the Receiver Operating Characteristic curve (True Positive Rate vs. False Positive Rate). Measures how well the model ranks positives above negatives, independent of classification threshold.

<a id="sigmoid"></a>
### Sigmoid

Activation $\sigma(z) = 1 / (1 + e^{-z})$ mapping any real number to $(0, 1)$. Used in [logistic regression](#22-logistic-regression) to output probabilities.

<a id="softmax"></a>
### Softmax

Multi-class extension of [sigmoid](#sigmoid): $g(z_i) = e^{z_i} / \sum_j e^{z_j}$. Outputs a valid probability distribution over classes.

<a id="support-vector"></a>
### Support Vector

Training points that lie on or within the margin boundary in [SVM](#25-support-vector-machine-svm). Only support vectors determine the decision boundary in the kernelized dual formulation.

<a id="underfitting"></a>
### Underfitting

Model is too simple to capture patterns in the data. High error on both training and test sets. Remedy: increase model complexity or add [features](#feature).

<a id="variance"></a>
### Variance

Error from sensitivity to which training samples are included. High [variance](#variance) means the model changes significantly with different training sets ([overfitting](#overfitting)). Reduced by [regularization](#regularization) and [ensembling](#ensemble).

<a id="weak-learner"></a>
### Weak Learner

A simple model (e.g., shallow [decision tree](#23-decision-tree)) that performs only slightly better than chance. Combined in [boosting](#boosting) to form a strong ensemble.

---

## Quick Pseudocode Template (End-to-End)

```text
# 1. Prepare
X, y ← load_data()
X, y ← prepare_data(X, y)

# 2. Split
(X_train, X_test, y_train, y_test) ← split(X, y, test_size=0.25)

# 3. Tune & Train
best_params ← grid_search(X_train, y_train, Model, param_grid, cv=5)
model ← Model(**best_params)
model.fit(X_train, y_train)

# 4. Evaluate
metrics ← evaluate(model, X_test, y_test, task)
PRINT metrics

# 5. Deploy
save(model, "model.pkl")
serve_api(model)
```
