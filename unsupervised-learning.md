# Unsupervised Learning Cheat Sheet

> Quick reference for writing pseudocode. Symbols, shapes, and update rules are stated explicitly so you can translate directly into code.

---

## Table of Contents

0. [Model Quick Reference](#model-quick-reference)
1. [Life Cycle of Unsupervised Models](#1-life-cycle-of-unsupervised-models)
   - [Metrics Overview](#metrics-overview)
2. [Unsupervised Models](#2-unsupervised-models)
   - [K-Means](#21-k-means)
   - [Hierarchical Clustering](#22-hierarchical-clustering)
   - [DBSCAN](#23-dbscan)
   - [Gaussian Mixture Model (GMM)](#24-gaussian-mixture-model-gmm)
   - [PCA](#25-pca)
   - [t-SNE](#26-t-sne)
   - [UMAP](#27-umap)
   - [Autoencoder](#28-autoencoder)
   - [Isolation Forest](#29-isolation-forest)
   - [One-Class SVM](#210-one-class-svm)
3. [Dictionary](#3-dictionary) — [A–Z index](#anomaly-detection)

---

## Model Quick Reference

> **Quick Reference** — Pick a model by scenario. Start with **Best choice**, fall back to **Alternatives** if constraints apply.

### By Task Type

| Task | Best choice | Alternatives |
|------|-------------|--------------|
| **Clustering** — spherical, fixed $k$ | [K-Means](#21-k-means) | [GMM](#24-gaussian-mixture-model-gmm) |
| **Clustering** — unknown $k$, hierarchy | [Hierarchical Clustering](#22-hierarchical-clustering) | [GMM](#24-gaussian-mixture-model-gmm) + BIC |
| **Clustering** — arbitrary shape, noise | [DBSCAN](#23-dbscan) | [Hierarchical Clustering](#22-hierarchical-clustering) |
| **Clustering** — soft assignments | [GMM](#24-gaussian-mixture-model-gmm) | [K-Means](#21-k-means) (hard only) |
| **Dim. reduction** — linear, fast | [PCA](#25-pca) | [Autoencoder](#28-autoencoder) |
| **Dim. reduction** — visualization (2D) | [t-SNE](#26-t-sne), [UMAP](#27-umap) | [PCA](#25-pca) (linear only) |
| **Dim. reduction** — nonlinear, reconstruct | [Autoencoder](#28-autoencoder) | [PCA](#25-pca) + kernel extensions |
| **Anomaly detection** — tabular, fast | [Isolation Forest](#29-isolation-forest) | [One-Class SVM](#210-one-class-svm) |
| **Anomaly detection** — high-dim, boundary | [One-Class SVM](#210-one-class-svm) | [Isolation Forest](#29-isolation-forest) |

### By Scenario

| Scenario | Best choice | Why |
|----------|-------------|-----|
| **Know number of clusters** $k$ | [K-Means](#21-k-means) | Simple, fast, scalable |
| **Don't know** $k$ | [Hierarchical Clustering](#22-hierarchical-clustering) | Dendrogram reveals structure |
| **Outliers / noise** in data | [DBSCAN](#23-dbscan) | Labels noise points explicitly |
| **Overlapping clusters** | [GMM](#24-gaussian-mixture-model-gmm) | Soft cluster membership |
| **Large dataset** clustering | [K-Means](#21-k-means), [Mini-batch K-Means](#21-k-means) | $O(nkd)$ per iteration, scalable |
| **Preprocess for supervised model** | [PCA](#25-pca) | Reduce collinearity and dimension |
| **Visualize high-dim data** | [UMAP](#27-umap) → [t-SNE](#26-t-sne) | Preserve local/global structure |
| **Need interpretable components** | [PCA](#25-pca) | Loadings show feature contributions |
| **Rare event / fraud detection** | [Isolation Forest](#29-isolation-forest) | Few anomalies, no labels needed |
| **Only "normal" examples** available | [One-Class SVM](#210-one-class-svm) | Learns boundary of normal region |

### Decision Flow

```text
START
├─ Goal: group similar points?
│   ├─ Know k, spherical clusters? → K-Means → GMM
│   ├─ Unknown k? → Hierarchical Clustering
│   ├─ Arbitrary shapes / noise? → DBSCAN
│   └─ Need soft probabilities? → GMM
│
├─ Goal: reduce dimensions?
│   ├─ Linear, fast, interpretable? → PCA
│   ├─ 2D visualization? → UMAP → t-SNE
│   └─ Nonlinear reconstruction? → Autoencoder
│
└─ Goal: find anomalies?
    ├─ Tabular, fast? → Isolation Forest
    └─ Tight normal boundary? → One-Class SVM
```

---

## 1. Life Cycle of Unsupervised Models

### Life Cycle

#### 1. Data Collection and Preparation

| Step | Pseudocode skeleton |
|------|---------------------|
| **Data Cleaning** | `remove_duplicates(X)` → `handle_missing(X)` → `clip_outliers(X)` → `encode_categorical(X)` |
| **Feature Engineering** | `scale(X)` → `reduce_noise(X)` → `select_features(X, method)` |

```text
FUNCTION prepare_data(raw_X):
    X ← clean(raw_X)
    X ← engineer_features(X)
    RETURN X
```

**Common cleaning strategies**

| Issue | Strategy |
|-------|----------|
| Missing values | <a href="#imputation">imputation</a>: mean/median (numeric), mode (categorical), drop row |
| Outliers | IQR clip, z-score; or keep for <a href="#anomaly-detection">anomaly detection</a> |
| Duplicates | drop or aggregate |
| Categorical | <a href="#one-hot-encoding">one-hot encode</a>, frequency encode |

**Common feature transforms**

| Transform | When to use |
|-----------|-------------|
| StandardScaler | distance-based methods (K-Means, DBSCAN, SVM) |
| MinMaxScaler | bounded input for neural methods |
| Log / Box-Cox | right-skewed distributions |

---

#### 2. Feature Scaling & Validation Strategy

Unlike supervised learning, there are no labels to split. Common strategies:

| Strategy | Pseudocode idea | Use when |
|----------|-----------------|----------|
| **Full-data fit** | `model.fit(X)` on all data | Exploratory analysis, visualization |
| **Holdout stability** | Fit on subset; compare cluster labels / scores | Check robustness |
| **Internal metrics** | <a href="#silhouette-score">Silhouette</a>, <a href="#davies-bouldin-index">Davies-Bouldin</a>, <a href="#inertia">Inertia</a> | Compare $k$ or hyperparameters |
| **External metrics** | ARI, NMI (if ground-truth labels exist) | Benchmarking only |

```text
FUNCTION choose_k(X, k_range):
    scores ← []
    FOR k IN k_range:
        model ← KMeans(n_clusters=k)
        labels ← model.fit_predict(X)
        scores.append(silhouette_score(X, labels))
    RETURN argmax(scores)
```

---

#### 3. Model Training

```text
FUNCTION train(X, algorithm, hyperparams):
    model ← algorithm(**hyperparams)
    model.fit(X)
    RETURN model
```

**Hyperparameter Tuning**

| Method | Pseudocode idea | Use when |
|--------|-----------------|----------|
| **Grid Search** | Try all combinations; score via internal metric | Small search space |
| **Elbow method** | Plot <a href="#inertia">inertia</a> vs. $k$; pick "elbow" | K-Means, choosing $k$ |
| **Silhouette analysis** | Maximize silhouette over $k$ | Clustering model selection |
| **BIC / AIC** | Minimize information criterion | <a href="#24-gaussian-mixture-model-gmm">GMM</a> component count |

---

#### 4. Model Evaluation

```text
FUNCTION evaluate(model, X, task):
    IF task == "clustering":
        RETURN clustering_metrics(X, model.labels_)
    ELIF task == "dim_reduction":
        RETURN explained_variance, reconstruction_error
    ELIF task == "anomaly":
        RETURN anomaly_scores, threshold_metrics
```

##### Metrics Overview

A <a href="#metric">metric</a> quantifies how well an unsupervised model captures structure in $X$. Without labels, use **internal** metrics; with ground-truth labels (evaluation only), use **external** metrics.

**Clustering metrics** (no labels required):

**<a href="#silhouette-score">Silhouette Score</a>**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

Range $[-1, 1]$; higher = better separation. $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-cluster distance.

**<a href="#davies-bouldin-index">Davies-Bouldin Index</a>**

$$DB = \frac{1}{k}\sum_{i=1}^{k} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$$

Lower = better. $\sigma_i$ = cluster spread, $d(c_i,c_j)$ = centroid distance.

**<a href="#calinski-harabasz-index">Calinski-Harabasz Index</a>**

$$CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}$$

Higher = denser, well-separated clusters.

**<a href="#inertia">Inertia</a> (WCSS)**

$$J = \sum_{i=1}^{n} \|x_i - \mu_{c_i}\|^2$$

Lower = tighter clusters; use elbow method to pick $k$ (not comparable across $k$ directly).

**Dimensionality reduction metrics:**

**Explained variance ratio** (PCA) — fraction of total variance captured per component; cumulative sum → choose number of components.

**Reconstruction error** — $\|X - \hat{X}\|^2 / n$; lower = better fidelity.

**Anomaly detection metrics** (labels needed for evaluation):

Precision, Recall, F1 on flagged anomalies; or ROC-AUC on <a href="#anomaly-score">anomaly scores</a>.

---

#### 5. Interpretation & Deployment

```text
FUNCTION deploy(model, artifacts):
    save(model, "model.pkl")
    save(scaler, "scaler.pkl")
    save(cluster_centers_or_components, "params.json")

# Serving
FUNCTION predict_endpoint(request):
    X ← preprocess(request.features, scaler)
  RETURN {"cluster": model.predict(X), "score": model.score(X)}
```

| Output | Use case |
|--------|----------|
| Cluster labels | Customer segmentation, document grouping |
| Reduced coordinates | Visualization, downstream supervised model input |
| Anomaly scores | Fraud detection, system monitoring |

---

## 2. Unsupervised Models

### 2.1 K-Means

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Spherical clusters; known $k$; large $n$; fast baseline |
| **Cons** | Must specify $k$; sensitive to initialization and scale; assumes equal-variance clusters |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | data matrix | (n, d) |
| k | number of clusters | scalar |
| $\mu_j$ | centroid of cluster $j$ | (d,) |
| $c_i$ | cluster assignment of point $i$ | scalar |
| n | number of samples | scalar |
| d | number of features | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Assumption** — clusters are spherical with similar variance

2. **Objective (<a href="#inertia">Inertia</a> / WCSS)**  
   $J = \sum_{i=1}^{n} \|x_i - \mu_{c_i}\|^2$

3. **Assignment** — assign each point to nearest centroid  
   $c_i = \arg\min_j \|x_i - \mu_j\|^2$

4. **Centroid update**  
   $\mu_j = \dfrac{1}{|C_j|}\sum_{i \in C_j} x_i$

5. **Goal**  
   $\{\mu_j, c_i\}^* = \arg\min J$ (NP-hard; Lloyd's algorithm finds local minimum)

</details>

<details>
<summary><strong>d. Update Rules</strong> (Lloyd's algorithm)</summary>

1. **Initialize** $k$ centroids (random points or <a href="#k-means-plus-plus">k-means++</a>)

2. **Assign** each $x_i$ to nearest $\mu_j$

3. **Update** $\mu_j$ ← mean of assigned points

4. Repeat steps 2–3 until assignments stabilize or max iterations

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| n_clusters ($k$) | 2 – 20 | <a href="#elbow-method">elbow</a> or <a href="#silhouette-score">silhouette</a> |
| init | k-means++, random | k-means++ reduces bad local minima |
| n_init | 10 – 50 | run multiple times; keep best inertia |
| max_iter | 100 – 300 | per run |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#silhouette-score">Silhouette</a>, <a href="#davies-bouldin-index">Davies-Bouldin</a>, <a href="#calinski-harabasz-index">Calinski-Harabasz</a>, <a href="#inertia">Inertia</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare — scale features!
X ← StandardScaler().fit_transform(X)

# Train
model ← KMeans(n_clusters=5, init="k-means++", n_init=20)
labels ← model.fit_predict(X)
centroids ← model.cluster_centers_
inertia ← model.inertia_

# Evaluate
sil ← silhouette_score(X, labels)
```

</details>

---

### 2.2 Hierarchical Clustering

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Unknown $k$; need dendrogram; nested cluster structure |
| **Cons** | Slow $O(n^2)$–$O(n^3)$; sensitive to noise; irreversible merges/splits |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| X | (n, d) data matrix |
| $D_{ij}$ | distance between points $i, j$ |
| linkage | rule to merge cluster distances |
| dendrogram | tree of merge heights |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Agglomerative** — start with each point as its own cluster; iteratively merge closest pair

2. **Distance between clusters** (<a href="#linkage">linkage</a>):
   - Single: $d(C_i, C_j) = \min_{x \in C_i, y \in C_j} \|x - y\|$
   - Complete: $d(C_i, C_j) = \max_{x \in C_i, y \in C_j} \|x - y\|$
   - Average: mean pairwise distance
   - Ward: minimizes increase in within-cluster variance

3. **Stop** when desired number of clusters $k$ reached or distance threshold exceeded

4. **Output** — dendrogram; cut at height $h$ to get $k$ clusters

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Compute pairwise distance matrix $D$

2. **While** clusters $> k$:
   - Find pair $(C_i, C_j)$ with minimum linkage distance
   - Merge $C_i, C_j$ into $C_{new}$
   - Record merge height

3. **Cut** dendrogram at chosen height → final labels

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Options |
|----------------|---------|
| n_clusters ($k$) | cut dendrogram at height |
| linkage | ward, complete, average, single |
| metric | euclidean, cosine, cityblock |
| distance_threshold | alternative to fixed $k$ |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#silhouette-score">Silhouette</a>, <a href="#davies-bouldin-index">Davies-Bouldin</a>, cophenetic correlation (dendrogram fidelity). See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare
X ← StandardScaler().fit_transform(X)

# Train
model ← AgglomerativeClustering(n_clusters=4, linkage="ward")
labels ← model.fit_predict(X)

# Visualize structure
linkage_matrix ← linkage(X, method="ward")
dendrogram(linkage_matrix)
```

</details>

---

### 2.3 DBSCAN

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Arbitrary cluster shapes; noise/outliers; unknown $k$ |
| **Cons** | Sensitive to <a href="#eps">eps</a> and <a href="#min-samples">min_samples</a>; struggles with varying density |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $\varepsilon$ (eps) | neighborhood radius |
| min_samples | min points to form dense region |
| core point | $\geq$ min_samples neighbors within eps |
| border point | in eps of core but not core |
| noise | label $-1$ |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **$\varepsilon$-neighborhood**  
   $N_\varepsilon(x) = \{x' : \|x - x'\| \leq \varepsilon\}$

2. **Core point** — $|N_\varepsilon(x)| \geq$ min_samples

3. **Density-reachable** — chain of core points within $\varepsilon$

4. **Cluster** — maximal set of density-connected points

5. **Noise** — points not in any cluster (label $-1$)

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Mark all points unvisited

2. **For each** unvisited point $x$:
   - If $|N_\varepsilon(x)| <$ min_samples → mark noise (may change later)
   - Else start new cluster; **expand** via breadth-first search through density-reachable core points

3. Assign cluster IDs; border points get cluster of adjacent core

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| eps ($\varepsilon$) | tune via k-distance plot | too small → all noise |
| min_samples | 3 – 2×d | higher → fewer, denser clusters |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#silhouette-score">Silhouette</a> (exclude noise points), cluster count, noise fraction. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare — scale features!
X ← StandardScaler().fit_transform(X)

# Train
model ← DBSCAN(eps=0.5, min_samples=5)
labels ← model.fit_predict(X)        # -1 = noise

# Evaluate
n_clusters ← len(set(labels)) - (1 if -1 in labels else 0)
n_noise ← sum(labels == -1)
```

</details>

---

### 2.4 Gaussian Mixture Model (GMM)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Soft cluster assignments; overlapping elliptical clusters; need probabilities |
| **Cons** | Assumes Gaussian components; sensitive to initialization; slower than K-Means |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | data | (n, d) |
| K | number of components | scalar |
| $\pi_k$ | mixing weight of component $k$ | scalar |
| $\mu_k$ | mean of component $k$ | (d,) |
| $\Sigma_k$ | covariance of component $k$ | (d, d) |
| $\gamma_{ik}$ | responsibility of component $k$ for point $i$ | (n, K) |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Mixture model**  
   $p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k)$

2. **Responsibility (<a href="#em-algorithm">E-step</a>)**  
   $\gamma_{ik} = \dfrac{\pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j\, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$

3. **Log-likelihood**  
   $\mathcal{L} = \sum_i \log \sum_k \pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$

4. **Assignment** — $\hat{c}_i = \arg\max_k \gamma_{ik}$ (hard) or use $\gamma_{ik}$ (soft)

</details>

<details>
<summary><strong>d. Update Rules</strong> · <a href="#em-algorithm">EM algorithm</a></summary>

1. **Initialize** $\pi_k, \mu_k, \Sigma_k$

2. **E-step** — compute responsibilities $\gamma_{ik}$

3. **M-step** — update parameters:  
   $N_k = \sum_i \gamma_{ik}$  
   $\pi_k = N_k / n$  
   $\mu_k = \dfrac{1}{N_k}\sum_i \gamma_{ik} x_i$  
   $\Sigma_k = \dfrac{1}{N_k}\sum_i \gamma_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top$

4. Repeat until $\mathcal{L}$ converges

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| n_components ($K$) | 2 – 20 | minimize BIC/AIC |
| covariance_type | full, tied, diag, spherical | full = most flexible |
| n_init | 5 – 10 | multiple random starts |
| max_iter | 100 – 200 | EM iterations |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

BIC, AIC, log-likelihood, <a href="#silhouette-score">Silhouette</a> on hard assignments. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare
X ← StandardScaler().fit_transform(X)

# Train
model ← GaussianMixture(n_components=3, covariance_type="full", n_init=10)
labels ← model.fit_predict(X)
proba ← model.predict_proba(X)       # soft assignments
bic ← model.bic(X)

# Evaluate
sil ← silhouette_score(X, labels)
```

</details>

---

### 2.5 PCA

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Linear dim. reduction; decorrelate features; visualization; preprocessing |
| **Cons** | Only linear projections; components less interpretable with many features |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | centered data matrix | (n, d) |
| W | principal component directions | (d, k) |
| Z | projected data | (n, k) |
| k | number of components kept | scalar |
| $\lambda_j$ | eigenvalue of component $j$ | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Center data** — $\tilde{X} = X - \bar{X}$

2. **Covariance matrix** — $C = \dfrac{1}{n}\tilde{X}^\top \tilde{X}$

3. **Eigendecomposition** — $C v_j = \lambda_j v_j$; $v_j$ = principal directions

4. **Projection** — $Z = \tilde{X} W_k$ where $W_k = [v_1, \ldots, v_k]$

5. **Explained variance ratio** — $\lambda_j / \sum_i \lambda_i$

6. **Reconstruction** — $\hat{X} = Z W_k^\top + \bar{X}$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Center $X$

2. Compute SVD: $\tilde{X} = U \Sigma V^\top$ (equivalent to eigendecomposition of $C$)

3. Keep top $k$ columns of $V$ as $W_k$

4. Project: $Z \leftarrow \tilde{X} W_k$

5. No iterative gradient update — closed-form via SVD

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| n_components ($k$) | 2 – d | cumulative explained variance ≥ 0.95 |
| svd_solver | auto, full, randomized | randomized for large $n$ |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#explained-variance-ratio">Explained variance ratio</a>, <a href="#reconstruction-error">reconstruction error</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare — center/scale
X ← StandardScaler().fit_transform(X)

# Train
model ← PCA(n_components=0.95)       # keep 95% variance
Z ← model.fit_transform(X)             # reduced representation
loadings ← model.components_
var_ratio ← model.explained_variance_ratio_

# Reconstruct
X_hat ← model.inverse_transform(Z)
recon_error ← mean((X - X_hat)^2)
```

</details>

---

### 2.6 t-SNE

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | 2D/3D visualization of high-dim data; explore local structure |
| **Cons** | Non-parametric; slow $O(n^2)$; distances between clusters not meaningful; no transform for new points |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| X | high-dim input | (n, d) |
| Y | low-dim embedding | (n, 2) or (n, 3) |
| $p_{j|i}$ | high-dim similarity | scalar |
| $q_{ij}$ | low-dim similarity | scalar |
| perplexity | effective number of neighbors | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **High-dim affinities** (Gaussian, perplexity-controlled)  
   $p_{j|i} \propto \exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)$

2. **Symmetrize** — $p_{ij} = (p_{j|i} + p_{i|j}) / 2n$

3. **Low-dim affinities** (Student-t, heavy tails)  
   $q_{ij} = \dfrac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$

4. **Objective (KL divergence)**  
   $C = \sum_{i \neq j} p_{ij} \log \dfrac{p_{ij}}{q_{ij}}$

</details>

<details>
<summary><strong>d. Update Rules</strong> · <a href="#gradient-descent">gradient descent</a></summary>

1. Initialize low-dim points $y_i$ randomly

2. **Compute** gradients $\dfrac{\partial C}{\partial y_i}$

3. **Update** $y_i \leftarrow y_i - \eta \dfrac{\partial C}{\partial y_i} + \alpha(\Delta y_i^{\text{prev}})$ (momentum)

4. Repeat until convergence; early exaggeration in first iterations

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| perplexity | 5 – 50 | ~ number of effective neighbors |
| learning_rate | 10 – 1000 | often auto |
| n_iter | 1000 – 5000 | more for large $n$ |
| early_exaggeration | 12 | separates clusters early |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Visual inspection; KL divergence value; trustworthiness / continuity scores. Not for downstream prediction. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare — PCA pre-reduction recommended for large d
X ← StandardScaler().fit_transform(X)
X ← PCA(n_components=50).fit_transform(X)

# Train (fit_transform only — no predict on new data)
model ← TSNE(n_components=2, perplexity=30, n_iter=1000)
Y ← model.fit_transform(X)           # (n, 2) embedding

# Visualize
scatter(Y[:, 0], Y[:, 1], c=optional_labels)
```

</details>

---

### 2.7 UMAP

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | 2D visualization; preserve global + local structure; faster than t-SNE |
| **Cons** | Hyperparameter-sensitive; less established theory than PCA/t-SNE |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| X | input data | (n, d) |
| Y | embedding | (n, k) |
| k (n_neighbors) | local neighborhood size | scalar |
| min_dist | minimum spacing in embedding | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Construct** weighted k-neighbor graph in high-dim space

2. **Define** fuzzy simplicial set (approximates manifold structure)

3. **Optimize** low-dim layout to preserve topological structure

4. **Cross-entropy loss** between high-dim and low-dim fuzzy sets

5. **Balance** local (n_neighbors) vs. global (min_dist, spread) structure

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Build k-neighbor graph; compute edge weights

2. Initialize embedding (spectral or random)

3. **Stochastic gradient descent** on cross-entropy layout loss

4. Repeat until convergence

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| n_neighbors | 5 – 50 | low = local, high = global |
| min_dist | 0.0 – 0.5 | 0 = tight clusters |
| n_components | 2 – 3 | visualization |
| metric | euclidean, cosine | feature space distance |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Visual inspection; trustworthiness; can `.transform()` new points (unlike t-SNE). See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare
X ← StandardScaler().fit_transform(X)

# Train
model ← UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
Y ← model.fit_transform(X)

# Transform new points (supported)
Y_new ← model.transform(X_new)
```

</details>

---

### 2.8 Autoencoder

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Nonlinear dim. reduction; denoising; anomaly detection via reconstruction error |
| **Cons** | Needs more data; tuning-heavy; less interpretable than PCA |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | input | (n, d) |
| $h$ | bottleneck (latent) | (n, k) |
| encoder | $f_\theta$ | d → k |
| decoder | $g_\phi$ | k → d |
| $\hat{X}$ | reconstruction | (n, d) |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Encode** — $h = f_\theta(X)$

2. **Decode** — $\hat{X} = g_\phi(h)$

3. **Loss (<a href="#reconstruction-error">reconstruction</a>)**  
   $\mathcal{L} = \dfrac{1}{n}\sum_i \|x_i - \hat{x}_i\|^2$

4. **Objective**  
   $\theta^*, \phi^* = \arg\min_{\theta,\phi} \mathcal{L}$

5. **Anomaly score** — high $\|x_i - \hat{x}_i\|^2$ = unusual point

</details>

<details>
<summary><strong>d. Update Rules</strong> · backpropagation</summary>

1. **Forward** — $h = f_\theta(X)$; $\hat{X} = g_\phi(h)$; compute $\mathcal{L}$

2. **Backward** — gradients via chain rule through encoder + decoder

3. **Update** $\theta, \phi$ with Adam/SGD

4. Repeat for each epoch

5. **Latent representation** — use $h$ as reduced features

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| latent_dim ($k$) | 2 – 64 | bottleneck size |
| hidden layers | 1 – 3 | encoder/decoder symmetry |
| learning_rate | 1e-4 – 1e-2 | |
| epochs | 50 – 200 | early stopping on val loss |
| denoising | Gaussian noise | robust representations |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#reconstruction-error">Reconstruction error</a>; anomaly scores from per-sample error. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare
X ← MinMaxScaler().fit_transform(X)

# Build & train (conceptual)
encoder ← MLP(input_dim=d, hidden=[128, 64], output=latent_dim)
decoder ← MLP(input_dim=latent_dim, hidden=[64, 128], output=d)
FOR epoch IN 1..max_epochs:
    h ← encoder(X)
    X_hat ← decoder(h)
    loss ← MSE(X, X_hat)
    backprop_and_update(encoder, decoder, loss)

# Use
Z ← encoder(X)                       # reduced representation
anomaly_score ← mean((X - decoder(encoder(X)))^2, axis=1)
```

</details>

---

### 2.9 Isolation Forest

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Anomaly detection on tabular data; few anomalies; fast, scalable |
| **Cons** | Less effective in very high dimensions; assumes anomalies are "few and different" |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| X | data | (n, d) |
| T | number of isolation trees | scalar |
| $\psi$ | subsample size per tree | scalar |
| s(x) | anomaly score of point $x$ | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Idea** — anomalies are few and far from others → isolated quickly in random trees

2. **Path length** $h(x)$ — number of splits to isolate point $x$

3. **Average path** $E[h(x)]$ across trees

4. **Anomaly score**  
   $s(x) = 2^{-E[h(x)] / c(\psi)}$  
   where $c(\psi)$ normalizes by average path in random tree

5. **Decision** — $s(x) \approx 1$ → anomaly; $s(x) \ll 0.5$ → normal

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. **For** $t = 1, \ldots, T$:
   - Sample subsample of size $\psi$
   - Build isolation tree: recursively split on random feature at random threshold until point isolated or max depth

2. **Score** each point by average path length across trees

3. **Threshold** — flag points with $s(x) > \tau$ (or top contamination %)

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| n_estimators ($T$) | 100 – 300 | more = stable scores |
| contamination | 0.01 – 0.1 | expected anomaly fraction |
| max_samples ($\psi$) | 256 – 512 | subsample per tree |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Precision/Recall on labeled anomalies; ROC-AUC on <a href="#anomaly-score">anomaly scores</a>; <a href="#contamination">contamination</a> rate. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare
X ← StandardScaler().fit_transform(X)

# Train
model ← IsolationForest(n_estimators=200, contamination=0.05)
model.fit(X)

# Predict
labels ← model.predict(X)            # 1 = normal, -1 = anomaly
scores ← model.score_samples(X)      # lower = more anomalous

# Flag top anomalies
threshold ← percentile(scores, 5)
anomalies ← X[scores < threshold]
```

</details>

---

### 2.10 One-Class SVM

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Learn boundary of "normal" data; only normal examples at train time |
| **Cons** | Slow on large $n$; sensitive to scaling; kernel tuning needed |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| X | training data (normal only) | (n, d) |
| $\nu$ | upper bound on outlier fraction | scalar |
| R | hypersphere radius | scalar |
| $\rho$ | center offset | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Goal** — find smallest hypersphere enclosing most of the data

2. **Primal (kernelized)** — separate normal data from origin in feature space

3. **$\nu$-SVM formulation** — $\nu$ controls fraction of outliers and support vectors

4. **Decision** — $f(x) = \text{sign}(\sum_i \alpha_i K(x_i, x) - \rho)$

5. **Anomaly** — $f(x) < 0$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Solve quadratic program for $\alpha_i$ (dual formulation)

2. Support vectors define boundary

3. **Predict** — negative decision → anomaly

4. Kernel options: RBF (default), linear, poly

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| nu ($\nu$) | 0.01 – 0.2 | expected anomaly rate |
| kernel | rbf, linear | RBF for nonlinear boundary |
| gamma | 1e-3 – 1e0 | RBF width |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Precision/Recall on labeled anomalies; ROC-AUC; false positive rate on normal validation. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Prepare — scale features!
X_train ← StandardScaler().fit_transform(X_normal)   # normal data only
X_test ← scaler.transform(X_test)

# Train
model ← OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
model.fit(X_train)

# Predict
labels ← model.predict(X_test)       # 1 = normal, -1 = anomaly
scores ← model.decision_function(X_test)
```

</details>

---

## 3. Dictionary

Click any term below, or follow links throughout the cheat sheet.

**A–C:** [Anomaly Detection](#anomaly-detection) · [Anomaly Score](#anomaly-score) · [ARI](#ari) · [BIC](#bic) · [Calinski-Harabasz Index](#calinski-harabasz-index) · [Centroid](#centroid) · [Cluster](#cluster) · [Contamination](#contamination) · [Cophenetic Correlation](#cophenetic-correlation)

**D–G:** [Davies-Bouldin Index](#davies-bouldin-index) · [Dendrogram](#dendrogram) · [Dimensionality Reduction](#dimensionality-reduction) · [Elbow Method](#elbow-method) · [EM Algorithm](#em-algorithm) · [eps](#eps) · [Explained Variance Ratio](#explained-variance-ratio)

**H–P:** [Hierarchical Clustering](#hierarchical-clustering) · [Hyperparameter](#hyperparameter) · [Imputation](#imputation) · [Inertia](#inertia) · [Isolation Forest](#isolation-forest) · [K-Means++](#k-means-plus-plus) · [Latent Space](#latent-space) · [Linkage](#linkage) · [Metric](#metric) · [min_samples](#min-samples)

**R–Z:** [Reconstruction Error](#reconstruction-error) · [Responsibility](#responsibility) · [Silhouette Score](#silhouette-score) · [Soft Clustering](#soft-clustering) · [WCSS](#wcss)

---

<a id="anomaly-detection"></a>
### Anomaly Detection

Finding rare or unusual observations that deviate from the majority pattern. Also called outlier detection. No labels required at training time.

<a id="anomaly-score"></a>
### Anomaly Score

A continuous value per point indicating how "unusual" it is. Higher (Isolation Forest) or lower (One-Class SVM decision function) scores flag anomalies depending on the algorithm.

<a id="ari"></a>
### ARI

**A**djusted **R**and **I**ndex. External clustering metric when ground-truth labels exist. Corrects for chance; 1 = perfect match, 0 = random.

<a id="bic"></a>
### BIC

**B**ayesian **I**nformation **C**riterion. Penalized log-likelihood for model selection. Lower BIC → better trade-off of fit vs. complexity. Used to choose $K$ in <a href="#24-gaussian-mixture-model-gmm">GMM</a>.

<a id="calinski-harabasz-index"></a>
### Calinski-Harabasz Index

Ratio of between-cluster to within-cluster dispersion. Higher = better-defined clusters. Also called Variance Ratio Criterion.

<a id="centroid"></a>
### Centroid

The mean of all points in a <a href="#cluster">cluster</a>. K-Means iteratively updates centroids to minimize <a href="#inertia">inertia</a>.

<a id="cluster"></a>
### Cluster

A group of similar data points. Unsupervised learning discovers clusters without label guidance.

<a id="contamination"></a>
### Contamination

Expected proportion of anomalies in the data. Used in <a href="#29-isolation-forest">Isolation Forest</a> and <a href="#anomaly-detection">anomaly detection</a> to set thresholds.

<a id="cophenetic-correlation"></a>
### Cophenetic Correlation

Measures how faithfully a <a href="#dendrogram">dendrogram</a> preserves original pairwise distances. Closer to 1 = better representation.

<a id="davies-bouldin-index"></a>
### Davies-Bouldin Index

Average similarity between each cluster and its most similar cluster. Lower = better separation. Uses cluster centroids and spreads.

<a id="dendrogram"></a>
### Dendrogram

Tree diagram showing merge order and heights in <a href="#hierarchical-clustering">hierarchical clustering</a>. Cut at a height to obtain $k$ clusters.

<a id="dimensionality-reduction"></a>
### Dimensionality Reduction

Mapping high-dimensional data to fewer dimensions while preserving structure. Methods: <a href="#25-pca">PCA</a> (linear), <a href="#26-t-sne">t-SNE</a>/<a href="#27-umap">UMAP</a> (visualization), <a href="#28-autoencoder">Autoencoder</a> (nonlinear).

<a id="elbow-method"></a>
### Elbow Method

Plot <a href="#inertia">inertia</a> vs. $k$; choose $k$ at the "elbow" where marginal gain diminishes.

<a id="em-algorithm"></a>
### EM Algorithm

**E**xpectation-**M**aximization. Iterative method for models with latent variables. E-step computes expected assignments; M-step updates parameters. Used in <a href="#24-gaussian-mixture-model-gmm">GMM</a>.

<a id="eps"></a>
### eps

Neighborhood radius $\varepsilon$ in <a href="#23-dbscan">DBSCAN</a>. Defines local density reachability.

<a id="explained-variance-ratio"></a>
### Explained Variance Ratio

Fraction of total data variance captured by each <a href="#25-pca">PCA</a> component. Cumulative sum guides choice of $k$.

<a id="gmm"></a>
### GMM

<a href="#24-gaussian-mixture-model-gmm">Gaussian Mixture Model</a>. Soft clustering via mixture of Gaussians.

<a id="hierarchical-clustering"></a>
### Hierarchical Clustering

Builds a hierarchy of clusters via agglomerative (bottom-up) or divisive (top-down) merging. Produces a <a href="#dendrogram">dendrogram</a>.

<a id="hyperparameter"></a>
### Hyperparameter

Configuration set before training — e.g., $k$, eps, perplexity, latent dimension. Tuned via grid search or internal metrics.

<a id="imputation"></a>
### Imputation

Filling missing values — mean/median for numeric features, mode for categorical.

<a id="inertia"></a>
### Inertia

Within-cluster sum of squares (WCSS). K-Means objective; sum of squared distances to centroids. See <a href="#wcss">WCSS</a>.

<a id="isolation-forest"></a>
### Isolation Forest

<a href="#29-isolation-forest">Isolation Forest</a>. Anomaly detection by measuring how quickly random splits isolate each point.

<a id="k-means-plus-plus"></a>
### K-Means++

Smart initialization for K-Means: spread initial centroids far apart. Reduces risk of poor local minima.

<a id="latent-space"></a>
### Latent Space

Low-dimensional representation learned by models like <a href="#28-autoencoder">Autoencoders</a> or <a href="#25-pca">PCA</a>. Encodes compressed structure of the data.

<a id="linkage"></a>
### Linkage

Rule for measuring distance between clusters in <a href="#hierarchical-clustering">hierarchical clustering</a>. Options: single, complete, average, ward.

<a id="metric"></a>
### Metric

A number quantifying unsupervised model quality — e.g., <a href="#silhouette-score">Silhouette</a>, <a href="#reconstruction-error">reconstruction error</a>, BIC.

<a id="min-samples"></a>
### min_samples

Minimum points within <a href="#eps">eps</a> radius to form a core point in <a href="#23-dbscan">DBSCAN</a>.

<a id="pca"></a>
### PCA

<a href="#25-pca">Principal Component Analysis</a>. Linear dimensionality reduction via eigendecomposition of the covariance matrix.

<a id="reconstruction-error"></a>
### Reconstruction Error

$\|X - \hat{X}\|^2 / n$. Measures fidelity of <a href="#25-pca">PCA</a> or <a href="#28-autoencoder">Autoencoder</a> reconstructions. High error → possible anomaly.

<a id="responsibility"></a>
### Responsibility

In <a href="#24-gaussian-mixture-model-gmm">GMM</a>, $\gamma_{ik}$ = probability that component $k$ generated point $i$. Soft cluster membership.

<a id="silhouette-score"></a>
### Silhouette Score

Per-point measure of cluster cohesion vs. separation. Mean silhouette across points; range $[-1, 1]$, higher is better.

<a id="soft-clustering"></a>
### Soft Clustering

Assigning probabilistic membership to multiple clusters. <a href="#24-gaussian-mixture-model-gmm">GMM</a> outputs soft assignments; K-Means is hard clustering.

<a id="t-sne"></a>
### t-SNE

<a href="#26-t-sne">t-Distributed Stochastic Neighbor Embedding</a>. Nonlinear visualization method preserving local structure.

<a id="umap"></a>
### UMAP

<a href="#27-umap">Uniform Manifold Approximation and Projection</a>. Visualization preserving local and global structure; faster than t-SNE.

<a id="autoencoder"></a>
### Autoencoder

<a href="#28-autoencoder">Autoencoder</a>. Neural network that compresses and reconstructs data for nonlinear dimensionality reduction or anomaly detection.

<a id="wcss"></a>
### WCSS

**W**ithin-**C**luster **S**um of **S**quares. Same as <a href="#inertia">inertia</a>.

<a id="one-hot-encoding"></a>
### One-Hot Encoding

Convert categorical feature to binary vector with one active category.

---

## Quick Pseudocode Template (End-to-End)

```text
# 1. Prepare
X ← load_data()
X ← prepare_data(X)
X ← StandardScaler().fit_transform(X)

# 2. Choose model & hyperparameters
best_k ← choose_k(X, k_range=2..10)     # clustering
model ← Algorithm(**best_params)

# 3. Fit
labels ← model.fit_predict(X)           # clustering
Z ← model.fit_transform(X)              # dim. reduction
scores ← model.fit(X).score_samples(X)  # anomaly

# 4. Evaluate
metrics ← evaluate(model, X, task)
PRINT metrics

# 5. Deploy / use
save(model, "model.pkl")
serve_clusters_or_embeddings(model)
```
