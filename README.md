# ClassicML

A machine learning library built from scratch in C++ — no Eigen, no Boost, no ML dependencies.
Implements classic algorithms with a clean API, a generic gradient descent optimizer, and a polished terminal output system.

Built as a learning project to understand what happens under the hood.

---

## Models

| Model | Type | Notes |
|---|---|---|
| `LinearRegression` | Regression | Batch / SGD / Mini-batch GD |
| `RidgeRegression` | Regression | L2 regularization via gradient |
| `LassoRegression` | Regression | L1 subgradient method |
| `LogisticRegression` | Classification | Sigmoid + binary cross-entropy |
| `DecisionTree` | Classification | CART, Gini impurity, recursive splits |
| `KNearestNeighbors` | Classification | Euclidean / Manhattan / Cosine |
| `KMeans` | Clustering | Random or K-Means++ init, multi-restart |
| `GaussianNB` | Classification | Per-class Gaussian, log-space posteriors |
| `SVM` | Classification | Hinge loss, SGD, linear kernel |

---

## Quick Start

```cpp
#include "models.cpp"
#include "pre.h"
#include "ino.h"

// Generate data
auto [X, y] = generate_blobs(200, 3, 42);
auto [X_train, X_test, y_train, y_test] = split_xy(X, y);

// Scale
StandardScaler scaler;
scaler.fit(X_train);
X_train = scaler.transform(X_train);
X_test  = scaler.transform(X_test);

// Train
LogisticRegression lr(GradientDescent<LinearParams, Matrix>::MiniBatch, 0.1, 32, 0.01);
lr.fit(X_train, y_train, 300, /*verbose=*/true);

// Evaluate
Vec preds = lr.predict(X_test);
classification_report(y_test, preds);
```

---

## Optimizer

All gradient-based models share a single templated optimizer:

```cpp
GradientDescent<ParamType, DataType>(type, learning_rate, batch_size, lambda)
```

Supports three modes:

- `Batch` — full-dataset gradient per step
- `Stochastic` — one sample per step
- `MiniBatch` — configurable batch size

The update rule with optional weight decay:
```
w = w - lr * (grad / n + λ * w)
```

---

## Data Utilities

```cpp
// Synthetic datasets
generate_blobs(n, k, seed)               // k Gaussian clusters
generate_linearly_separable(n, seed)     // 2-class linearly separable
generate_moons(n, seed, noise)           // two interleaving half-circles
generate_xor(n, seed, noise)             // XOR pattern (nonlinear)

// I/O
load_csv("path/to/data.csv")
save_csv("path/to/out.csv", X, y)

// Preprocessing
StandardScaler   // zero mean, unit variance
MinMaxScaler     // scale to [0, 1]

// Split
train_test_split(data, test_size, seed)
```

---

## Metrics

```cpp
// Classification
accuracy(y_true, y_pred)
precision(y_true, y_pred)
recall(y_true, y_pred)
f1_score(y_true, y_pred)
confusion_matrix(y_true, y_pred)
classification_report(y_true, y_pred)   // prints everything

// Regression
r2_score(y_true, y_pred)
rmse(y_true, y_pred)
mae(y_true, y_pred)
regression_report(y_true, y_pred)       // prints everything
```

---

## Building

No build system yet — compile demos directly:

```bash
g++ -std=c++17 -O2 demos/logistic_regression_demo.cpp -o lr_demo && ./lr_demo
```

Each demo is self-contained and shows a different model with synthetic data.

---

## Project Structure

```
ClassicML/
├── models.cpp        # All model implementations
├── optimizer.h       # GradientDescent<T> template
├── pre.h / pre.cpp   # StandardScaler, MinMaxScaler, train_test_split
├── metrics.hpp       # Evaluation metrics
├── utils.h / utils.cpp  # Linear algebra, distance functions, math utils
├── ino.h / ino.cpp   # Data generators and CSV I/O
├── logger.hpp        # Colored terminal logging
├── type.hpp          # Vec / Matrix type aliases
├── params.hpp        # LinearParams with operator overloads
└── demos/            # One demo per model
```

---

## What's Inside

A few things worth knowing about the internals:

- **K-Means++ initialization** — centroids seeded by weighted distance² sampling, not randomly, so the algorithm converges faster and more reliably
- **Log-space Naive Bayes** — posteriors computed in log-space to avoid float underflow when multiplying many small probabilities
- **Generic optimizer** — the gradient descent implementation is fully decoupled from model logic via a `compute_gradients` callback; adding a new model doesn't touch the optimizer
- **Custom sigmoid** — uses a Taylor series `exp` implementation rather than `<cmath>` (was fun to write, `std::exp` is obviously faster in production)

---

## Demos

```bash
# Each demo prints a loss curve, metrics, and model docs to the terminal
demos/linear_regression_demo.cpp
demos/logistic_regression_demo.cpp
demos/decision_tree_demo.cpp
demos/knn_demo.cpp
demos/kmeans_demo.cpp
demos/svm_demo.cpp
demos/ridge_regression_demo.cpp
demos/lasso_regression_demo.cpp
demos/gaussian_nb_demo.cpp
```
