# ClassicML

Student project: classic ML algorithms implemented from scratch in C++.
No Eigen, no Boost, no ML dependencies — just the STL.

## Models

| Model | Type | Notes |
|---|---|---|
| `LinearRegression` | Regression | Batch / SGD / Mini-batch GD |
| `RidgeRegression` | Regression | L2 regularization via gradient |
| `LassoRegression` | Regression | L1 subgradient method |
| `LogisticRegression` | Classification | Sigmoid + binary cross-entropy |
| `DecisionTree` | Classification | CART, Gini impurity, recursive splits |
| `KNearestNeighbors` | Classification | Euclidean / Manhattan / Cosine |
| `KMeans` | Clustering | K-Means++ init, multi-restart |
| `GaussianNB` | Classification | Per-class Gaussian, log-space posteriors |
| `SVM` | Classification | Hinge loss, SGD, linear kernel |

## Quick Start

```cpp
#include "cml.hpp"

auto [X, y] = generate_blobs(200, 3, 42);

StandardScaler scaler;
scaler.fit(X);
X = scaler.transform(X);

size_t split = X.size() * 0.8;
Matrix X_train(X.begin(), X.begin() + split);
Matrix X_test(X.begin() + split, X.end());
Vec y_train(y.begin(), y.begin() + split);
Vec y_test(y.begin() + split, y.end());

LogisticRegression lr(GradientDescent<LinearParams, Matrix>::MiniBatch, 0.1, 32, 0.01);
lr.fit(X_train, y_train, 300, true);

Vec preds = lr.predict(X_test);
classification_report(y_test, preds);
```

## Building

Single header (`cml.hpp`), two compilation units:

```bash
g++ -std=c++17 -O2 my_prog.cpp utils.cpp ino.cpp -o my_prog && ./my_prog
```

`utils.cpp` and `ino.cpp` contain non-inline definitions (`matinv`,
`custom_exp`, CSV I/O, data generators) and must be compiled alongside
your program.

## Project Structure

```
ClassicML/
├── cml.hpp              # Single user-facing include
├── models/              # Per-model headers
├── optimizer.h          # GradientDescent<T> template
├── pre.h                # Scalers + train_test_split
├── metrics.hpp          # Accuracy, F1, R², etc.
├── utils.h / utils.cpp  # Linear algebra, distances, math
├── ino.h / ino.cpp      # Data generators, CSV I/O
├── logger.hpp           # Colored terminal output
├── type.hpp             # Vec / Matrix type aliases
└── params.hpp           # LinearParams with operator overloads
```

## Utilities

- **Synthetic data:** `generate_blobs`, `generate_linearly_separable`,
  `generate_moons`, `generate_xor`
- **I/O:** `load_csv`, `save_csv`
- **Preprocessing:** `StandardScaler`, `MinMaxScaler`
- **Metrics:** `accuracy`, `precision`, `recall`, `f1_score`,
  `confusion_matrix`, `classification_report`, `r2_score`, `rmse`,
  `mae`, `regression_report`
- **Math:** `dot`, `euclidean_dist`, `manhattan_distance`, `cosine`,
  `variance`, `covariance`, `matmul`, `matinv`, `sigmoid`
