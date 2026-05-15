// Lasso regression demo on synthetic 2D data.
#include "demo_common.hpp"
#include <random>

int main() {
    report_header("LassoRegression");
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0, 0.2);
    Matrix X;
    Vec y;
    for (int i = 0; i < 60; ++i) {
        X.push_back({noise(rng), noise(rng)});
        y.push_back(2.0 * X.back()[0] + noise(rng));
    }
    report_dataset(X, y);
    auto [X_tr, X_te, y_tr, y_te] = split_xy(X, y);
    StandardScaler sc;
    sc.fit(X_tr);
    X_tr = sc.transform(X_tr);
    X_te = sc.transform(X_te);
    LassoRegression model(GradientDescent<LinearParams, Matrix>::MiniBatch, 0.05, 16, 0.1);
    model.fit(X_tr, y_tr, 250, true);
    print_loss_bars(model.costs);
    report_regression_metrics(y_te, model.predict(X_te));
    Log::success("LassoRegression demo finished.");
    return 0;
}
