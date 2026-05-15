// Ridge regression demo on synthetic 2D data.
#include "demo_common.hpp"
#include <random>

int main() {
    report_header("RidgeRegression");
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0, 0.3);
    Matrix X;
    Vec y;
    for (int i = 0; i < 80; ++i) {
        double x1 = noise(rng), x2 = noise(rng);
        X.push_back({x1, x2});
        y.push_back(1.5 * x1 + 0.5 * x2 + noise(rng));
    }
    report_dataset(X, y);
    auto [X_tr, X_te, y_tr, y_te] = split_xy(X, y);
    StandardScaler sc;
    sc.fit(X_tr);
    X_tr = sc.transform(X_tr);
    X_te = sc.transform(X_te);
    RidgeRegression model(GradientDescent<LinearParams, Matrix>::Batch, 0.05, 32, 1.0);
    model.fit(X_tr, y_tr, 200, true);
    print_loss_bars(model.costs);
    report_regression_metrics(y_te, model.predict(X_te));
    Log::success("RidgeRegression demo finished.");
    return 0;
}
