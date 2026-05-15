// Linear regression demo: 1D then 2D synthetic regression with scaling and R² report.
#include "demo_common.hpp"
#include <cmath>
#include <random>

int main() {
    report_header("LinearRegression");

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Matrix X1;
    Vec y1;
    for (int i = 0; i < 50; ++i) {
        double x = static_cast<double>(i) / 10.0;
        X1.push_back({x});
        y1.push_back(3.0 * x + noise(rng));
    }
    report_dataset(X1, y1, "1D Training Set");

    auto [X_tr, X_te, y_tr, y_te] = split_xy(X1, y1, 0.2, 42);
    StandardScaler sc1;
    sc1.fit(X_tr);
    X_tr = sc1.transform(X_tr);
    X_te = sc1.transform(X_te);

    LinearRegression lr1(GradientDescent<LinearParams, Matrix>::MiniBatch, 0.05, 16, 0.001);
    lr1.fit(X_tr, y_tr, 200, true);
    print_loss_bars(lr1.costs);
    report_regression_metrics(y_te, lr1.predict(X_te));

    Matrix X2;
    Vec y2;
    for (int i = 0; i < 40; ++i) {
        double x1 = noise(rng), x2 = noise(rng);
        X2.push_back({x1, x2});
        y2.push_back(2.0 * x1 - x1 * x2 + noise(rng));
    }
    report_dataset(X2, y2, "2D Training Set");
    auto split2 = split_xy(X2, y2, 0.2, 42);
    X_tr = std::get<0>(split2);
    X_te = std::get<1>(split2);
    y_tr = std::get<2>(split2);
    y_te = std::get<3>(split2);
    StandardScaler sc2;
    sc2.fit(X_tr);
    X_tr = sc2.transform(X_tr);
    X_te = sc2.transform(X_te);

    LinearRegression lr2(GradientDescent<LinearParams, Matrix>::Batch, 0.05, 32, 0.0);
    lr2.fit(X_tr, y_tr, 150, true);
    report_regression_metrics(y_te, lr2.predict(X_te));

    Log::success("LinearRegression demo finished.");
    return 0;
}
