// Linear regression demo: 1D then 2D synthetic regression with scaling and R² report.
#include "../logger.hpp"
#include "../ino.h"
#include "../metrics.hpp"
#include "../pre.h"
#include "demo_common.hpp"
#include "../models.cpp"
#include <cmath>
#include <random>

int main() {
    Log::header("LinearRegression Demo");
    Log::info("Part 1: y ≈ 3x (1 feature). Part 2: two-feature linear surface.");

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Matrix X1;
    Vec y1;
    for (int i = 0; i < 50; ++i) {
        double x = static_cast<double>(i) / 10.0;
        X1.push_back({x});
        y1.push_back(3.0 * x + noise(rng));
    }

    Log::info("1D dataset — shape: ", X1.size(), " x 1");
    print_matrix(X1, 5);
    print_vec(y1, 5);

    auto [X_tr, X_te, y_tr, y_te] = split_xy(X1, y1, 0.2, 42);
    StandardScaler sc1;
    sc1.fit(X_tr);
    X_tr = sc1.transform(X_tr);
    X_te = sc1.transform(X_te);

    LinearRegression lr1(GradientDescent<LinearParams, Matrix>::MiniBatch, 0.05, 16, 0.001);
    lr1.fit(X_tr, y_tr, 200, true);
    print_loss_bars(lr1.costs);
    regression_report(y_te, lr1.predict(X_te));

    Log::divider();
    Matrix X2;
    Vec y2;
    for (int i = 0; i < 40; ++i) {
        double x1 = noise(rng), x2 = noise(rng);
        X2.push_back({x1, x2});
        y2.push_back(2.0 * x1 - x1 * x2 + noise(rng));
    }
    Log::info("2D dataset — shape: ", X2.size(), " x 2");
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
    regression_report(y_te, lr2.predict(X_te));

    LinearRegression::docs();
    Log::divider();
    Log::success("LinearRegression demo finished.");
    return 0;
}
