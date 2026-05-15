// Lasso demo: sparse-relevant features; L1 drives unused weights toward zero.
#include "../logger.hpp"
#include "../metrics.hpp"
#include "../pre.h"
#include "demo_common.hpp"
#include "../models.h"
#include <random>

using namespace cml;

int main() {
    Log::header("LassoRegression Demo");
    Log::info("Only feature 0 drives y; features 1-4 are noise columns.");

    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0.0, 0.3);
    Matrix X;
    Vec y;
    for (int i = 0; i < 80; ++i) {
        double f0 = noise(rng);
        X.push_back({f0, noise(rng), noise(rng), noise(rng), noise(rng)});
        y.push_back(4.0 * f0 + noise(rng));
    }

    auto [X_train, X_test, y_train, y_test] = split_xy(X, y);
    StandardScaler scaler;
    scaler.fit(X_train);
    X_train = scaler.transform(X_train);
    X_test = scaler.transform(X_test);

    LassoRegression ls(GradientDescent<LinearParams, Matrix>::Batch, 0.05, 32, 0.2);
    ls.fit(X_train, y_train, 200, true);
    print_loss_bars(ls.costs);

    Log::info("Learned weights:");
    print_vec(ls.weights, 5);

    Vec preds = ls.predict(X_test);
    regression_report(y_test, preds);
    LassoRegression::docs();
    Log::divider();
    Log::success("LassoRegression demo finished.");
    return 0;
}
