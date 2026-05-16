// Ridge regression demo: same synthetic data as linear regression; compare L2 strengths.
#include "../cml.hpp"
#include "demo_common.hpp"

int main() {
    Log::header("RidgeRegression Demo");
    Log::info("Shows weight shrinkage as L2 lambda increases.");

    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.4);
    Matrix X;
    Vec y;
    for (int i = 0; i < 60; ++i) {
        double x = static_cast<double>(i) / 8.0;
        X.push_back({x});
        y.push_back(2.5 * x + noise(rng));
    }

    auto [X_train, X_test, y_train, y_test] = split_xy(X, y);
    StandardScaler scaler;
    scaler.fit(X_train);
    X_train = scaler.transform(X_train);
    X_test = scaler.transform(X_test);

    for (double lam : {0.0, 0.1, 1.0}) {
        Log::info("Training with l2_lambda=", lam);
        RidgeRegression rr(GradientDescent<LinearParams, Matrix>::Batch, 0.05, 32, lam);
        rr.fit(X_train, y_train, 150, true);
        Vec preds = rr.predict(X_test);
        Log::metric("R²", rr.score(y_test, preds));
        Log::metric("|w|", std::abs(rr.weights[0]));
    }

    RidgeRegression::docs();
    Log::divider();
    Log::success("RidgeRegression demo finished.");
    return 0;
}
