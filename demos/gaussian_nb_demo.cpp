// Gaussian Naive Bayes demo: hand-crafted 3-class Gaussian clusters.
#include "../logger.hpp"
#include "../metrics.hpp"
#include "demo_common.hpp"
#include "../models.cpp"
#include <random>

int main() {
    Log::header("GaussianNB Demo");
    Log::info("Priors/means/variances model per-class Gaussian features.");

    std::mt19937 rng(0);
    std::normal_distribution<double> n0(0.0, 0.3), n1(3.0, 0.4), n2(-2.0, 0.5);
    Matrix X;
    Vec y;
    for (int i = 0; i < 40; ++i) {
        X.push_back({n0(rng), n0(rng)});
        y.push_back(0);
    }
    for (int i = 0; i < 40; ++i) {
        X.push_back({n1(rng), n1(rng)});
        y.push_back(1);
    }
    for (int i = 0; i < 40; ++i) {
        X.push_back({n2(rng), n2(rng)});
        y.push_back(2);
    }

    auto [X_train, X_test, y_train, y_test] = split_xy(X, y);

    GaussianNB gnb;
    gnb.fit(X_train, y_train, true);

    Matrix log_probs = gnb.predict_prob(X_test);
    Log::info("Log-posteriors for first 3 test samples:");
    for (int i = 0; i < 3 && i < static_cast<int>(log_probs.size()); ++i) {
        print_vec(log_probs[i], 3);
    }

    Vec preds = gnb.predict(X_test);
    classification_report(y_test, preds);
    GaussianNB::docs();
    Log::divider();
    Log::success("GaussianNB demo finished.");
    return 0;
}
