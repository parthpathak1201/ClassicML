#pragma once
#include "../cml.hpp"

inline void print_loss_bars(const Vec &costs) {
    if (costs.empty()) return;
    double max_c = *std::max_element(costs.begin(), costs.end());
    if (max_c <= 0) max_c = 1.0;
    Log::info("Loss curve (ASCII):");
    const size_t n = std::min(costs.size(), size_t(20));
    for (size_t i = 0; i < n; ++i) {
        int bars = static_cast<int>(40.0 * costs[i] / max_c);
        std::string bar(static_cast<size_t>(bars), '#');
        Log::info("  e", i + 1, " ", bar);
    }
}

inline std::tuple<Matrix, Matrix, Vec, Vec>
split_xy(const Matrix &X, const Vec &y, double test_size = 0.2, unsigned seed = 42) {
    struct Sample {
        Vec row;
        double label;
    };
    std::vector<Sample> data;
    data.reserve(X.size());
    for (size_t i = 0; i < X.size(); ++i) data.push_back({X[i], y[i]});

    std::vector<Sample> data_mut = data;
    auto parts = train_test_split(data_mut, test_size, seed);

    Matrix X_train, X_test;
    Vec y_train, y_test;
    for (const auto &s : parts.first) {
        X_train.push_back(s.row);
        y_train.push_back(s.label);
    }
    for (const auto &s : parts.second) {
        X_test.push_back(s.row);
        y_test.push_back(s.label);
    }
    return {X_train, X_test, y_train, y_test};
}
