#pragma once
#include "classicml.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

using namespace cml;

// Prints ASCII loss bars from a cost history vector.
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

// Stratified-style random train/test split for feature matrix and labels.
inline std::tuple<Matrix, Matrix, Vec, Vec>
split_xy(const Matrix &X, const Vec &y, double test_size = 0.2, unsigned seed = 42) {
    struct Sample { Vec row; double label; };
    std::vector<Sample> data;
    data.reserve(X.size());
    for (size_t i = 0; i < X.size(); ++i) data.push_back({X[i], y[i]});
    std::vector<Sample> data_mut = data;
    auto parts = train_test_split(data_mut, test_size, seed);
    Matrix X_train, X_test;
    Vec y_train, y_test;
    for (const auto &s : parts.first) { X_train.push_back(s.row); y_train.push_back(s.label); }
    for (const auto &s : parts.second) { X_test.push_back(s.row); y_test.push_back(s.label); }
    return {X_train, X_test, y_train, y_test};
}

// Full Report header: bold model name banner.
inline void report_header(const std::string &model_name) {
    Log::divider();
    Log::header(Style::Bold, "═══ ", model_name, " — Full Report ═══");
    Log::divider();
}

// Dataset dimensions and basic feature/label statistics.
inline void report_dataset(const Matrix &X, const Vec &y, const std::string &name = "Dataset") {
    Log::header(name);
    if (X.empty()) {
        Log::warn("Empty dataset.");
        return;
    }
    size_t rows = X.size();
    size_t cols = X[0].size();
    Log::metric("Samples", static_cast<int>(rows));
    Log::metric("Features", static_cast<int>(cols));
    if (!y.empty()) {
        double y_min = y[0], y_max = y[0], y_sum = 0;
        for (double v : y) {
            y_min = std::min(y_min, v);
            y_max = std::max(y_max, v);
            y_sum += v;
        }
        Log::metric("Label min", y_min);
        Log::metric("Label max", y_max);
        Log::metric("Label mean", y_sum / static_cast<double>(y.size()));
    }
    double f_sum = 0, f_min = X[0][0], f_max = X[0][0];
    size_t cnt = 0;
    for (const auto &row : X)
        for (double v : row) {
            f_sum += v;
            f_min = std::min(f_min, v);
            f_max = std::max(f_max, v);
            ++cnt;
        }
    if (cnt > 0) {
        Log::metric("Feature mean", f_sum / static_cast<double>(cnt));
        Log::metric("Feature min", f_min);
        Log::metric("Feature max", f_max);
    }
    Log::divider();
}

// Regression metrics block (R², RMSE, MAE).
inline void report_regression_metrics(const Vec &y_true, const Vec &y_pred) {
    Log::header("Metrics");
    Log::metric("R²", r2_score(y_true, y_pred));
    Log::metric("RMSE", rmse(y_true, y_pred));
    Log::metric("MAE", mae(y_true, y_pred));
    regression_report(y_true, y_pred);
}

// Classification metrics block (accuracy + full report).
inline void report_classification_metrics(const Vec &y_true, const Vec &y_pred) {
    Log::header("Metrics");
    Log::metric("Accuracy", accuracy(y_true, y_pred));
    classification_report(y_true, y_pred);
}
