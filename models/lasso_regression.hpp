#pragma once
#include "base_linear_model.hpp"
#include "../logger.hpp"
#include <numeric>
#include <cstdlib>

class [[maybe_unused]] LassoRegression : public BaseLinearModel {
public:
    using BaseLinearModel::BaseLinearModel;

    static void docs() {
        Log::header("LassoRegression");
        Log::info("Linear regression with L1 penalty (subgradient) in the gradient.");
        Log::info("Constructor: LassoRegression(grad_type, lr, batch_size, l1_lambda)");
        Log::info("fit(X, y, n_epochs, verbose)");
        Log::info("predict(X) → Vec");
        Log::info("score(y_true, y_pred) → R²");
        Log::divider();
    }

    double compute_regularization_cost(const Vec &w, size_t n_samples) const override {
        double l1 = 0.0;
        for (const auto &v : w) l1 += std::abs(v);
        double lambda = optimizer->get_l2_lambda();
        if (lambda == 0.0) return 0.0;
        return (lambda / static_cast<double>(n_samples)) * l1;
    }

private:
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients) override {
        size_t n_features = X[0].size();
        size_t batch_size = end - start;
        Vec predictions(batch_size);
        double sum_error_for_bias = 0.0;

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = start[i];
            predictions[i] = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.bias;
            sum_error_for_bias += (predictions[i] - y[idx]);
        }

        gradients.weights.resize(n_features);
        for (size_t j = 0; j < n_features; ++j) {
            double sum_error_for_weight = 0.0;
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = start[i];
                sum_error_for_weight += (predictions[i] - y[idx]) * X[idx][j];
            }

            double sign_w = (params.weights[j] > 0) ? 1.0 : (params.weights[j] < 0 ? -1.0 : 0.0);
            gradients.weights[j] = sum_error_for_weight + (optimizer->get_l2_lambda() * sign_w);
        }

        gradients.bias = sum_error_for_bias;
    }
};
