#pragma once
#include "base_linear_model.hpp"
#include "../logger.hpp"
#include <numeric>

class [[maybe_unused]] RidgeRegression : public BaseLinearModel {
public:
    using BaseLinearModel::BaseLinearModel;

    static void docs() {
        Log::header("RidgeRegression");
        Log::info("Linear regression with L2 penalty in the gradient.");
        Log::info("Constructor: RidgeRegression(grad_type, lr, batch_size, l2_lambda)");
        Log::info("fit(X, y, n_epochs, verbose)");
        Log::info("predict(X) → Vec");
        Log::info("score(y_true, y_pred) → R²");
        Log::divider();
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
            gradients.weights[j] = sum_error_for_weight + (optimizer->get_l2_lambda() * params.weights[j]);
        }

        gradients.bias = sum_error_for_bias;
    }
};
