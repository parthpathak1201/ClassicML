#pragma once
#include "../type.hpp"
#include "../params.hpp"
#include "../utils.h"
#include "../pre.h"
#include "../optimizer.h"
#include "../logger.hpp"
#include "detail.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <map>
#include <set>
#include <random>
#include <limits>
#include <string>

// LogisticRegression
// Binary classifier using sigmoid + gradient descent. Outputs probabilities via predict_prob().
// Supports L2 regularization via the optimizer's l2_lambda parameter.
// Usage:
//   LogisticRegression lr(GradientDescent<...>::MiniBatch, 0.01, 32, 0.0);
//   lr.fit(X_train, y_train, 500);
//   Vec preds = lr.predict(X_test);
class [[maybe_unused]] LogisticRegression {
public:
    Vec weights;
    StandardScaler scaler;
    double bias;
    Vec costs;
    GradientDescent<LinearParams, Matrix> *optimizer;
    bool owns_optimizer;

    explicit LogisticRegression(
        GradientDescent<LinearParams, Matrix>::Type grad_desc_type = GradientDescent<LinearParams, Matrix>::Batch,
        double learning_rate = 0.01,
        size_t m_batch_size = 32,
        double l2_lambda = 0.0) {
        optimizer = new GradientDescent<LinearParams, Matrix>(grad_desc_type, learning_rate, m_batch_size, l2_lambda);
        owns_optimizer = true;
        this->scaler = StandardScaler();
        this->bias = 0.0;
        this->costs = {};
    }

    explicit LogisticRegression(GradientDescent<LinearParams, Matrix> *opt) {
        optimizer = opt;
        owns_optimizer = false;
        this->scaler = StandardScaler();
        this->bias = 0.0;
        this->costs = {};
    }

    LogisticRegression(const LogisticRegression &other) {
        weights = other.weights;
        scaler = other.scaler;
        bias = other.bias;
        costs = other.costs;
        if (other.owns_optimizer) {
            optimizer = new GradientDescent<LinearParams, Matrix>(*other.optimizer);
            owns_optimizer = true;
        } else {
            optimizer = other.optimizer;
            owns_optimizer = false;
        }
    }

    ~LogisticRegression() {
        if (owns_optimizer && optimizer) {
            delete optimizer;
        }
    }

    static void docs() {
        Log::header("LogisticRegression");
        Log::info("Binary classifier using sigmoid + gradient descent.");
        Log::info("Constructor: LogisticRegression(grad_type, lr, batch_size, l2_lambda)");
        Log::info("fit(X, y, n_epochs, verbose)");
        Log::info("predict(X) → Vec, predict_prob(X) → Vec");
        Log::info("score(y_true, y_pred) → accuracy");
        Log::divider();
    }

private:
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients) {
        size_t n_features = params.weights.size();
        size_t batch_size = end - start;

        // Ensure gradients.weights has the correct size before writing to it
        gradients.weights.resize(n_features);
        std::fill(gradients.weights.begin(), gradients.weights.end(), 0.0);
        gradients.bias = 0.0;

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = start[i];


            double z = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.bias;
            double prediction = sigmoid(z);
            double error = prediction - y[idx];

            gradients.bias += error;


            for (size_t j = 0; j < n_features; ++j) {
                gradients.weights[j] += error * X[idx][j];
            }
        }
    }

public:
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false) {
        Matrix X_scaled = X;
        if (X_scaled.empty() || y.empty() || X_scaled.size() != y.size()) {
            throw std::invalid_argument("Input data is empty or dimensions do not match.");
        }

        scaler.fit(X_scaled);
        X_scaled = scaler.transform(X_scaled);

        size_t n_samples = X_scaled.size();
        size_t n_features = X_scaled[0].size();
        weights.resize(n_features);
        weights.assign(n_features, 0.0);
        bias = 0.0;
        costs.clear();

        optimizer->set_n_samples(n_samples);
        LinearParams params = {weights, bias};

        for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
            optimizer->shuffle_indices();

            auto grad_func = [this, &X_scaled, &y](const Matrix &, const size_t *start, const size_t *end,
                                                   const LinearParams &p, LinearParams &grad) {
                this->compute_gradients(X_scaled, y, start, end, p, grad);
            };

            optimizer->step(params, X_scaled, grad_func, epoch);

            weights = params.weights;
            bias = params.bias;

            double total_cost = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double z = std::inner_product(X_scaled[i].begin(), X_scaled[i].end(), weights.begin(), 0.0) + bias;
                double prediction = sigmoid(z);

                // Clip predictions to avoid log(0) in BCE loss
                if (prediction == 0) {
                    prediction = 1e-15;
                } else if (prediction == 1) {
                    prediction = 1 - 1e-15;
                }

                total_cost += -y[i] * log(prediction) - (1 - y[i]) * log(1 - prediction);
            }

            double l2_cost = 0.0;
            for (const auto &w: weights) {
                l2_cost += w * w;
            }
            double l2_lambda = optimizer->get_l2_lambda();
            costs.push_back(
                (total_cost / static_cast<double>(n_samples)) + (l2_lambda / (2.0 * static_cast<double>(n_samples))) *
                l2_cost);
            if (detail::log_epoch_milestone(epoch, n_epochs, verbose)) {
                Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs),
                           Color::Yellow, " loss=", Color::White, costs.back());
            }
        }
        if (verbose && !costs.empty()) {
            Log::success("LogisticRegression fit complete. final loss=", costs.back());
        }
    }


    Vec predict_prob(const Matrix &X) {
        Matrix X_scaled = X;
        if (X_scaled.empty() || X_scaled[0].size() != weights.size()) {
            throw std::invalid_argument("Input data for prediction is invalid or dimensions mismatch.");
        }

        X_scaled = scaler.transform(X_scaled);

        Vec predictions;
        predictions.reserve(X_scaled.size());
        for (const auto &sample: X_scaled) {
            double z = std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias;
            predictions.push_back(sigmoid(z));
        }
        return predictions;
    }

    Vec predict(const Matrix &X) {
        Vec temp = predict_prob(X);

        for (auto &p: temp) {
            if (p >= 0.5) {
                p = 1.0;
            } else {
                p = 0.0;
            }
        }
        return temp;
    }

    double score(const Vec &y_true, const Vec &y_pred) {
        if (y_true.size() != y_pred.size())
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        double correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i)
            if (y_true[i] == y_pred[i]) correct++;
        return correct / y_true.size();
    }
};
