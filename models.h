#pragma once
#include "guard.hpp"
#include "params.hpp"
#include "optimizer.h"
#include "pre.h"
#include "utils.h"
#include "metrics.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace cml {


namespace {

inline bool log_epoch_milestone(size_t epoch, size_t total, bool verbose) {
    if (!verbose) return false;
    const size_t step = std::max(size_t(1), total / 10);
    return epoch % step == 0 || epoch == total - 1;
}

} // namespace

// LinearRegression
// Fits a linear model y = Xw + b using gradient descent (Batch/SGD/MiniBatch).
// Supports L2 regularization via the optimizer's l2_lambda parameter.
// Usage:
//   LinearRegression lr(GradientDescent<...>::Batch, 0.01, 32, 0.001);
//   lr.fit(X_train, y_train, 500);
//   Vec preds = lr.predict(X_test);
//   double r2 = lr.score(y_test, preds);
class [[maybe_unused]] LinearRegression {
public:
    Vec weights;
    StandardScaler scaler;
    double bias;
    Vec costs;
    GradientDescent<LinearParams, Matrix> *optimizer;
    bool owns_optimizer;

    LinearRegression(
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

    LinearRegression(GradientDescent<LinearParams, Matrix> *opt) {
        optimizer = opt;
        owns_optimizer = false;
        this->scaler = StandardScaler();
        this->bias = 0.0;
        this->costs = {};
    }

    ~LinearRegression() {
        if (owns_optimizer && optimizer) {
            delete optimizer;
        }
    }

    static void docs() {
        Log::header("LinearRegression");
        Log::info("Fits y = Xw + b using gradient descent.");
        Log::info("Constructor: LinearRegression(grad_type, lr, batch_size, l2_lambda)");
        Log::info("fit(X, y, n_epochs, verbose)");
        Log::info("predict(X) → Vec");
        Log::info("score(y_true, y_pred) → R²");
        Log::divider();
    }

private:
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients) {
        size_t n_features = X[0].size();
        size_t batch_size = end - start;
        Vec predictions(batch_size);
        double sum_error_for_bias = 0.0;

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = start[i];
            predictions[i] = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.
                             bias;
            sum_error_for_bias += (predictions[i] - y[idx]);
        }

        gradients.weights.resize(n_features);
        for (size_t j = 0; j < n_features; ++j) {
            double sum_error_for_weight = 0.0;
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = start[i];
                sum_error_for_weight += (predictions[i] - y[idx]) * X[idx][j];
            }
            gradients.weights[j] = sum_error_for_weight;
        }

        gradients.bias = sum_error_for_bias;
    }

public:
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false) {
        chk_fit(X, y);
        Matrix X_scaled = X;

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

            double total_error = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = std::inner_product(X_scaled[i].begin(), X_scaled[i].end(), weights.begin(), 0.0) + bias;
                total_error += (pred - y[i]) * (pred - y[i]);
            }

            double l2_cost = 0.0;
            for (const auto &w: weights) {
                l2_cost += w * w;
            }
            double l2_lambda = optimizer->get_l2_lambda();
            costs.push_back(
                (total_error / static_cast<double>(n_samples)) + (l2_lambda / (2.0 * static_cast<double>(n_samples))) *
                l2_cost);
            if (log_epoch_milestone(epoch, n_epochs, verbose)) {
                Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs),
                           Color::Yellow, " loss=", Color::White, costs.back());
            }
        }
        if (verbose && !costs.empty()) {
            Log::success("LinearRegression fit complete. final loss=", costs.back());
        }
    }

    Vec predict(const Matrix &X) {
        if (!weights.empty()) chk_predict(X, weights.size());
        Matrix X_scaled = X;
        X_scaled = scaler.transform(X_scaled);

        Vec predictions;
        predictions.reserve(X_scaled.size());
        for (const auto &sample: X_scaled) {
            predictions.push_back(std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias);
        }
        return predictions;
    }

    double score(const Vec &y_true, const Vec &y_pred) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        }

        long double ss_res = 0.0, ss_tot = 0.0;
        long double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
        for (size_t i = 0; i < y_true.size(); ++i) {
            ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
            ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
        }
        return 1 - (ss_res / ss_tot);
    }
};

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
        Log::info("score(y_true, y_pred) → pseudo R²");
        Log::divider();
    }

private:
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients) {
        size_t n_features = params.weights.size();
        size_t batch_size = end - start;


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
        chk_fit(X, y);
        Matrix X_scaled = X;

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
            if (log_epoch_milestone(epoch, n_epochs, verbose)) {
                Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs),
                           Color::Yellow, " loss=", Color::White, costs.back());
            }
        }
        if (verbose && !costs.empty()) {
            Log::success("LogisticRegression fit complete. final loss=", costs.back());
        }
    }


    Vec predict_prob(const Matrix &X) {
        if (!weights.empty()) chk_predict(X, weights.size());
        Matrix X_scaled = X;
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
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        }

        long double ss_res = 0.0, ss_tot = 0.0;
        long double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
        for (size_t i = 0; i < y_true.size(); ++i) {
            ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
            ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
        }
        return 1 - (ss_res / ss_tot);
    }
};

// RidgeRegression
// Linear regression with L2 penalty applied directly in the gradient (not via optimizer).
// Usage:
//   RidgeRegression rr(GradientDescent<...>::Batch, 0.01, 32, 1.0);
//   rr.fit(X_train, y_train, 500);
//   Vec preds = rr.predict(X_test);
class [[maybe_unused]] RidgeRegression {
public:
    Vec weights;
    StandardScaler scaler;
    double bias;
    Vec costs;
    GradientDescent<LinearParams, Matrix> *optimizer;
    bool owns_optimizer;

    RidgeRegression(
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

    RidgeRegression(GradientDescent<LinearParams, Matrix> *opt) {
        optimizer = opt;
        owns_optimizer = false;
        this->scaler = StandardScaler();
        this->bias = 0.0;
        this->costs = {};
    }

    ~RidgeRegression() {
        if (owns_optimizer && optimizer) {
            delete optimizer;
        }
    }

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
                           const LinearParams &params, LinearParams &gradients) {
        size_t n_features = X[0].size();
        size_t batch_size = end - start;
        Vec predictions(batch_size);
        double sum_error_for_bias = 0.0;

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = start[i];
            predictions[i] = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.
                             bias;
            sum_error_for_bias += (predictions[i] - y[idx]);
        }

        gradients.weights.resize(n_features);
        for (size_t j = 0; j < n_features; ++j) {
            double sum_error_for_weight = 0.0;
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = start[i];
                sum_error_for_weight += (predictions[i] - y[idx]) * X[idx][j];
            }
            gradients.weights[j] = sum_error_for_weight + (optimizer->get_l2_lambda() * params.weights[j]);;
        }

        gradients.bias = sum_error_for_bias;
    }

public:
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false) {
        chk_fit(X, y);
        Matrix X_scaled = X;

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

            double total_error = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = std::inner_product(X_scaled[i].begin(), X_scaled[i].end(), weights.begin(), 0.0) + bias;
                total_error += (pred - y[i]) * (pred - y[i]);
            }

            double l2_cost = 0.0;
            for (const auto &w: weights) {
                l2_cost += w * w;
            }
            double l2_lambda = optimizer->get_l2_lambda();
            costs.push_back(
                (total_error / static_cast<double>(n_samples)) + (l2_lambda / (2.0 * static_cast<double>(n_samples))) *
                l2_cost);
            if (log_epoch_milestone(epoch, n_epochs, verbose)) {
                Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs),
                           Color::Yellow, " loss=", Color::White, costs.back());
            }
        }
        if (verbose && !costs.empty()) {
            Log::success("RidgeRegression fit complete. final loss=", costs.back());
        }
    }

    Vec predict(const Matrix &X) {
        if (!weights.empty()) chk_predict(X, weights.size());
        Matrix X_scaled = X;
        X_scaled = scaler.transform(X_scaled);

        Vec predictions;
        predictions.reserve(X_scaled.size());
        for (const auto &sample: X_scaled) {
            predictions.push_back(std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias);
        }
        return predictions;
    }

    double score(const Vec &y_true, const Vec &y_pred) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        }

        long double ss_res = 0.0, ss_tot = 0.0;
        long double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
        for (size_t i = 0; i < y_true.size(); ++i) {
            ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
            ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
        }
        return 1.0 - (ss_res / ss_tot);
    }
};

// LassoRegression
// Linear regression with L1 penalty (subgradient) applied in compute_gradients.
// Usage:
//   LassoRegression lr(GradientDescent<...>::Batch, 0.01, 32, 0.1);
//   lr.fit(X_train, y_train, 500);
//   Vec preds = lr.predict(X_test);
class [[maybe_unused]] LassoRegression {
public:
    Vec weights;
    StandardScaler scaler;
    double bias;
    Vec costs;
    GradientDescent<LinearParams, Matrix> *optimizer;
    bool owns_optimizer;

    LassoRegression(
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

    LassoRegression(GradientDescent<LinearParams, Matrix> *opt) {
        optimizer = opt;
        owns_optimizer = false;
        this->scaler = StandardScaler();
        this->bias = 0.0;
        this->costs = {};
    }

    ~LassoRegression() {
        if (owns_optimizer && optimizer) {
            delete optimizer;
        }
    }

    static void docs() {
        Log::header("LassoRegression");
        Log::info("Linear regression with L1 penalty (subgradient) in the gradient.");
        Log::info("Constructor: LassoRegression(grad_type, lr, batch_size, l1_lambda)");
        Log::info("fit(X, y, n_epochs, verbose)");
        Log::info("predict(X) → Vec");
        Log::info("score(y_true, y_pred) → R²");
        Log::divider();
    }

private:
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients) {
        size_t n_features = X[0].size();
        size_t batch_size = end - start;
        Vec predictions(batch_size);
        double sum_error_for_bias = 0.0;

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = start[i];
            predictions[i] = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.
                             bias;
            sum_error_for_bias += (predictions[i] - y[idx]);
        }

        gradients.weights.resize(n_features);
        for (size_t j = 0; j < n_features; ++j) {
            double sum_error_for_weight = 0.0;
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = start[i];
                sum_error_for_weight += (predictions[i] - y[idx]) * X[idx][j];
            }

            // Subgradient of |w| for L1 penalty
            double sign_w = (params.weights[j] > 0) ? 1.0 : (params.weights[j] < 0 ? -1.0 : 0.0);
            gradients.weights[j] = sum_error_for_weight + (optimizer->get_l2_lambda() * sign_w);
        }

        gradients.bias = sum_error_for_bias;
    }

public:
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false) {
        chk_fit(X, y);
        Matrix X_scaled = X;

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

            double total_error = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = std::inner_product(X_scaled[i].begin(), X_scaled[i].end(), weights.begin(), 0.0) + bias;
                total_error += (pred - y[i]) * (pred - y[i]);
            }

            double l2_cost = 0.0;
            for (const auto &w: weights) {
                l2_cost += w * w;
            }

            double l1_cost = 0.0;
            for (const auto &w: weights) {
                l1_cost += std::abs(w);
            }
            double l2_lambda = optimizer->get_l2_lambda();
            costs.push_back(
                (total_error / static_cast<double>(n_samples)) + (l2_lambda / static_cast<double>(n_samples)) *
                l1_cost);
            if (log_epoch_milestone(epoch, n_epochs, verbose)) {
                Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs),
                           Color::Yellow, " loss=", Color::White, costs.back());
            }
        }
        if (verbose && !costs.empty()) {
            Log::success("LassoRegression fit complete. final loss=", costs.back());
        }
    }

    Vec predict(const Matrix &X) {
        if (!weights.empty()) chk_predict(X, weights.size());
        Matrix X_scaled = X;
        X_scaled = scaler.transform(X_scaled);

        Vec predictions;
        predictions.reserve(X_scaled.size());
        for (const auto &sample: X_scaled) {
            predictions.push_back(std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias);
        }
        return predictions;
    }

    double score(const Vec &y_true, const Vec &y_pred) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        }

        long double ss_res = 0.0, ss_tot = 0.0;
        long double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
        for (size_t i = 0; i < y_true.size(); ++i) {
            ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
            ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
        }
        return 1.0 - (ss_res / ss_tot);
    }
};

// DecisionTree
// CART-style classifier using Gini impurity and recursive binary splits.
// Usage:
//   DecisionTree dt(10, 2);
//   dt.fit(X_train, y_train);
//   Vec preds = dt.predict(X_test);
class [[maybe_unused]] DecisionTree {
public:
    struct Node {
        int feature_idx = -1;
        double threshold = 0.0;
        Node *left = nullptr;
        Node *right = nullptr;
        int label = -1;

        [[nodiscard]] bool is_leaf() const { return feature_idx == -1; }

        ~Node() {
            delete left;
            delete right;
        }
    };

private:
    struct SplitResult {
        int feature_idx = -1;
        double threshold = 0.0;
    };

    int max_depth;
    int min_samples_split;
    Node *root = nullptr;
    StandardScaler scaler;

    // Gini impurity: measures node "mixedness". Pure node = 0.
    double compute_gini(const std::vector<size_t> &indices, const Vec &y) {
        if (indices.empty()) {
            return 0.0;
        }
        std::map<int, int> counts;
        for (size_t idx: indices) {
            counts[static_cast<int>(y[idx])]++;
        }

        double gini = 1.0;
        for (auto const &[label, count]: counts) {
            double p = static_cast<double>(count) / indices.size();
            gini -= p * p;
        }
        return gini;
    }

    int majority_vote(const std::vector<size_t> &indices, const Vec &y) {
        std::map<int, int> counts;
        for (size_t idx: indices) {
            counts[static_cast<int>(y[idx])]++;
        }
        int max_count = 0;
        int majority_label = -1;
        for (auto const &[label, count]: counts) {
            if (count > max_count) {
                max_count = count;
                majority_label = label;
            }
        }
        return majority_label;
    }

    SplitResult best_split(const Matrix &X, const Vec &y, const std::vector<size_t> &indices) {
        double best_ig = 0.0;
        int best_feature = -1;
        double best_threshold = 0.0;
        double current_gini = compute_gini(indices, y);

        size_t n_features = X[0].size();
        for (size_t j = 0; j < n_features; ++j) {
            std::set<double> unique_values;
            for (size_t idx: indices) {
                unique_values.insert(X[idx][j]);
            }

            for (double threshold: unique_values) {
                std::vector<size_t> left_indices, right_indices;
                for (size_t idx: indices) {
                    if (X[idx][j] <= threshold) {
                        left_indices.push_back(idx);
                    } else {
                        right_indices.push_back(idx);
                    }
                }

                if (left_indices.empty() || right_indices.empty()) {
                    continue;
                }

                double left_gini = compute_gini(left_indices, y);
                double right_gini = compute_gini(right_indices, y);
                double p_left = static_cast<double>(left_indices.size()) / indices.size();
                double p_right = static_cast<double>(right_indices.size()) / indices.size();
                double ig = current_gini - (p_left * left_gini + p_right * right_gini);

                if (ig > best_ig) {
                    best_ig = ig;
                    best_feature = j;
                    best_threshold = threshold;
                }
            }
        }
        return {best_feature, best_threshold};
    }

    Node *build_tree(const Matrix &X, const Vec &y, const std::vector<size_t> &indices, int depth) {
        if (depth >= max_depth || indices.size() < min_samples_split) {
            Node *leaf = new Node();
            leaf->label = majority_vote(indices, y);
            return leaf;
        }

        std::set<int> labels;
        for (size_t idx: indices) {
            labels.insert(static_cast<int>(y[idx]));
        }
        if (labels.size() == 1) {
            Node *leaf = new Node();
            leaf->label = *labels.begin();
            return leaf;
        }

        SplitResult split = best_split(X, y, indices);

        if (split.feature_idx == -1) {
            Node *leaf = new Node();
            leaf->label = majority_vote(indices, y);
            return leaf;
        }

        std::vector<size_t> left_indices, right_indices;
        for (size_t idx: indices) {
            if (X[idx][split.feature_idx] <= split.threshold) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        Node *node = new Node();
        node->feature_idx = split.feature_idx;
        node->threshold = split.threshold;
        node->left = build_tree(X, y, left_indices, depth + 1);
        node->right = build_tree(X, y, right_indices, depth + 1);
        return node;
    }

    int traverse(const Vec &sample, Node *node) const {
        if (node->is_leaf()) {
            return node->label;
        }
        if (sample[node->feature_idx] <= node->threshold) {
            return traverse(sample, node->left);
        } else {
            return traverse(sample, node->right);
        }
    }

    void print_tree_helper(Node *node, int depth, const char *branch) const {
        if (!node) return;
        if (node->is_leaf()) {
            Log::tree_node(depth, true, branch, "class=", node->label, " ", Color::Green, "✓");
        } else {
            Log::tree_node(depth, false, branch, "X[", node->feature_idx, "] <= ", node->threshold);
            print_tree_helper(node->left, depth + 1, "├── ");
            print_tree_helper(node->right, depth + 1, "└── ");
        }
    }

public:
    DecisionTree(int max_depth = 10, int min_samples_split = 2)
        : max_depth(max_depth), min_samples_split(min_samples_split), root(nullptr) {
    }

    ~DecisionTree() {
        delete root;
    }

    static void docs() {
        Log::header("DecisionTree");
        Log::info("CART classifier using Gini impurity and recursive splits.");
        Log::info("Constructor: DecisionTree(max_depth, min_samples_split)");
        Log::info("fit(X, y, verbose), predict(X) → Vec, print_tree()");
        Log::info("score(y_true, y_pred) → accuracy");
        Log::divider();
    }

    void fit(const Matrix &X_, const Vec &y, bool verbose = false) {
        chk_fit(X_, y);
        Matrix X = X_;
        scaler.fit(X);
        X = scaler.transform(X);
        if (root) {
            delete root;
            root = nullptr;
        }
        std::vector<size_t> initial_indices(X.size());
        std::iota(initial_indices.begin(), initial_indices.end(), 0);
        root = build_tree(X, y, initial_indices, 0);
        if (verbose) {
            Log::success("DecisionTree fit complete.");
        }
    }

    Vec predict(const Matrix &X_) {
        if (!X_.empty() && root) chk_predict(X_, X_[0].size());
        Matrix X = X_;
        X = scaler.transform(X);
        Vec predictions;
        predictions.reserve(X.size());
        for (const auto &sample: X) {
            predictions.push_back(traverse(sample, root));
        }
        return predictions;
    }

    double score(const Vec &y_true, const Vec &y_pred) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        }
        double correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) {
                correct++;
            }
        }
        return correct / y_true.size();
    }

    void print_tree() const {
        Log::header("DecisionTree");
        print_tree_helper(root, 0, "└── ");
    }
};

// KNearestNeighbors
// Lazy learner: stores training data, classifies by majority vote among k nearest neighbors.
// Usage:
//   KNearestNeighbors knn(5, KNearestNeighbors::Euclidean);
//   knn.fit(X_train, y_train);
//   Vec preds = knn.predict(X_test);
class [[maybe_unused]] KNearestNeighbors {
public:
    enum DistanceType { Manhattan, Euclidean, Cosine };

    KNearestNeighbors(int k = 5, DistanceType distance_type = Euclidean)
        : k(k), distance_type(distance_type) {
    }

    static void docs() {
        Log::header("KNearestNeighbors");
        Log::info("Lazy learner: majority vote among k nearest neighbors.");
        Log::info("Constructor: KNearestNeighbors(k, distance_type)");
        Log::info("fit(X, y, verbose), predict(X) → Vec");
        Log::info("score(y_true, y_pred) → accuracy");
        Log::divider();
    }

    int k;
    DistanceType distance_type;
    StandardScaler scaler;
    Matrix X_train;
    Vec y_train;

    void fit(const Matrix &X, const Vec &y, bool verbose = false) {
        chk_fit(X, y);
        scaler.fit(X);
        X_train = scaler.transform(X);
        y_train = y;
        if (verbose) {
            Log::success("KNearestNeighbors fit complete. stored ", X_train.size(), " samples.");
        }
    }

    double compute_distance(const std::vector<double> &a, const std::vector<double> &b) {
        if (distance_type == Euclidean) {
            return euclidean_dist(a, b);
        } else if (distance_type == Manhattan) {
            return manhattan_distance(a, b);
        } else if (distance_type == Cosine) {
            return cosine(a, b);
        }

        return euclidean_dist(a, b); //default
    }

    Vec predict(const Matrix &X_) {
        if (!X_train.empty() && !X_.empty()) chk_predict(X_, X_train[0].size());
        // Clamp k to training set size if user passed k > n_train
        if (k > static_cast<int>(X_train.size())) {
            Log::warn("k exceeded training set size; clamped to n_train=", X_train.size());
        }
        k = std::min(k, static_cast<int>(X_train.size()));


        Matrix X = scaler.transform(X_);
        Vec predictions;
        predictions.reserve(X.size());

        for (const auto &sample: X) {
            std::vector<std::pair<double, int> > distances;
            for (size_t i = 0; i < X_train.size(); ++i) {
                double dist = compute_distance(sample, X_train[i]);
                distances.emplace_back(dist, static_cast<int>(y_train[i]));
            }
            std::nth_element(distances.begin(), distances.begin() + k, distances.end());
            std::map<int, int> class_counts;
            for (int i = 0; i < k; ++i) {
                class_counts[distances[i].second]++;
            }
            int max_count = 0;
            int predicted_label = -1;
            for (const auto &[label, count]: class_counts) {
                if (count > max_count) {
                    max_count = count;
                    predicted_label = label;
                }
            }
            predictions.push_back(predicted_label);
        }
        return predictions;
    }

    long double score(const Vec &y_true, const Vec &y_pred) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        }
        long double correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) {
                correct++;
            }
        }
        return correct / static_cast<long double>(y_true.size());
    }
};

// KMeans
// Partitions data into k clusters by minimizing within-cluster sum of squared distances.
// Supports random or K-Means++ initialization and multiple restarts.
// Usage:
//   KMeans km(3, 300, 10, KMeans::KMeansPP);
//   km.fit(X);
//   std::vector<int> labels = km.predict(X);
class [[maybe_unused]] KMeans {
public:
    enum InitType { Random, KMeansPP };

private:
    int k;
    int max_iter;
    int n_restarts;
    InitType init_type;
    double best_inertia;
    StandardScaler scaler;
    Matrix X_train;
    Matrix centroids;
    std::vector<Matrix> clusters;

    void init_cluster_size() {
        clusters.assign(k, Matrix{});
        for (int i = 0; i < k; ++i)
            clusters[i].reserve(X_train.size());
    }

    // K-Means++ init: spread centroids by weighting distance² from existing centroids
    Matrix init_centroids(const Matrix& X) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, static_cast<int>(X.size()) - 1);
        Matrix cents;

        if (init_type == Random) {
            for (int i = 0; i < k; ++i)
                cents.push_back(X[dist(rng)]);
            return cents;
        }

        // KMeans++
        cents.push_back(X[dist(rng)]);
        for (int i = 1; i < k; ++i) {
            Vec distances(X.size());
            for (size_t j = 0; j < X.size(); ++j) {
                double min_d = std::numeric_limits<double>::max();
                for (const auto& c : cents) {
                    double d = square_distance(X[j], c);
                    if (d < min_d) min_d = d;
                }
                distances[j] = min_d;
            }
            std::discrete_distribution<int> weighted(distances.begin(), distances.end());
            cents.push_back(X[weighted(rng)]);
        }
        return cents;
    }

    void assign_clusters() {
        for (int i = 0; i < k; ++i) clusters[i].clear();
        for (auto& s : X_train) {
            double b_d = -1;
            int b_i = 0;
            for (int i = 0; i < k; ++i) {
                double d = square_distance(s, centroids[i]);
                if (b_d < 0 || d < b_d) { b_d = d; b_i = i; }
            }
            clusters[b_i].push_back(s);
        }
    }

    void update_centroids() {
        for (int i = 0; i < k; ++i) {
            if (clusters[i].empty()) {
                std::mt19937 mt(i);
                std::uniform_int_distribution<int> d(0, static_cast<int>(X_train.size()) - 1);
                centroids[i] = X_train[d(mt)];
                continue;
            }
            Vec n(X_train[0].size(), 0.0);
            for (const auto& p : clusters[i])
                for (size_t j = 0; j < p.size(); ++j)
                    n[j] += p[j];
            for (double& v : n) v /= clusters[i].size();
            centroids[i] = n;
        }
    }

    double run_once() {
        centroids = init_centroids(X_train);
        init_cluster_size();
        for (int iter = 0; iter < max_iter; ++iter) {
            Matrix old_centroids = centroids;
            assign_clusters();
            update_centroids();
            bool converged = true;
            for (int i = 0; i < k; ++i)
                if (square_distance(centroids[i], old_centroids[i]) > 1e-6)
                    { converged = false; break; }
            if (converged) break;
        }
        return score(X_train);
    }

public:
    KMeans(int k = 3, int max_iter = 300, int n_restarts = 10, InitType init_type = KMeansPP)
        : k(k), max_iter(max_iter), n_restarts(n_restarts),
          init_type(init_type), best_inertia(std::numeric_limits<double>::max()) {}

    static void docs() {
        Log::header("KMeans");
        Log::info("Partitions data into k clusters via Lloyd's algorithm.");
        Log::info("Constructor: KMeans(k, max_iter, n_restarts, init_type)");
        Log::info("fit(X, verbose), predict(X), get_centroids(), get_inertia()");
        Log::divider();
    }

    void fit(const Matrix& X, bool verbose = false) {
        chk_X(X);
        scaler.fit(X);
        X_train = scaler.transform(X);
        best_inertia = std::numeric_limits<double>::max();
        Matrix best_centroids;

        for (int r = 0; r < n_restarts; ++r) {
            double inertia = run_once();
            if (verbose) {
                Log::info("Restart ", r + 1, "/", n_restarts, "  inertia=", inertia);
            }
            if (inertia < best_inertia) {
                best_inertia = inertia;
                best_centroids = centroids;
            }
        }
        centroids = best_centroids;
        if (verbose) {
            Log::success("KMeans fit complete. best inertia=", best_inertia);
        }
    }

    double score(const Matrix& X) {
        double total = 0;
        for (const auto& point : X) {
            double min_d = -1;
            for (const auto& c : centroids) {
                double d = square_distance(point, c);
                if (min_d < 0 || d < min_d) min_d = d;
            }
            total += min_d;
        }
        return total;
    }

    std::vector<int> predict(const Matrix& X) {
        Matrix X_scaled = scaler.transform(X);
        std::vector<int> labels;
        for (const auto& p : X_scaled) {
            int best_idx = 0;
            double min_d = -1;
            for (int i = 0; i < k; ++i) {
                double d = square_distance(p, centroids[i]);
                if (min_d < 0 || d < min_d) { min_d = d; best_idx = i; }
            }
            labels.push_back(best_idx);
        }
        return labels;
    }

    Matrix get_centroids() const { return centroids; }
    double get_inertia() const { return best_inertia; }
};

// GaussianNB
// Naive Bayes classifier assuming Gaussian features with per-class means/variances.
// Usage:
//   GaussianNB gnb;
//   gnb.fit(X_train, y_train);
//   Vec preds = gnb.predict(X_test);
class [[maybe_unused]] GaussianNB {
public:
    GaussianNB() = default;

    static void docs() {
        Log::header("GaussianNB");
        Log::info("Gaussian Naive Bayes classifier (per-class means/variances).");
        Log::info("fit(X, y, verbose), predict(X), predict_prob(X)");
        Log::info("score(y_true, y_pred) → accuracy");
        Log::divider();
    }

private:
    Matrix means;
    Matrix variances;
    Vec priors;
    Matrix X_train;
    Vec y_train;
    std::vector<int> classes;

    inline double log_gaussian(double x, double mean, double variance) {
        const double var = std::max(variance, 1e-9);
        return -0.5 * (std::log(2 * M_PI * var) + std::pow(x - mean, 2) / var);
    }


    Matrix give_means(const Matrix &X, const Vec &y, const std::vector<int> &cls) {
        Matrix all_means;

        for (int c: cls) {
            Vec m(X[0].size(), 0.0);
            int count = 0;

            for (size_t i = 0; i < X.size(); ++i) {
                if (static_cast<int>(y[i]) == c) {
                    for (size_t j = 0; j < X[0].size(); ++j) {
                        m[j] += X[i][j];
                    }
                    count++;
                }
            }

            for (double &val: m) {
                val /= (count > 0 ? count : 1);
            }

            all_means.push_back(m);
        }

        return all_means;
    }

    Matrix give_variances(const Matrix &X,
                          const Vec &y,
                          const std::vector<int> &cls,
                          const Matrix &all_means) {
        Matrix all_vars;

        for (size_t k = 0; k < cls.size(); ++k) {
            Vec v(X[0].size(), 0.0);
            int count = 0;
            int target_class = cls[k];

            for (size_t i = 0; i < X.size(); ++i) {
                if (static_cast<int>(y[i]) == target_class) {
                    for (size_t j = 0; j < X[0].size(); ++j) {
                        double diff = X[i][j] - all_means[k][j];
                        v[j] += diff * diff;
                    }
                    count++;
                }
            }

            for (double &val: v) {
                val = (val / (count > 0 ? count : 1)) + 1e-9;
            }

            all_vars.push_back(v);
        }

        return all_vars;
    }

    Vec give_priors(const Vec &y) {
        Vec priors_(classes.size(), 0.0);
        for (size_t k = 0; k < classes.size(); ++k) {
            int count = 0;
            for (double label: y)
                if (static_cast<int>(label) == classes[k]) count++;
            priors_[k] = static_cast<double>(count) / y.size();
        }
        return priors_;
    }

    void init() {
        means = give_means(X_train, y_train, classes);
        variances = give_variances(X_train, y_train, classes, means);
        priors = give_priors(y_train);
    }

    void extract_classes() {
        std::set<int> unique_classes;
        for (double label: y_train) {
            unique_classes.insert(static_cast<int>(label));
        }
        classes.assign(unique_classes.begin(), unique_classes.end());
    }

public:
    void fit(const Matrix &X, const Vec &y, bool verbose = false) {
        chk_fit(X, y);
        X_train = X;
        y_train = y;
        extract_classes();
        init();
        if (verbose) {
            Log::success("GaussianNB fit complete. classes=", classes.size());
        }
    }

    Vec predict(const Matrix &X) {
        if (!X_train.empty() && !X.empty()) chk_predict(X, X_train[0].size());
        Vec preds;

        for (const auto &sample: X) {
            Vec scores = log_posteriors(sample);

            int best_idx = 0;
            for (size_t i = 1; i < classes.size(); ++i) {
                if (scores[i] > scores[best_idx]) {
                    best_idx = i;
                }
            }

            preds.push_back(classes[best_idx]);
        }

        return preds;
    }

    Matrix predict_prob(const Matrix &X) {
        Matrix all_scores;

        for (const auto &sample: X) {
            all_scores.push_back(log_posteriors(sample));
        }

        return all_scores;
    }

    double score(const Vec &y_true, const Vec &y_pred) {
        int correct = 0;

        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) {
                correct++;
            }
        }

        return static_cast<double>(correct) / y_true.size();
    }

private:
    Vec log_posteriors(const Vec &sample) {
        Vec scores;

        for (size_t k = 0; k < classes.size(); ++k) {
            // Work in log-space to avoid float underflow from multiplying many small probs
            double score = std::log(priors[k]);

            for (size_t j = 0; j < sample.size(); ++j) {
                score += log_gaussian(sample[j],
                                      means[k][j],
                                      variances[k][j]);
            }

            scores.push_back(score);
        }

        return scores;
    }
};

// SVM
// Linear SVM trained with hinge loss and SGD. Labels are mapped to {-1, +1} internally.
// Usage:
//   SVM svm(0.001, 0.01, 1000);
//   svm.fit(X_train, y_train);
//   Vec preds = svm.predict(X_test);
class [[maybe_unused]] SVM {
public:
    SVM(double lr = 0.001, double lambda = 0.01, int n_epochs = 1000)
        : lr(lr), lambda(lambda), n_epochs(n_epochs), bias(0.0) {
    }

    static void docs() {
        Log::header("SVM");
        Log::info("Linear SVM with hinge loss (SGD). Labels mapped to {-1,+1} internally.");
        Log::info("Constructor: SVM(lr, lambda, n_epochs)");
        Log::info("fit(X, y, verbose), predict(X) → Vec");
        Log::info("score(y_true, y_pred) → accuracy");
        Log::divider();
    }

    void fit(const Matrix &X, const Vec &y, bool verbose = false) {
        chk_fit(X, y);
        Matrix X_scaled = X;
        scaler.fit(X_scaled);
        X_scaled = scaler.transform(X_scaled);

        size_t n_samples = X.size();
        size_t n_features = X[0].size();
        weights.assign(n_features, 0.0);
        bias = 0.0;

        Vec y_svm = to_svm_labels(y);

        for (int epoch = 0; epoch < n_epochs; ++epoch) {
            for (size_t i = 0; i < n_samples; ++i) {
                double condition = y_svm[i] * (std::inner_product(X_scaled[i].begin(), X_scaled[i].end(),
                                                                  weights.begin(), 0.0) + bias);
                if (condition >= 1) {
                    for (size_t j = 0; j < n_features; ++j) {
                        weights[j] -= lr * (lambda * weights[j]);
                    }
                } else {
                    for (size_t j = 0; j < n_features; ++j) {
                        weights[j] -= lr * (lambda * weights[j] - y_svm[i] * X_scaled[i][j]);
                    }
                    bias -= lr * (-y_svm[i]);
                }
            }
            if (log_epoch_milestone(static_cast<size_t>(epoch), static_cast<size_t>(n_epochs), verbose)) {
                Vec preds_scaled;
                preds_scaled.reserve(n_samples);
                for (size_t i = 0; i < n_samples; ++i) {
                    double raw = std::inner_product(X_scaled[i].begin(), X_scaled[i].end(),
                                                    weights.begin(), 0.0) + bias;
                    preds_scaled.push_back((raw >= 0) ? 1.0 : -1.0);
                }
                Vec preds = from_svm_labels(preds_scaled);
                double acc = score(y, preds);
                Log::epoch(epoch + 1, n_epochs, Color::Yellow, " acc=", Color::White, acc);
            }
        }
        if (verbose) {
            Vec preds = predict(X);
            Log::success("SVM fit complete. train accuracy=", score(y, preds));
        }
    }

    Vec predict(const Matrix &X) {
        if (!weights.empty()) chk_predict(X, weights.size());
        Matrix X_scaled = X;
        X_scaled = scaler.transform(X_scaled);

        Vec y_pred_svm;
        y_pred_svm.reserve(X.size());

        for (const auto &sample: X_scaled) {
            double raw = std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias;
            y_pred_svm.push_back((raw >= 0) ? 1.0 : -1.0);
        }
        return from_svm_labels(y_pred_svm);
    }

    double score(const Vec &y_true, const Vec &y_pred) {
        if (y_true.empty() || y_true.size() != y_pred.size()) {
            return 0.0;
        }
        double correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) {
                correct++;
            }
        }
        return correct / y_true.size();
    }

private:
    double lr;
    double lambda;
    int n_epochs;
    Vec weights;
    double bias;
    StandardScaler scaler;

    // Map {0, 1} labels to {-1, +1} for hinge loss
    Vec to_svm_labels(const Vec &y) {
        Vec result;
        result.reserve(y.size());
        for (double label: y) {
            result.push_back((label == 0.0) ? -1.0 : 1.0);
        }
        return result;
    }

    // Map {-1, +1} predictions back to {0, 1}
    Vec from_svm_labels(const Vec &y) {
        Vec result;
        result.reserve(y.size());
        for (double label: y) {
            result.push_back((label == -1.0) ? 0.0 : 1.0);
        }
        return result;
    }
};
} // namespace cml
