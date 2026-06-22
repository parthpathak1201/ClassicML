#pragma once
#include "../estimator.hpp"
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
#include <string>

class BaseLinearModel : public Estimator {
public:
    Vec weights;
    StandardScaler scaler;
    double bias;
    Vec costs;
    GradientDescent<LinearParams, Matrix> *optimizer;
    bool owns_optimizer;

    BaseLinearModel(
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

    BaseLinearModel(GradientDescent<LinearParams, Matrix> *opt) {
        optimizer = opt;
        owns_optimizer = false;
        this->scaler = StandardScaler();
        this->bias = 0.0;
        this->costs = {};
    }

    BaseLinearModel(const BaseLinearModel&) = delete;
    BaseLinearModel& operator=(const BaseLinearModel&) = delete;

    BaseLinearModel(BaseLinearModel&& other) noexcept
        : weights(std::move(other.weights)),
          scaler(std::move(other.scaler)),
          bias(other.bias),
          costs(std::move(other.costs)),
          optimizer(other.optimizer),
          owns_optimizer(other.owns_optimizer) {
        other.optimizer = nullptr;
        other.owns_optimizer = false;
    }

    BaseLinearModel& operator=(BaseLinearModel&& other) noexcept {
        if (this != &other) {
            if (owns_optimizer && optimizer) delete optimizer;
            weights = std::move(other.weights);
            scaler = std::move(other.scaler);
            bias = other.bias;
            costs = std::move(other.costs);
            optimizer = other.optimizer;
            owns_optimizer = other.owns_optimizer;
            other.optimizer = nullptr;
            other.owns_optimizer = false;
        }
        return *this;
    }

    ~BaseLinearModel() override {
        if (owns_optimizer && optimizer) {
            delete optimizer;
        }
    }

    virtual double compute_regularization_cost(const Vec &w, size_t n_samples) const {
        double l2 = 0.0;
        for (const auto &v : w) l2 += v * v;
        double lambda = optimizer->get_l2_lambda();
        if (lambda == 0.0) return 0.0;
        return (lambda / (2.0 * static_cast<double>(n_samples))) * l2;
    }

private:
    virtual void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                                    const LinearParams &params, LinearParams &gradients) = 0;

public:
    void fit(const Matrix &X, const Vec &y) override {
        fit(X, y, 100, false);
    }

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

            double total_error = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = std::inner_product(X_scaled[i].begin(), X_scaled[i].end(), weights.begin(), 0.0) + bias;
                total_error += (pred - y[i]) * (pred - y[i]);
            }

            costs.push_back(
                (total_error / static_cast<double>(n_samples)) +
                compute_regularization_cost(weights, n_samples));
            if (detail::log_epoch_milestone(epoch, n_epochs, verbose)) {
                Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs),
                           Color::Yellow, " loss=", Color::White, costs.back());
            }
        }
        if (verbose && !costs.empty()) {
            Log::success("fit complete. final loss=", costs.back());
        }
    }

    Vec predict(const Matrix &X) override {
        Matrix X_scaled = X;
        if (X_scaled.empty() || X_scaled[0].size() != weights.size()) {
            throw std::invalid_argument("Input data for prediction is invalid or dimensions mismatch.");
        }

        X_scaled = scaler.transform(X_scaled);

        Vec predictions;
        predictions.reserve(X_scaled.size());
        for (const auto &sample: X_scaled) {
            predictions.push_back(std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias);
        }
        return predictions;
    }

    double score(const Vec &y_true, const Vec &y_pred) override {
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
