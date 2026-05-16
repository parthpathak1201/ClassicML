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
            if (detail::log_epoch_milestone(static_cast<size_t>(epoch), static_cast<size_t>(n_epochs), verbose)) {
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
