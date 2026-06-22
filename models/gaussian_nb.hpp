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
#include <map>
#include <set>
#include <random>
#include <limits>
#include <string>

// GaussianNB
// Naive Bayes classifier assuming Gaussian features with per-class means/variances.
// Usage:
//   GaussianNB gnb;
//   gnb.fit(X_train, y_train);
//   Vec preds = gnb.predict(X_test);
class [[maybe_unused]] GaussianNB : public Estimator {
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
    void fit(const Matrix &X, const Vec &y) override {
        fit(X, y, false);
    }

    void fit(const Matrix &X, const Vec &y, bool verbose = false) {
        X_train = X;
        y_train = y;
        extract_classes();
        init();
        if (verbose) {
            Log::success("GaussianNB fit complete. classes=", classes.size());
        }
    }

    Vec predict(const Matrix &X) override {
        if (classes.empty()) {
            throw std::invalid_argument("GaussianNB has not been fitted yet. Call fit() first.");
        }

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
        if (classes.empty()) {
            throw std::invalid_argument("GaussianNB has not been fitted yet. Call fit() first.");
        }

        Matrix all_scores;

        for (const auto &sample: X) {
            all_scores.push_back(log_posteriors(sample));
        }

        return all_scores;
    }

    double score(const Vec &y_true, const Vec &y_pred) override {
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
