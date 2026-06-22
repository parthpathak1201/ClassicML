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

// KNearestNeighbors
// Lazy learner: stores training data, classifies by majority vote among k nearest neighbors.
// Usage:
//   KNearestNeighbors knn(5, KNearestNeighbors::Euclidean);
//   knn.fit(X_train, y_train);
//   Vec preds = knn.predict(X_test);
class [[maybe_unused]] KNearestNeighbors : public Estimator {
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

    void fit(const Matrix &X, const Vec &y) override {
        fit(X, y, false);
    }

    int get_k() const { return k; }
    void set_k(int k_) { k = k_; }
    DistanceType get_distance_type() const { return distance_type; }
    void set_distance_type(DistanceType dt) { distance_type = dt; }

    void fit(const Matrix &X, const Vec &y, bool verbose = false) {
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
            return 1.0 - cosine(a, b);
        }

        return euclidean_dist(a, b); //default
    }

    Vec predict(const Matrix &X_) override {
        // Clamp k to training set size if user passed k > n_train
        if (X_train.empty()) {
            throw std::invalid_argument("KNearestNeighbors: no training data, call fit() first");
        }

        int kk = std::min(k, static_cast<int>(X_train.size()));
        if (kk <= 0) {
            throw std::invalid_argument("KNearestNeighbors: invalid k value");
        }

        Matrix X = scaler.transform(X_);
        Vec predictions;
        predictions.reserve(X.size());

        for (const auto &sample: X) {
            std::vector<std::pair<double, int> > distances;
            distances.reserve(X_train.size());
            for (size_t i = 0; i < X_train.size(); ++i) {
                double dist = compute_distance(sample, X_train[i]);
                distances.emplace_back(dist, static_cast<int>(y_train[i]));
            }

            size_t k_sz = static_cast<size_t>(kk);
            if (k_sz < distances.size()) {
                std::nth_element(distances.begin(), distances.begin() + (k_sz - 1), distances.end());
            }

            std::map<int, int> class_counts;
            for (size_t i = 0; i < k_sz; ++i) {
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

    double score(const Vec &y_true, const Vec &y_pred) override {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("Dimension mismatch between true and predicted values.");
        }
        double correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) {
                correct++;
            }
        }
        return correct / static_cast<double>(y_true.size());
    }

private:
    int k;
    DistanceType distance_type;
    StandardScaler scaler;
    Matrix X_train;
    Vec y_train;
};
