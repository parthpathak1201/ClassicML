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
