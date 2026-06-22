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
#include <memory>

// DecisionTree
// CART-style classifier using Gini impurity and recursive binary splits.
// Usage:
//   DecisionTree dt(10, 2);
//   dt.fit(X_train, y_train);
//   Vec preds = dt.predict(X_test);
class [[maybe_unused]] DecisionTree : public Estimator {
public:
    struct Node {
        int feature_idx = -1;
        double threshold = 0.0;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        int label = -1;

        [[nodiscard]] bool is_leaf() const { return feature_idx == -1; }
    };

private:
    struct SplitResult {
        int feature_idx = -1;
        double threshold = 0.0;
    };

    int max_depth;
    int min_samples_split;
    std::unique_ptr<Node> root;
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

    std::unique_ptr<Node> build_tree(const Matrix &X, const Vec &y, const std::vector<size_t> &indices, int depth) {
        if (depth >= max_depth || indices.size() < min_samples_split) {
            auto leaf = std::make_unique<Node>();
            leaf->label = majority_vote(indices, y);
            return leaf;
        }

        std::set<int> labels;
        for (size_t idx: indices) {
            labels.insert(static_cast<int>(y[idx]));
        }
        if (labels.size() == 1) {
            auto leaf = std::make_unique<Node>();
            leaf->label = *labels.begin();
            return leaf;
        }

        SplitResult split = best_split(X, y, indices);

        if (split.feature_idx == -1) {
            auto leaf = std::make_unique<Node>();
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

        auto node = std::make_unique<Node>();
        node->feature_idx = split.feature_idx;
        node->threshold = split.threshold;
        node->left = build_tree(X, y, left_indices, depth + 1);
        node->right = build_tree(X, y, right_indices, depth + 1);
        return node;
    }

    int traverse(const Vec &sample, Node *node) const {
        if (!node) {
            throw std::invalid_argument("DecisionTree traverse reached null node");
        }
        if (node->is_leaf()) {
            return node->label;
        }
        if (node->feature_idx < 0 || static_cast<size_t>(node->feature_idx) >= sample.size()) {
            throw std::out_of_range("DecisionTree feature index out of range in traverse");
        }
        if (sample[node->feature_idx] <= node->threshold) {
            return traverse(sample, node->left.get());
        } else {
            return traverse(sample, node->right.get());
        }
    }

    void print_tree_helper(Node *node, int depth, const char *branch) const {
        if (!node) return;
        if (node->is_leaf()) {
            Log::tree_node(depth, true, branch, "class=", node->label, " ", Color::Green, "✓");
        } else {
            Log::tree_node(depth, false, branch, "X[", node->feature_idx, "] <= ", node->threshold);
            print_tree_helper(node->left.get(), depth + 1, "├── ");
            print_tree_helper(node->right.get(), depth + 1, "└── ");
        }
    }

public:
    DecisionTree(int max_depth = 10, int min_samples_split = 2)
        : max_depth(max_depth), min_samples_split(min_samples_split) {
    }

    DecisionTree(const DecisionTree &) = delete;
    DecisionTree &operator=(const DecisionTree &) = delete;
    DecisionTree(DecisionTree &&) = default;
    DecisionTree &operator=(DecisionTree &&) = default;

    void fit(const Matrix &X_, const Vec &y) override {
        fit(X_, y, false);
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
        Matrix X = X_;
        scaler.fit(X);
        X = scaler.transform(X);
        root.reset();
        std::vector<size_t> initial_indices(X.size());
        std::iota(initial_indices.begin(), initial_indices.end(), 0);
        root = build_tree(X, y, initial_indices, 0);
        if (verbose) {
            Log::success("DecisionTree fit complete.");
        }
    }

    Vec predict(const Matrix &X_) override {
        if (!root) {
            throw std::invalid_argument("DecisionTree has not been fitted yet. Call fit() first.");
        }
        Matrix X = X_;
        X = scaler.transform(X);
        Vec predictions;
        predictions.reserve(X.size());
        for (const auto &sample: X) {
            predictions.push_back(traverse(sample, root.get()));
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
        return correct / y_true.size();
    }

    void print_tree() const {
        Log::header("DecisionTree");
        print_tree_helper(root.get(), 0, "└── ");
    }
};
