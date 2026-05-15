#pragma once
#include "logger.hpp"
#include "metrics.hpp"
#include "optimizer.h"
#include "params.hpp"
#include "type.hpp"
#include "utils.h"
#include <memory>
#include <vector>

namespace cml {

class LinearRegression {
public:
    Vec weights;
    double bias;
    Vec costs;

    LinearRegression(GradientDescent<LinearParams, Matrix>::Type type = GradientDescent<LinearParams, Matrix>::Batch,
                     double lr = 0.01, size_t batch_size = 32, double l2 = 0.0);
    explicit LinearRegression(GradientDescent<LinearParams, Matrix> *ext_opt);
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false);
    Vec predict(const Matrix &X);
    double score(const Vec &y_true, const Vec &y_pred);
    static void docs();
private:
    std::unique_ptr<GradientDescent<LinearParams, Matrix>> owned_optimizer;
    GradientDescent<LinearParams, Matrix> *optimizer;
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients);
};

class LogisticRegression {
public:
    Vec weights;
    double bias;
    Vec costs;

    explicit LogisticRegression(GradientDescent<LinearParams, Matrix>::Type type = GradientDescent<LinearParams, Matrix>::Batch,
                                double lr = 0.01, size_t batch_size = 32, double l2 = 0.0);
    explicit LogisticRegression(GradientDescent<LinearParams, Matrix> *ext_opt);
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false);
    Vec predict_prob(const Matrix &X);
    Vec predict(const Matrix &X);
    double score(const Vec &y_true, const Vec &y_pred);
    static void docs();
private:
    std::unique_ptr<GradientDescent<LinearParams, Matrix>> owned_optimizer;
    GradientDescent<LinearParams, Matrix> *optimizer;
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients);
};

class RidgeRegression {
public:
    Vec weights;
    double bias;
    Vec costs;

    RidgeRegression(GradientDescent<LinearParams, Matrix>::Type type = GradientDescent<LinearParams, Matrix>::Batch,
                    double lr = 0.01, size_t batch_size = 32, double l2 = 0.0);
    explicit RidgeRegression(GradientDescent<LinearParams, Matrix> *ext_opt);
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false);
    Vec predict(const Matrix &X);
    double score(const Vec &y_true, const Vec &y_pred);
    static void docs();
private:
    std::unique_ptr<GradientDescent<LinearParams, Matrix>> owned_optimizer;
    GradientDescent<LinearParams, Matrix> *optimizer;
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients);
};

class LassoRegression {
public:
    Vec weights;
    double bias;
    Vec costs;

    LassoRegression(GradientDescent<LinearParams, Matrix>::Type type = GradientDescent<LinearParams, Matrix>::Batch,
                    double lr = 0.01, size_t batch_size = 32, double l1_lambda = 0.0);
    explicit LassoRegression(GradientDescent<LinearParams, Matrix> *ext_opt);
    void fit(const Matrix &X, const Vec &y, size_t n_epochs = 100, bool verbose = false);
    Vec predict(const Matrix &X);
    double score(const Vec &y_true, const Vec &y_pred);
    static void docs();
private:
    std::unique_ptr<GradientDescent<LinearParams, Matrix>> owned_optimizer;
    GradientDescent<LinearParams, Matrix> *optimizer;
    void compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end,
                           const LinearParams &params, LinearParams &gradients);
};

class DecisionTree {
public:
    struct Node {
        int feature_idx = -1;
        double threshold = 0.0;
        Node *left = nullptr;
        Node *right = nullptr;
        int label = -1;
        [[nodiscard]] bool is_leaf() const { return feature_idx == -1; }
        ~Node();
    };

    DecisionTree(int max_depth = 10, int min_samples_split = 2);
    ~DecisionTree();
    void fit(const Matrix &X, const Vec &y, bool verbose = false);
    Vec predict(const Matrix &X);
    double score(const Vec &y_true, const Vec &y_pred);
    void print_tree() const;
    static void docs();
private:
    struct SplitResult { int feature_idx = -1; double threshold = 0.0; };
    int max_depth;
    int min_samples_split;
    Node *root;
    double compute_gini(const std::vector<size_t> &indices, const Vec &y);
    int majority_vote(const std::vector<size_t> &indices, const Vec &y);
    SplitResult best_split(const Matrix &X, const Vec &y, const std::vector<size_t> &indices);
    Node *build_tree(const Matrix &X, const Vec &y, const std::vector<size_t> &indices, int depth);
    int traverse(const Vec &sample, Node *node) const;
    void print_tree_helper(Node *node, int depth, const char *branch) const;
};

class KNearestNeighbors {
public:
    enum DistanceType { Manhattan, Euclidean, Cosine };
    int k;
    DistanceType distance_type;
    Matrix X_train;
    Vec y_train;

    KNearestNeighbors(int k = 5, DistanceType distance_type = Euclidean);
    void fit(const Matrix &X, const Vec &y, bool verbose = false);
    double compute_distance(const std::vector<double> &a, const std::vector<double> &b);
    Vec predict(const Matrix &X);
    long double score(const Vec &y_true, const Vec &y_pred);
    static void docs();
};

class KMeans {
public:
    enum InitType { Random, KMeansPP };
    KMeans(int k = 3, int max_iter = 300, int n_restarts = 10, InitType init_type = KMeansPP, unsigned seed = 42);
    void fit(const Matrix &X, bool verbose = false);
    double score(const Matrix &X);
    std::vector<int> predict(const Matrix &X);
    Matrix get_centroids() const;
    double get_inertia() const;
    static void docs();
private:
    int k;
    int max_iter;
    int n_restarts;
    InitType init_type;
    unsigned seed_;
    double best_inertia;
    Matrix X_train;
    Matrix centroids;
    std::vector<Matrix> clusters;
    void init_cluster_size();
    Matrix init_centroids(const Matrix &X, int restart_index);
    void assign_clusters();
    void update_centroids(int restart_index);
    double run_once(int restart_index);
};

class GaussianNB {
public:
    GaussianNB() = default;
    void fit(const Matrix &X, const Vec &y, bool verbose = false);
    Vec predict(const Matrix &X);
    Matrix predict_prob(const Matrix &X);
    double score(const Vec &y_true, const Vec &y_pred);
    static void docs();
private:
    Matrix means;
    Matrix variances;
    Vec priors;
    Matrix X_train;
    Vec y_train;
    std::vector<int> classes;
    double log_gaussian(double x, double mean, double variance);
    Matrix give_means(const Matrix &X, const Vec &y, const std::vector<int> &cls);
    Matrix give_variances(const Matrix &X, const Vec &y, const std::vector<int> &cls, const Matrix &all_means);
    Vec give_priors(const Vec &y);
    void init();
    void extract_classes();
    Vec log_posteriors(const Vec &sample);
};

class SVM {
public:
    SVM(double lr = 0.001, double lambda = 0.01, int n_epochs = 1000);
    void fit(const Matrix &X, const Vec &y, bool verbose = false);
    Vec predict(const Matrix &X);
    double score(const Vec &y_true, const Vec &y_pred);
    static void docs();
private:
    double lr;
    double lambda;
    int n_epochs;
    Vec weights;
    double bias;
    Vec to_svm_labels(const Vec &y);
    Vec from_svm_labels(const Vec &y);
};

} // namespace cml
