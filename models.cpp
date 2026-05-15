#include "models.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace cml {
namespace {

bool log_epoch_milestone(size_t epoch, size_t total, bool verbose) {
    if (!verbose) return false;
    const size_t step = std::max<size_t>(1, total / 10);
    return epoch % step == 0 || epoch == total - 1;
}

void validate_xy(const Matrix &X, const Vec &y, const char *where) {
    if (X.empty() || y.empty() || X.size() != y.size() || X[0].empty()) {
        throw std::invalid_argument(std::string(where) + ": empty data or row/label count mismatch");
    }
}

void validate_predict_matrix(const Matrix &X, size_t n_features) {
    if (X.empty() || X[0].size() != n_features) {
        throw std::invalid_argument("Input data for prediction is invalid or dimensions mismatch.");
    }
}

double regression_r2(const Vec &y_true, const Vec &y_pred) { return r2_score(y_true, y_pred); }

} // namespace

LinearRegression::LinearRegression(GradientDescent<LinearParams, Matrix>::Type type, double lr, size_t batch_size, double l2)
    : bias(0.0), owned_optimizer(std::make_unique<GradientDescent<LinearParams, Matrix>>(type, lr, batch_size, l2)), optimizer(owned_optimizer.get()) {}

LinearRegression::LinearRegression(GradientDescent<LinearParams, Matrix> *ext_opt)
    : bias(0.0), owned_optimizer(nullptr), optimizer(ext_opt) {
    if (!optimizer) throw std::invalid_argument("External optimizer must not be null");
}

void LinearRegression::docs() {
    Log::header("LinearRegression");
    Log::info("Fits y = Xw + b using gradient descent.");
    Log::info("Constructor: LinearRegression(grad_type, lr, batch_size, l2_lambda)");
    Log::info("fit(X, y, n_epochs, verbose), predict(X), score(y_true, y_pred) -> R2");
    Log::divider();
}

void LinearRegression::compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end, const LinearParams &params, LinearParams &gradients) {
    const size_t n_features = X[0].size();
    const size_t batch = static_cast<size_t>(end - start);
    gradients.weights.assign(n_features, 0.0);
    gradients.bias = 0.0;
    for (size_t i = 0; i < batch; ++i) {
        const size_t idx = start[i];
        const double pred = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.bias;
        const double err = pred - y[idx];
        gradients.bias += err;
        for (size_t j = 0; j < n_features; ++j) gradients.weights[j] += err * X[idx][j];
    }
}

void LinearRegression::fit(const Matrix &X, const Vec &y, size_t n_epochs, bool verbose) {
    validate_xy(X, y, "fit");
    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();
    weights.assign(n_features, 0.0);
    bias = 0.0;
    costs.clear();
    optimizer->set_n_samples(n_samples);
    LinearParams params{weights, bias};
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
        optimizer->shuffle_indices();
        auto grad_func = [this, &X, &y](const Matrix &, const size_t *s, const size_t *e, const LinearParams &p, LinearParams &g) { compute_gradients(X, y, s, e, p, g); };
        optimizer->step(params, X, grad_func, epoch);
        weights = params.weights; bias = params.bias;
        double total_error = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            const double pred = std::inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias;
            const double err = pred - y[i];
            total_error += err * err;
        }
        double l2_cost = 0.0;
        for (double w : weights) l2_cost += w * w;
        const double l2 = optimizer->get_l2_lambda();
        costs.push_back(total_error / static_cast<double>(n_samples) + (l2 / (2.0 * static_cast<double>(n_samples))) * l2_cost);
        if (log_epoch_milestone(epoch, n_epochs, verbose)) Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs), Color::Yellow, " loss=", Color::White, costs.back());
    }
    if (verbose && !costs.empty()) Log::success("LinearRegression fit complete. final loss=", costs.back());
}

Vec LinearRegression::predict(const Matrix &X) {
    validate_predict_matrix(X, weights.size());
    Vec predictions; predictions.reserve(X.size());
    for (const auto &sample : X) predictions.push_back(std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias);
    return predictions;
}

double LinearRegression::score(const Vec &y_true, const Vec &y_pred) { return regression_r2(y_true, y_pred); }

LogisticRegression::LogisticRegression(GradientDescent<LinearParams, Matrix>::Type type, double lr, size_t batch_size, double l2)
    : bias(0.0), owned_optimizer(std::make_unique<GradientDescent<LinearParams, Matrix>>(type, lr, batch_size, l2)), optimizer(owned_optimizer.get()) {}

LogisticRegression::LogisticRegression(GradientDescent<LinearParams, Matrix> *ext_opt)
    : bias(0.0), owned_optimizer(nullptr), optimizer(ext_opt) {
    if (!optimizer) throw std::invalid_argument("External optimizer must not be null");
}

void LogisticRegression::docs() {
    Log::header("LogisticRegression");
    Log::info("Binary classifier using sigmoid + gradient descent.");
    Log::info("Constructor: LogisticRegression(grad_type, lr, batch_size, l2_lambda)");
    Log::info("fit(X, y, n_epochs, verbose), predict(X), predict_prob(X), score(y_true, y_pred) -> accuracy");
    Log::divider();
}

void LogisticRegression::compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end, const LinearParams &params, LinearParams &gradients) {
    const size_t n_features = params.weights.size();
    const size_t batch = static_cast<size_t>(end - start);
    gradients.weights.assign(n_features, 0.0);
    gradients.bias = 0.0;
    for (size_t i = 0; i < batch; ++i) {
        const size_t idx = start[i];
        const double z = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.bias;
        const double err = sigmoid(z) - y[idx];
        gradients.bias += err;
        for (size_t j = 0; j < n_features; ++j) gradients.weights[j] += err * X[idx][j];
    }
}

void LogisticRegression::fit(const Matrix &X, const Vec &y, size_t n_epochs, bool verbose) {
    validate_xy(X, y, "fit");
    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();
    weights.assign(n_features, 0.0); bias = 0.0; costs.clear();
    optimizer->set_n_samples(n_samples);
    LinearParams params{weights, bias};
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
        optimizer->shuffle_indices();
        auto grad_func = [this, &X, &y](const Matrix &, const size_t *s, const size_t *e, const LinearParams &p, LinearParams &g) { compute_gradients(X, y, s, e, p, g); };
        optimizer->step(params, X, grad_func, epoch);
        weights = params.weights; bias = params.bias;
        double total_cost = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            const double z = std::inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias;
            const double pred = std::min(1.0 - 1e-15, std::max(1e-15, sigmoid(z)));
            total_cost += -y[i] * std::log(pred) - (1.0 - y[i]) * std::log(1.0 - pred);
        }
        double l2_cost = 0.0; for (double w : weights) l2_cost += w * w;
        const double l2 = optimizer->get_l2_lambda();
        costs.push_back(total_cost / static_cast<double>(n_samples) + (l2 / (2.0 * static_cast<double>(n_samples))) * l2_cost);
        if (log_epoch_milestone(epoch, n_epochs, verbose)) Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs), Color::Yellow, " loss=", Color::White, costs.back());
    }
    if (verbose && !costs.empty()) Log::success("LogisticRegression fit complete. final loss=", costs.back());
}

Vec LogisticRegression::predict_prob(const Matrix &X) {
    validate_predict_matrix(X, weights.size());
    Vec predictions; predictions.reserve(X.size());
    for (const auto &sample : X) predictions.push_back(sigmoid(std::inner_product(sample.begin(), sample.end(), weights.begin(), 0.0) + bias));
    return predictions;
}

Vec LogisticRegression::predict(const Matrix &X) {
    Vec p = predict_prob(X);
    for (double &v : p) v = (v >= 0.5) ? 1.0 : 0.0;
    return p;
}

double LogisticRegression::score(const Vec &y_true, const Vec &y_pred) { return accuracy(y_true, y_pred); }

RidgeRegression::RidgeRegression(GradientDescent<LinearParams, Matrix>::Type type, double lr, size_t batch_size, double l2)
    : bias(0.0), owned_optimizer(std::make_unique<GradientDescent<LinearParams, Matrix>>(type, lr, batch_size, l2)), optimizer(owned_optimizer.get()) {}
RidgeRegression::RidgeRegression(GradientDescent<LinearParams, Matrix> *ext_opt) : bias(0.0), owned_optimizer(nullptr), optimizer(ext_opt) { if (!optimizer) throw std::invalid_argument("External optimizer must not be null"); }
void RidgeRegression::docs() { Log::header("RidgeRegression"); Log::info("Linear regression with L2 penalty in the gradient."); Log::info("Constructor: RidgeRegression(grad_type, lr, batch_size, l2_lambda)"); Log::info("fit(X, y, n_epochs, verbose), predict(X), score(y_true, y_pred) -> R2"); Log::divider(); }
void RidgeRegression::compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end, const LinearParams &params, LinearParams &gradients) {
    const size_t n_features = X[0].size(); const size_t batch = static_cast<size_t>(end - start);
    gradients.weights.assign(n_features, 0.0); gradients.bias = 0.0;
    for (size_t i = 0; i < batch; ++i) {
        const size_t idx = start[i]; const double pred = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.bias; const double err = pred - y[idx];
        gradients.bias += err; for (size_t j = 0; j < n_features; ++j) gradients.weights[j] += err * X[idx][j];
    }
    for (size_t j = 0; j < n_features; ++j) gradients.weights[j] += optimizer->get_l2_lambda() * params.weights[j];
}
void RidgeRegression::fit(const Matrix &X, const Vec &y, size_t n_epochs, bool verbose) {
    validate_xy(X, y, "fit");
    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();
    weights.assign(n_features, 0.0);
    bias = 0.0;
    costs.clear();
    optimizer->set_n_samples(n_samples);
    LinearParams params{weights, bias};
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
        optimizer->shuffle_indices();
        auto grad_func = [this, &X, &y](const Matrix &, const size_t *s, const size_t *e, const LinearParams &p, LinearParams &g) {
            compute_gradients(X, y, s, e, p, g);
        };
        optimizer->step(params, X, grad_func, epoch);
        weights = params.weights;
        bias = params.bias;
        double total_error = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            const double pred = std::inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias;
            const double err = pred - y[i];
            total_error += err * err;
        }
        double l2_cost = 0.0;
        for (double w : weights) l2_cost += w * w;
        const double l2 = optimizer->get_l2_lambda();
        costs.push_back(total_error / static_cast<double>(n_samples) + (l2 / (2.0 * static_cast<double>(n_samples))) * l2_cost);
        if (log_epoch_milestone(epoch, n_epochs, verbose)) {
            Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs), Color::Yellow, " loss=", Color::White, costs.back());
        }
    }
    if (verbose && !costs.empty()) Log::success("RidgeRegression fit complete. final loss=", costs.back());
}
Vec RidgeRegression::predict(const Matrix &X) { validate_predict_matrix(X, weights.size()); Vec out; out.reserve(X.size()); for (const auto &s : X) out.push_back(std::inner_product(s.begin(), s.end(), weights.begin(), 0.0) + bias); return out; }
double RidgeRegression::score(const Vec &y_true, const Vec &y_pred) { return regression_r2(y_true, y_pred); }

LassoRegression::LassoRegression(GradientDescent<LinearParams, Matrix>::Type type, double lr, size_t batch_size, double l1_lambda)
    : bias(0.0), owned_optimizer(std::make_unique<GradientDescent<LinearParams, Matrix>>(type, lr, batch_size, l1_lambda)), optimizer(owned_optimizer.get()) {}
LassoRegression::LassoRegression(GradientDescent<LinearParams, Matrix> *ext_opt) : bias(0.0), owned_optimizer(nullptr), optimizer(ext_opt) { if (!optimizer) throw std::invalid_argument("External optimizer must not be null"); }
void LassoRegression::docs() { Log::header("LassoRegression"); Log::info("Linear regression with L1 penalty (subgradient) in the gradient."); Log::info("Constructor: LassoRegression(grad_type, lr, batch_size, l1_lambda)"); Log::info("fit(X, y, n_epochs, verbose), predict(X), score(y_true, y_pred) -> R2"); Log::divider(); }
void LassoRegression::compute_gradients(const Matrix &X, const Vec &y, const size_t *start, const size_t *end, const LinearParams &params, LinearParams &gradients) {
    const size_t n_features = X[0].size(); const size_t batch = static_cast<size_t>(end - start);
    gradients.weights.assign(n_features, 0.0); gradients.bias = 0.0;
    for (size_t i = 0; i < batch; ++i) {
        const size_t idx = start[i]; const double pred = std::inner_product(X[idx].begin(), X[idx].end(), params.weights.begin(), 0.0) + params.bias; const double err = pred - y[idx];
        gradients.bias += err; for (size_t j = 0; j < n_features; ++j) gradients.weights[j] += err * X[idx][j];
    }
    // For LassoRegression, GradientDescent's generic l2_lambda slot stores the L1 strength.
    const double l1 = optimizer->get_l2_lambda();
    for (size_t j = 0; j < n_features; ++j) { const double sign = (params.weights[j] > 0.0) ? 1.0 : (params.weights[j] < 0.0 ? -1.0 : 0.0); gradients.weights[j] += l1 * sign; }
}
void LassoRegression::fit(const Matrix &X, const Vec &y, size_t n_epochs, bool verbose) {
    validate_xy(X, y, "fit"); const size_t n_samples = X.size(); const size_t n_features = X[0].size();
    weights.assign(n_features, 0.0); bias = 0.0; costs.clear(); optimizer->set_n_samples(n_samples); LinearParams params{weights, bias};
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
        optimizer->shuffle_indices(); auto grad_func = [this, &X, &y](const Matrix &, const size_t *s, const size_t *e, const LinearParams &p, LinearParams &g) { compute_gradients(X, y, s, e, p, g); }; optimizer->step(params, X, grad_func, epoch); weights = params.weights; bias = params.bias;
        double total_error = 0.0; for (size_t i = 0; i < n_samples; ++i) { const double pred = std::inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias; const double err = pred - y[i]; total_error += err * err; }
        double l1_cost = 0.0; for (double w : weights) l1_cost += std::abs(w); const double l1 = optimizer->get_l2_lambda(); costs.push_back(total_error / static_cast<double>(n_samples) + (l1 / static_cast<double>(n_samples)) * l1_cost);
        if (log_epoch_milestone(epoch, n_epochs, verbose)) Log::epoch(static_cast<int>(epoch + 1), static_cast<int>(n_epochs), Color::Yellow, " loss=", Color::White, costs.back());
    } if (verbose && !costs.empty()) Log::success("LassoRegression fit complete. final loss=", costs.back());
}
Vec LassoRegression::predict(const Matrix &X) { validate_predict_matrix(X, weights.size()); Vec out; out.reserve(X.size()); for (const auto &s : X) out.push_back(std::inner_product(s.begin(), s.end(), weights.begin(), 0.0) + bias); return out; }
double LassoRegression::score(const Vec &y_true, const Vec &y_pred) { return regression_r2(y_true, y_pred); }

DecisionTree::Node::~Node() { delete left; delete right; }
DecisionTree::DecisionTree(int max_depth_, int min_samples_split_) : max_depth(max_depth_), min_samples_split(min_samples_split_), root(nullptr) {}
DecisionTree::~DecisionTree() { delete root; }
void DecisionTree::docs() { Log::header("DecisionTree"); Log::info("CART classifier using Gini impurity and recursive splits."); Log::info("Constructor: DecisionTree(max_depth, min_samples_split)"); Log::info("fit(X, y, verbose), predict(X), print_tree(), score(y_true, y_pred) -> accuracy"); Log::divider(); }
double DecisionTree::compute_gini(const std::vector<size_t> &indices, const Vec &y) { if (indices.empty()) return 0.0; std::map<int, int> counts; for (size_t idx : indices) counts[static_cast<int>(y[idx])]++; double gini = 1.0; for (const auto &kv : counts) { const double p = static_cast<double>(kv.second) / static_cast<double>(indices.size()); gini -= p * p; } return gini; }
int DecisionTree::majority_vote(const std::vector<size_t> &indices, const Vec &y) { std::map<int, int> counts; for (size_t idx : indices) counts[static_cast<int>(y[idx])]++; int best_count = -1, best_label = -1; for (const auto &kv : counts) if (kv.second > best_count) { best_count = kv.second; best_label = kv.first; } return best_label; }
DecisionTree::SplitResult DecisionTree::best_split(const Matrix &X, const Vec &y, const std::vector<size_t> &indices) {
    double best_ig = 0.0; SplitResult best; const double current_gini = compute_gini(indices, y);
    for (size_t j = 0; j < X[0].size(); ++j) {
        std::set<double> unique_values; for (size_t idx : indices) unique_values.insert(X[idx][j]);
        for (double threshold : unique_values) {
            std::vector<size_t> left, right; for (size_t idx : indices) (X[idx][j] <= threshold ? left : right).push_back(idx);
            if (left.empty() || right.empty()) continue;
            const double p_left = static_cast<double>(left.size()) / static_cast<double>(indices.size()); const double p_right = static_cast<double>(right.size()) / static_cast<double>(indices.size());
            const double ig = current_gini - (p_left * compute_gini(left, y) + p_right * compute_gini(right, y));
            if (ig > best_ig) { best_ig = ig; best = {static_cast<int>(j), threshold}; }
        }
    } return best;
}
DecisionTree::Node *DecisionTree::build_tree(const Matrix &X, const Vec &y, const std::vector<size_t> &indices, int depth) {
    if (depth >= max_depth || indices.size() < static_cast<size_t>(min_samples_split)) { auto *leaf = new Node(); leaf->label = majority_vote(indices, y); return leaf; }
    std::set<int> labels; for (size_t idx : indices) labels.insert(static_cast<int>(y[idx])); if (labels.size() == 1) { auto *leaf = new Node(); leaf->label = *labels.begin(); return leaf; }
    SplitResult split = best_split(X, y, indices); if (split.feature_idx == -1) { auto *leaf = new Node(); leaf->label = majority_vote(indices, y); return leaf; }
    std::vector<size_t> left, right; for (size_t idx : indices) (X[idx][static_cast<size_t>(split.feature_idx)] <= split.threshold ? left : right).push_back(idx);
    auto *node = new Node(); node->feature_idx = split.feature_idx; node->threshold = split.threshold; node->left = build_tree(X, y, left, depth + 1); node->right = build_tree(X, y, right, depth + 1); return node;
}
int DecisionTree::traverse(const Vec &sample, Node *node) const { if (!node) throw std::runtime_error("DecisionTree is not fitted"); if (node->is_leaf()) return node->label; return traverse(sample, sample[static_cast<size_t>(node->feature_idx)] <= node->threshold ? node->left : node->right); }
void DecisionTree::print_tree_helper(Node *node, int depth, const char *branch) const { if (!node) return; if (node->is_leaf()) Log::tree_node(depth, true, branch, "class=", node->label); else { Log::tree_node(depth, false, branch, "X[", node->feature_idx, "] <= ", node->threshold); print_tree_helper(node->left, depth + 1, "L: "); print_tree_helper(node->right, depth + 1, "R: "); } }
void DecisionTree::fit(const Matrix &X, const Vec &y, bool verbose) { validate_xy(X, y, "fit"); delete root; root = nullptr; std::vector<size_t> idx(X.size()); std::iota(idx.begin(), idx.end(), 0); root = build_tree(X, y, idx, 0); if (verbose) Log::success("DecisionTree fit complete."); }
Vec DecisionTree::predict(const Matrix &X) { if (X.empty()) throw std::invalid_argument("predict: empty data"); Vec out; out.reserve(X.size()); for (const auto &s : X) out.push_back(traverse(s, root)); return out; }
double DecisionTree::score(const Vec &y_true, const Vec &y_pred) { return accuracy(y_true, y_pred); }
void DecisionTree::print_tree() const { Log::header("DecisionTree"); print_tree_helper(root, 0, "Root: "); }

KNearestNeighbors::KNearestNeighbors(int k_, DistanceType distance_type_) : k(k_), distance_type(distance_type_) {}
void KNearestNeighbors::docs() { Log::header("KNearestNeighbors"); Log::info("Lazy learner: majority vote among k nearest neighbors."); Log::info("Constructor: KNearestNeighbors(k, distance_type)"); Log::info("fit(X, y, verbose), predict(X), score(y_true, y_pred) -> accuracy"); Log::divider(); }
void KNearestNeighbors::fit(const Matrix &X, const Vec &y, bool verbose) { validate_xy(X, y, "fit"); X_train = X; y_train = y; if (verbose) Log::success("KNearestNeighbors fit complete. stored ", X_train.size(), " samples."); }
double KNearestNeighbors::compute_distance(const std::vector<double> &a, const std::vector<double> &b) { if (distance_type == Manhattan) return static_cast<double>(manhattan_distance(a, b)); if (distance_type == Cosine) return static_cast<double>(cosine(a, b)); return static_cast<double>(euclidean_dist(a, b)); }
Vec KNearestNeighbors::predict(const Matrix &X) {
    if (X.empty() || X_train.empty()) throw std::invalid_argument("predict: empty data or unfitted model");
    const int effective_k = std::min(k, static_cast<int>(X_train.size())); if (k != effective_k) Log::warn("k=", k, " exceeded training set size; using k=", effective_k);
    Vec out; out.reserve(X.size());
    for (const auto &sample : X) {
        std::vector<std::pair<double, int>> distances; distances.reserve(X_train.size());
        for (size_t i = 0; i < X_train.size(); ++i) distances.emplace_back(compute_distance(sample, X_train[i]), static_cast<int>(y_train[i]));
        std::nth_element(distances.begin(), distances.begin() + effective_k - 1, distances.end());
        std::map<int, int> counts; for (int i = 0; i < effective_k; ++i) counts[distances[static_cast<size_t>(i)].second]++;
        int best_count = -1, best_label = -1; for (const auto &kv : counts) if (kv.second > best_count) { best_count = kv.second; best_label = kv.first; }
        out.push_back(best_label);
    } return out;
}
long double KNearestNeighbors::score(const Vec &y_true, const Vec &y_pred) { return accuracy(y_true, y_pred); }

KMeans::KMeans(int k_, int max_iter_, int n_restarts_, InitType init_type_, unsigned seed) : k(k_), max_iter(max_iter_), n_restarts(n_restarts_), init_type(init_type_), seed_(seed), best_inertia(std::numeric_limits<double>::max()) {}
void KMeans::docs() { Log::header("KMeans"); Log::info("Partitions data into k clusters via Lloyd's algorithm."); Log::info("Constructor: KMeans(k, max_iter, n_restarts, init_type, seed)"); Log::info("fit(X, verbose), predict(X), get_centroids(), get_inertia()"); Log::divider(); }
void KMeans::init_cluster_size() { clusters.assign(static_cast<size_t>(k), Matrix{}); for (auto &c : clusters) c.reserve(X_train.size()); }
Matrix KMeans::init_centroids(const Matrix &X, int restart_index) {
    std::mt19937 rng(seed_ + static_cast<unsigned>(restart_index)); std::uniform_int_distribution<int> dist(0, static_cast<int>(X.size()) - 1); Matrix cents;
    if (init_type == Random) { for (int i = 0; i < k; ++i) cents.push_back(X[static_cast<size_t>(dist(rng))]); return cents; }
    cents.push_back(X[static_cast<size_t>(dist(rng))]);
    for (int i = 1; i < k; ++i) { Vec distances(X.size()); for (size_t j = 0; j < X.size(); ++j) { double min_d = std::numeric_limits<double>::max(); for (const auto &c : cents) min_d = std::min(min_d, static_cast<double>(square_distance(X[j], c))); distances[j] = min_d; } std::discrete_distribution<int> weighted(distances.begin(), distances.end()); cents.push_back(X[static_cast<size_t>(weighted(rng))]); }
    return cents;
}
void KMeans::assign_clusters() { for (auto &c : clusters) c.clear(); for (const auto &s : X_train) { double best_d = -1.0; int best_i = 0; for (int i = 0; i < k; ++i) { const double d = static_cast<double>(square_distance(s, centroids[static_cast<size_t>(i)])); if (best_d < 0.0 || d < best_d) { best_d = d; best_i = i; } } clusters[static_cast<size_t>(best_i)].push_back(s); } }
void KMeans::update_centroids(int restart_index) { std::mt19937 rng(seed_ + static_cast<unsigned>(restart_index)); std::uniform_int_distribution<int> pick(0, static_cast<int>(X_train.size()) - 1); for (int i = 0; i < k; ++i) { auto &cluster = clusters[static_cast<size_t>(i)]; if (cluster.empty()) { centroids[static_cast<size_t>(i)] = X_train[static_cast<size_t>(pick(rng))]; continue; } Vec next(X_train[0].size(), 0.0); for (const auto &p : cluster) for (size_t j = 0; j < p.size(); ++j) next[j] += p[j]; for (double &v : next) v /= static_cast<double>(cluster.size()); centroids[static_cast<size_t>(i)] = next; } }
double KMeans::run_once(int restart_index) { centroids = init_centroids(X_train, restart_index); init_cluster_size(); for (int iter = 0; iter < max_iter; ++iter) { Matrix old = centroids; assign_clusters(); update_centroids(restart_index); bool converged = true; for (int i = 0; i < k; ++i) if (square_distance(centroids[static_cast<size_t>(i)], old[static_cast<size_t>(i)]) > 1e-6) { converged = false; break; } if (converged) break; } return score(X_train); }
void KMeans::fit(const Matrix &X, bool verbose) { if (X.empty() || X[0].empty()) throw std::invalid_argument("fit: empty data"); X_train = X; best_inertia = std::numeric_limits<double>::max(); Matrix best; for (int r = 0; r < n_restarts; ++r) { const double inertia = run_once(r); if (verbose) Log::info("Restart ", r + 1, "/", n_restarts, "  inertia=", inertia); if (inertia < best_inertia) { best_inertia = inertia; best = centroids; } } centroids = best; if (verbose) Log::success("KMeans fit complete. best inertia=", best_inertia); }
double KMeans::score(const Matrix &X) { double total = 0.0; for (const auto &p : X) { double min_d = -1.0; for (const auto &c : centroids) { const double d = static_cast<double>(square_distance(p, c)); if (min_d < 0.0 || d < min_d) min_d = d; } total += min_d; } return total; }
std::vector<int> KMeans::predict(const Matrix &X) { std::vector<int> labels; labels.reserve(X.size()); for (const auto &p : X) { int best = 0; double min_d = -1.0; for (int i = 0; i < k; ++i) { const double d = static_cast<double>(square_distance(p, centroids[static_cast<size_t>(i)])); if (min_d < 0.0 || d < min_d) { min_d = d; best = i; } } labels.push_back(best); } return labels; }
Matrix KMeans::get_centroids() const { return centroids; }
double KMeans::get_inertia() const { return best_inertia; }

void GaussianNB::docs() { Log::header("GaussianNB"); Log::info("Gaussian Naive Bayes classifier (per-class means/variances)."); Log::info("fit(X, y, verbose), predict(X), predict_prob(X), score(y_true, y_pred) -> accuracy"); Log::divider(); }
double GaussianNB::log_gaussian(double x, double mean, double variance) { const double var = std::max(variance, 1e-9); static const double pi = 3.14159265358979323846; return -0.5 * (std::log(2.0 * pi * var) + ((x - mean) * (x - mean)) / var); }
Matrix GaussianNB::give_means(const Matrix &X, const Vec &y, const std::vector<int> &cls) { Matrix all; for (int c : cls) { Vec m(X[0].size(), 0.0); int count = 0; for (size_t i = 0; i < X.size(); ++i) if (static_cast<int>(y[i]) == c) { for (size_t j = 0; j < X[0].size(); ++j) m[j] += X[i][j]; count++; } for (double &v : m) v /= static_cast<double>(count > 0 ? count : 1); all.push_back(m); } return all; }
Matrix GaussianNB::give_variances(const Matrix &X, const Vec &y, const std::vector<int> &cls, const Matrix &all_means) { Matrix all; for (size_t kidx = 0; kidx < cls.size(); ++kidx) { Vec v(X[0].size(), 0.0); int count = 0; for (size_t i = 0; i < X.size(); ++i) if (static_cast<int>(y[i]) == cls[kidx]) { for (size_t j = 0; j < X[0].size(); ++j) { const double d = X[i][j] - all_means[kidx][j]; v[j] += d * d; } count++; } for (double &x : v) x = x / static_cast<double>(count > 0 ? count : 1) + 1e-9; all.push_back(v); } return all; }
Vec GaussianNB::give_priors(const Vec &y) { Vec out(classes.size(), 0.0); for (size_t kidx = 0; kidx < classes.size(); ++kidx) { int count = 0; for (double label : y) if (static_cast<int>(label) == classes[kidx]) count++; out[kidx] = static_cast<double>(count) / static_cast<double>(y.size()); } return out; }
void GaussianNB::init() { means = give_means(X_train, y_train, classes); variances = give_variances(X_train, y_train, classes, means); priors = give_priors(y_train); }
void GaussianNB::extract_classes() { std::set<int> unique; for (double y : y_train) unique.insert(static_cast<int>(y)); classes.assign(unique.begin(), unique.end()); }
void GaussianNB::fit(const Matrix &X, const Vec &y, bool verbose) { validate_xy(X, y, "fit"); X_train = X; y_train = y; extract_classes(); init(); if (verbose) Log::success("GaussianNB fit complete. classes=", classes.size()); }
Vec GaussianNB::log_posteriors(const Vec &sample) { Vec scores; scores.reserve(classes.size()); for (size_t kidx = 0; kidx < classes.size(); ++kidx) { double s = std::log(priors[kidx]); for (size_t j = 0; j < sample.size(); ++j) s += log_gaussian(sample[j], means[kidx][j], variances[kidx][j]); scores.push_back(s); } return scores; }
Vec GaussianNB::predict(const Matrix &X) { Vec out; out.reserve(X.size()); for (const auto &sample : X) { Vec scores = log_posteriors(sample); const auto it = std::max_element(scores.begin(), scores.end()); out.push_back(classes[static_cast<size_t>(std::distance(scores.begin(), it))]); } return out; }
Matrix GaussianNB::predict_prob(const Matrix &X) { Matrix out; out.reserve(X.size()); for (const auto &sample : X) out.push_back(log_posteriors(sample)); return out; }
double GaussianNB::score(const Vec &y_true, const Vec &y_pred) { return accuracy(y_true, y_pred); }

SVM::SVM(double lr_, double lambda_, int n_epochs_) : lr(lr_), lambda(lambda_), n_epochs(n_epochs_), bias(0.0) {}
void SVM::docs() { Log::header("SVM"); Log::info("Linear SVM with hinge loss (SGD). Labels mapped to {-1,+1} internally."); Log::info("Constructor: SVM(lr, lambda, n_epochs)"); Log::info("fit(X, y, verbose), predict(X), score(y_true, y_pred) -> accuracy"); Log::divider(); }
Vec SVM::to_svm_labels(const Vec &y) { Vec out; out.reserve(y.size()); for (double label : y) out.push_back(label == 0.0 ? -1.0 : 1.0); return out; }
Vec SVM::from_svm_labels(const Vec &y) { Vec out; out.reserve(y.size()); for (double label : y) out.push_back(label == -1.0 ? 0.0 : 1.0); return out; }
void SVM::fit(const Matrix &X, const Vec &y, bool verbose) {
    validate_xy(X, y, "fit"); const size_t n_samples = X.size(); const size_t n_features = X[0].size(); weights.assign(n_features, 0.0); bias = 0.0; Vec y_svm = to_svm_labels(y);
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        for (size_t i = 0; i < n_samples; ++i) {
            const double condition = y_svm[i] * (std::inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias);
            if (condition >= 1.0) { for (size_t j = 0; j < n_features; ++j) weights[j] -= lr * (lambda * weights[j]); }
            else { for (size_t j = 0; j < n_features; ++j) weights[j] -= lr * (lambda * weights[j] - y_svm[i] * X[i][j]); bias += lr * y_svm[i]; }
        }
        if (log_epoch_milestone(static_cast<size_t>(epoch), static_cast<size_t>(n_epochs), verbose)) { Vec preds = predict(X); Log::epoch(epoch + 1, n_epochs, Color::Yellow, " acc=", Color::White, score(y, preds)); }
    }
    if (verbose) Log::success("SVM fit complete. train accuracy=", score(y, predict(X)));
}
Vec SVM::predict(const Matrix &X) { validate_predict_matrix(X, weights.size()); Vec raw; raw.reserve(X.size()); for (const auto &s : X) raw.push_back((std::inner_product(s.begin(), s.end(), weights.begin(), 0.0) + bias >= 0.0) ? 1.0 : -1.0); return from_svm_labels(raw); }
double SVM::score(const Vec &y_true, const Vec &y_pred) { return accuracy(y_true, y_pred); }

} // namespace cml
