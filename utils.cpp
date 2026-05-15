#include "utils.h"
#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cml {

long double pwr(long double x, int n) {
    if (n < 0) return 1.0L / pwr(x, -n);
    long double res = 1.0L;
    while (n > 0) {
        if (n & 1) res *= x;
        x *= x;
        n >>= 1;
    }
    return res;
}

long double pwr(long double x, long double n) { return std::exp(n * std::log(x)); }

long double custom_exp(long double x) {
    long double sum = 1.0L, term = 1.0L;
    for (int n = 1; n < 100; ++n) {
        term *= x / n;
        sum += term;
    }
    return sum;
}

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

long double dot(const std::vector<long double> &a, const std::vector<long double> &b) {
    long double result = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

long double manhattan_distance(const std::vector<double> &a, const std::vector<double> &b) {
    long double distance = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) distance += std::abs(a[i] - b[i]);
    return distance;
}

long double euclidean_dist(const std::vector<double> &a, const std::vector<double> &b) {
    long double distance = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        distance += d * d;
    }
    return std::sqrt(distance);
}

long double square_distance(const std::vector<double> &a, const std::vector<double> &b) {
    long double distance = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        distance += d * d;
    }
    return distance;
}

long double cosine(const std::vector<double> &a, const std::vector<double> &b) {
    long double dot_ab = 0.0L, norm_a_sq = 0.0L, norm_b_sq = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_ab += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    const long double norm_a = std::sqrt(norm_a_sq);
    const long double norm_b = std::sqrt(norm_b_sq);
    return (norm_a == 0.0L || norm_b == 0.0L) ? 0.0L : dot_ab / (norm_a * norm_b);
}

std::vector<long double> vec_mean(const std::vector<std::vector<long double>> &data) {
    std::vector<long double> mean(data[0].size(), 0.0L);
    for (const auto &r : data) for (size_t i = 0; i < r.size(); ++i) mean[i] += r[i];
    for (auto &v : mean) v /= data.size();
    return mean;
}

std::vector<long double> vec_std(const std::vector<std::vector<long double>> &data, const std::vector<long double> &mean) {
    std::vector<long double> stdv(data[0].size(), 0.0L);
    for (const auto &r : data) {
        for (size_t i = 0; i < r.size(); ++i) {
            const long double d = r[i] - mean[i];
            stdv[i] += d * d;
        }
    }
    for (auto &v : stdv) v = std::sqrt(v / data.size());
    return stdv;
}

std::vector<std::vector<long double>> norm(const std::vector<std::vector<long double>> &data) {
    const auto mean = vec_mean(data);
    const auto stdv = vec_std(data, mean);
    std::vector<std::vector<long double>> normalized(data.size(), std::vector<long double>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            normalized[i][j] = (stdv[j] == 0.0L) ? 0.0L : (data[i][j] - mean[j]) / stdv[j];
        }
    }
    return normalized;
}

std::vector<std::vector<long double>> transpose(const std::vector<std::vector<long double>> &data) {
    std::vector<std::vector<long double>> out(data[0].size(), std::vector<long double>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) for (size_t j = 0; j < data[i].size(); ++j) out[j][i] = data[i][j];
    return out;
}

long double MSE(const std::vector<long double> &y_true, const std::vector<long double> &y_pred) {
    long double mse = 0.0L;
    for (size_t i = 0; i < y_true.size(); ++i) { const long double d = y_true[i] - y_pred[i]; mse += d * d; }
    return mse / y_true.size();
}

long double MAE(const std::vector<long double> &y_true, const std::vector<long double> &y_pred) {
    long double total = 0.0L;
    for (size_t i = 0; i < y_true.size(); ++i) total += std::abs(y_true[i] - y_pred[i]);
    return total / y_true.size();
}

long double R2(const std::vector<long double> &y_true, const std::vector<long double> &y_pred) {
    const long double mean = vec_mean({y_true})[0];
    long double ss_res = 0.0L, ss_tot = 0.0L;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const long double res = y_true[i] - y_pred[i];
        const long double tot = y_true[i] - mean;
        ss_res += res * res;
        ss_tot += tot * tot;
    }
    return 1.0L - ss_res / ss_tot;
}

std::vector<std::vector<long double>> matmul(const std::vector<std::vector<long double>> &A, const std::vector<std::vector<long double>> &B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) throw std::invalid_argument("Dimension mismatch");
    std::vector<std::vector<long double>> C(A.size(), std::vector<long double>(B[0].size(), 0.0L));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t k = 0; k < A[0].size(); ++k)
            for (size_t j = 0; j < B[0].size(); ++j) C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<long double>> add(const std::vector<std::vector<long double>> &A, const std::vector<std::vector<long double>> &B) {
    std::vector<std::vector<long double>> result(A.size(), std::vector<long double>(A[0].size(), 0.0L));
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) result[i][j] = A[i][j] + B[i][j];
    return result;
}

std::vector<std::vector<long double>> subtract(const std::vector<std::vector<long double>> &A, const std::vector<std::vector<long double>> &B) {
    std::vector<std::vector<long double>> result(A.size(), std::vector<long double>(A[0].size(), 0.0L));
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) result[i][j] = A[i][j] - B[i][j];
    return result;
}

std::vector<std::vector<long double>> matinv(std::vector<std::vector<long double>> A) {
    const int n = static_cast<int>(A.size());
    std::vector<std::vector<long double>> result(n, std::vector<long double>(n, 0.0L));
    for (int i = 0; i < n; ++i) result[i][i] = 1.0L;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const long double factor = A[j][i] / A[i][i];
            for (int k = 0; k < n; ++k) { A[j][k] -= factor * A[i][k]; result[j][k] -= factor * result[i][k]; }
        }
    }
    for (int i = n - 1; i > 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            const long double factor = A[j][i] / A[i][i];
            for (int k = 0; k < n; ++k) { A[j][k] -= factor * A[i][k]; result[j][k] -= factor * result[i][k]; }
        }
    }
    for (int i = 0; i < n; ++i) {
        const long double factor = A[i][i];
        for (int k = 0; k < n; ++k) { A[i][k] /= factor; result[i][k] /= factor; }
    }
    return result;
}

long double variance(const std::vector<long double> &data) {
    const long double mean = vec_mean({data})[0];
    long double var = 0.0L;
    for (long double v : data) { const long double d = v - mean; var += d * d; }
    return var / data.size();
}

long double covariance(const std::vector<long double> &x, const std::vector<long double> &y) {
    const long double mean_x = vec_mean({x})[0];
    const long double mean_y = vec_mean({y})[0];
    long double cov = 0.0L;
    for (size_t i = 0; i < x.size(); ++i) cov += (x[i] - mean_x) * (y[i] - mean_y);
    return cov / x.size();
}

std::vector<long double> row(const std::vector<std::vector<long double>> &data, int row_index) { return data[static_cast<size_t>(row_index)]; }

std::vector<long double> col(const std::vector<std::vector<long double>> &data, int col_index) {
    std::vector<long double> column(data.size());
    for (size_t i = 0; i < data.size(); ++i) column[i] = data[i][static_cast<size_t>(col_index)];
    return column;
}

long long factorial(int n) {
    if (n == 0 || n == 1) return 1;
    long long result = 1;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

long long ncr(int n, int r) {
    if (r > n) return 0;
    if (r == 0 || r == n) return 1;
    return factorial(n) / (factorial(r) * factorial(n - r));
}

void help() {
    Log::header("ClassicML Utils");
    Log::divider();
    Log::info("pwr, custom_exp, sigmoid, dot, manhattan_distance, euclidean_dist");
    Log::info("cosine, square_distance, variance, covariance");
    Log::info("vec_mean, vec_std, norm, transpose, matmul, matinv");
    Log::info("add, subtract, MSE, MAE, R2, factorial, ncr, row, col");
    Log::info("Call docs(\"name\") for details on a specific function.");
    Log::divider();
}

void docs(const std::string &func_name) {
    static const std::map<std::string, std::string> kDocs = {
        {"matmul", "matmul(A,B) -> C\n  Matrix multiply. C[i][j] = sum_k A[i][k]*B[k][j]."},
        {"matinv", "matinv(A) -> inverse(A)\n  Gauss-Jordan inverse. Square matrices only."},
        {"sigmoid", "sigmoid(z) -> (0,1)\n  1/(1+exp(-z)) using std::exp."},
        {"dot", "dot(a,b) -> scalar\n  Inner product of equal-length vectors."},
        {"euclidean_dist", "euclidean_dist(a,b) -> L2 distance\n  sqrt(sum (a_i-b_i)^2)."},
        {"manhattan_distance", "manhattan_distance(a,b) -> L1 distance\n  sum abs(a_i-b_i)."},
        {"cosine", "cosine(a,b) -> [-1,1]\n  Cosine similarity (not distance)."},
        {"pwr", "pwr(x,n) -> x^n\n  Fast exponentiation for integer exponents."},
        {"variance", "variance(data) -> population variance."},
        {"covariance", "covariance(x,y) -> population covariance."},
        {"transpose", "transpose(M) -> M^T\n  Swaps rows and columns."}
    };
    const auto it = kDocs.find(func_name);
    if (it == kDocs.end()) {
        Log::warn("Unknown function. Call help() for list.");
        return;
    }
    Log::header(func_name);
    Log::info(it->second);
    Log::divider();
}

} // namespace cml
