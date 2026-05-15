#include "pre.h"
#include "utils.h"
#include "logger.hpp"
#include "type.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Fast integer exponentiation via repeated squaring — O(log n)
inline long double pwr(long double x, int n) {
    long double res = 1;
    while (n > 0) {
        if (n & 1)
            res *= x;
        x *= x;
        n >>= 1;
    }
    return res;
}

// Fast exponentiation for real powers
inline long double pwr(long double x, long double n) { return std::exp(n * std::log(x)); }

// Taylor series approximation of e^x — 100 terms, sufficient for ML use
long double custom_exp(long double x) {
    long double sum = 1.0L, term = 1.0L;
    for (int n = 1; n < 100; ++n) {
        term *= x / n;
        sum += term;
    }
    return sum;
}

// factorial
inline long long factorial(int n) {
    if (n == 0 || n == 1)
        return 1;
    long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// ncr
inline long long ncr(int n, int r) {
    if (r > n)
        return 0;
    if (r == 0 || r == n)
        return 1;
    return factorial(n) / (factorial(r) * factorial(n - r));
}

// sigmoid function
inline long double sigmoid(long double z) { return 1.0L / (1.0L + custom_exp(-z)); }

// dot product of two vectors
inline long double dot(const std::vector<long double> &a,
                       const std::vector<long double> &b) {
    long double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Manhattan distance between two vectors
inline long double manhattan_distance(const std::vector<double> &a,
                                      const std::vector<double> &b) {
    long double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += std::abs(a[i] - b[i]);
    }
    return distance;
}

// Euclidean distance between two vectors
inline long double euclidean_dist(const std::vector<double> &a,
                                  const std::vector<double> &b) {
    long double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += pwr(a[i] - b[i], 2);
    }
    return std::sqrt(distance);
}

inline long double square_distance(const std::vector<double> &a,
                                   const std::vector<double> &b) {
    long double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += pwr(a[i] - b[i], 2);
    }
    return distance;
}

// Cosine similarity (not distance) — returns value in [-1, 1]
inline long double cosine(const std::vector<double> &a, const std::vector<double> &b) {
    long double dot_ab = 0.0;
    long double norm_a_sq = 0.0;
    long double norm_b_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_ab += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    long double norm_a = std::sqrt(norm_a_sq);
    long double norm_b = std::sqrt(norm_b_sq);
    if (norm_a == 0 || norm_b == 0)
        return 0.0;
    return dot_ab / (norm_a * norm_b);
}

// Mean of a matrix
inline std::vector<long double> vec_mean(const std::vector<std::vector<long double> > &data) {
    std::vector<long double> mean(data[0].size(), 0.0);
    for (const auto &row: data) {
        for (size_t i = 0; i < row.size(); ++i) {
            mean[i] += row[i];
        }
    }
    for (auto &val: mean) {
        val /= data.size();
    }
    return mean;
}

// Standard deviation of a matrix
inline std::vector<long double> vec_std(const std::vector<std::vector<long double> > &data,
                                   const std::vector<long double> &mean) {
    std::vector<long double> std(data[0].size(), 0.0);
    for (const auto &row: data) {
        for (size_t i = 0; i < row.size(); ++i) {
            std[i] += pwr(row[i] - mean[i], 2);
        }
    }
    for (auto &val: std) {
        val = std::sqrt(val / data.size());
    }
    return std;
}

// Normalize a matrix and return a copy (does not modify the original matrix)
inline std::vector<std::vector<long double> >
norm(const std::vector<std::vector<long double> > &data) {
    std::vector<long double> mean = vec_mean(data);
    std::vector<long double> std = vec_std(data, mean);
    std::vector<std::vector<long double> > normalized(data.size(),
                                            std::vector<long double>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            normalized[i][j] = (data[i][j] - mean[j]) / std[j];
        }
    }
    return normalized;
}

// Transpose of a matrix, new mat.
inline std::vector<std::vector<long double> >
transpose(const std::vector<std::vector<long double> > &data) {
    std::vector<std::vector<long double> > transposed(data[0].size(),
                                            std::vector<long double>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            transposed[j][i] = data[i][j];
        }
    }
    return transposed;
}

// Mean Squared Error
inline long double MSE(const std::vector<long double> &y_true,
                       const std::vector<long double> &y_pred) {
    long double mse = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        mse += pwr(y_true[i] - y_pred[i], 2);
    }
    return mse / y_true.size();
}

// Mean Absolute Error
inline long double MAE(const std::vector<long double> &y_true,
                       const std::vector<long double> &y_pred) {
    long double mae = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        mae += std::abs(y_true[i] - y_pred[i]);
    }
    return mae / y_true.size();
}

// R-squared
inline long double R2(const std::vector<long double> &y_true,
                      const std::vector<long double> &y_pred) {
    long double ss_res = 0.0, ss_tot = 0.0;
    long double mean = vec_mean({y_true})[0];
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_res += pwr(y_true[i] - y_pred[i], 2);
        ss_tot += pwr(y_true[i] - mean, 2);
    }
    return 1 - (ss_res / ss_tot);
}

// Matmul
inline std::vector<std::vector<long double> >
matmul(const std::vector<std::vector<long double> > &A,
       const std::vector<std::vector<long double> > &B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    size_t n = A.size();
    size_t m = A[0].size();
    size_t p = B[0].size();

    std::vector<std::vector<long double> > C(n, std::vector<long double>(p, 0));

    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < m; ++k)
            for (size_t j = 0; j < p; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// addition and subtraction of matrices
inline std::vector<std::vector<long double> > add(const std::vector<std::vector<long double> > &A,
                                        const std::vector<std::vector<long double> > &B) {
    std::vector<std::vector<long double> > result(A.size(),
                                        std::vector<long double>(A[0].size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

inline std::vector<std::vector<long double> >
subtract(const std::vector<std::vector<long double> > &A,
         const std::vector<std::vector<long double> > &B) {
    std::vector<std::vector<long double> > result(A.size(),
                                        std::vector<long double>(A[0].size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

// Gauss-Jordan elimination to compute matrix inverse in-place
std::vector<std::vector<long double> > matinv(std::vector<std::vector<long double> > A) {
    int n = static_cast<int>(A.size());
    std::vector<std::vector<long double> > result(n, std::vector<long double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        result[i][i] = 1.0;
    }

    // step 1 - eliminate the lower half

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            long double factor = A[j][i] / A[i][i];
            for (int k = 0; k < n; k++) {
                A[j][k] -= factor * A[i][k];
                result[j][k] -= factor * result[i][k];
            }
        }
    }

    // step 2 - eliminate the upper half
    for (int i = n - 1; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            long double factor = A[j][i] / A[i][i];
            for (int k = 0; k < n; k++) {
                A[j][k] -= factor * A[i][k];
                result[j][k] -= factor * result[i][k];
            }
        }
    }

    // step 3 - fix the diagonal
    for (int i = 0; i < n; i++) {
        long double factor = A[i][i];
        for (int k = 0; k < n; k++) {
            A[i][k] /= factor;
            result[i][k] /= factor;
        }
    }

    return result;
}

// Variance
inline long double variance(const std::vector<long double> &data) {
    long double mean = vec_mean({data})[0];
    long double var = 0.0;
    for (const auto &val: data) {
        var += pwr(val - mean, 2);
    }
    return var / data.size();
}

// Covariance
inline long double covariance(const std::vector<long double> &x,
                              const std::vector<long double> &y) {
    long double mean_x = vec_mean({x})[0];
    long double mean_y = vec_mean({y})[0];
    long double cov = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return cov / x.size();
}

// Fetch a row or column from a matrix
inline std::vector<long double> row(const std::vector<std::vector<long double> > &data,
                               int row_index) {
    return data[row_index];
}

inline std::vector<long double> col(const std::vector<std::vector<long double> > &data,
                               int col_index) {
    std::vector<long double> column(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        column[i] = data[i][col_index];
    }
    return column;
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
        {"matmul",
         "matmul(A,B) → C\n  Matrix multiply. C[i][j] = sum_k A[i][k]*B[k][j].\n  Complexity: O(n*m*p). Example: auto C = matmul(A, B);"},
        {"matinv",
         "matinv(A) → A⁻¹\n  Gauss-Jordan inverse. Square matrices only.\n  Complexity: O(n³). Example: auto inv = matinv(A);"},
        {"sigmoid",
         "sigmoid(z) → (0,1)\n  1/(1+e^{-z}) using custom_exp.\n  Complexity: O(1). Example: auto p = sigmoid(z);"},
        {"dot",
         "dot(a,b) → scalar\n  Inner product of equal-length vectors.\n  Complexity: O(n). Example: auto s = dot(a, b);"},
        {"euclidean_dist",
         "euclidean_dist(a,b) → L2 distance\n  sqrt(sum (a_i-b_i)²).\n  Complexity: O(n). Example: auto d = euclidean_dist(x, y);"},
        {"manhattan_distance",
         "manhattan_distance(a,b) → L1 distance\n  sum |a_i-b_i|.\n  Complexity: O(n). Example: auto d = manhattan_distance(x, y);"},
        {"cosine",
         "cosine(a,b) → [-1,1]\n  Cosine similarity (not distance).\n  Complexity: O(n). Example: auto s = cosine(x, y);"},
        {"pwr",
         "pwr(x,n) → x^n\n  Fast exponentiation (int or real exponent overloads).\n  Complexity: O(log n) for int n."},
        {"variance",
         "variance(data) → σ²\n  Population variance of a 1D vector.\n  Complexity: O(n)."},
        {"covariance",
         "covariance(x,y) → cov\n  Population covariance of two equal-length vectors.\n  Complexity: O(n)."},
        {"transpose",
         "transpose(M) → Mᵀ\n  Swaps rows and columns.\n  Complexity: O(n*m). Example: auto Mt = transpose(M);"},
    };

    auto it = kDocs.find(func_name);
    if (it == kDocs.end()) {
        Log::warn("Unknown function. Call help() for list.");
        return;
    }
    Log::header(func_name);
    Log::info(it->second);
    Log::divider();
}
