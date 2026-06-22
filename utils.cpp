#include "utils.h"
#include "logger.hpp"
#include <map>

// Taylor series approximation of e^x — 100 terms, sufficient for ML use
long double custom_exp(long double x) {
    long double sum = 1.0L, term = 1.0L;
    for (int n = 1; n < 100; ++n) {
        term *= x / n;
        sum += term;
    }
    return sum;
}

// Gauss-Jordan elimination to compute matrix inverse in-place
std::vector<std::vector<double>> matinv(std::vector<std::vector<double>> A) {
    int n = static_cast<int>(A.size());
    std::vector<std::vector<double>> result(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        result[i][i] = 1.0;
    }

    // step 1 - eliminate the lower half

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = 0; k < n; k++) {
                A[j][k] -= factor * A[i][k];
                result[j][k] -= factor * result[i][k];
            }
        }
    }

    // step 2 - eliminate the upper half
    for (int i = n - 1; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            double factor = A[j][i] / A[i][i];
            for (int k = 0; k < n; k++) {
                A[j][k] -= factor * A[i][k];
                result[j][k] -= factor * result[i][k];
            }
        }
    }

    // step 3 - fix the diagonal
    for (int i = 0; i < n; i++) {
        double factor = A[i][i];
        for (int k = 0; k < n; k++) {
            A[i][k] /= factor;
            result[i][k] /= factor;
        }
    }

    return result;
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
