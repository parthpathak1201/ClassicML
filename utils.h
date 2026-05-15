#pragma once
#include <string>
#include <vector>

namespace cml {

// NOTE: Functions in this file operate on std::vector<long double> and
// std::vector<std::vector<long double>>. They are standalone math utilities
// for educational use and are NOT integrated into the ML pipeline, which uses
// Vec (std::vector<double>) and Matrix (std::vector<std::vector<double>>).

long double pwr(long double x, int n);
long double pwr(long double x, long double n);
long double custom_exp(long double x);
double sigmoid(double z);
long double dot(const std::vector<long double> &a, const std::vector<long double> &b);
long double manhattan_distance(const std::vector<double> &a, const std::vector<double> &b);
long double euclidean_dist(const std::vector<double> &a, const std::vector<double> &b);
long double square_distance(const std::vector<double> &a, const std::vector<double> &b);
long double cosine(const std::vector<double> &a, const std::vector<double> &b);
long double variance(const std::vector<long double> &data);
long double covariance(const std::vector<long double> &x, const std::vector<long double> &y);
long double MSE(const std::vector<long double> &y_true, const std::vector<long double> &y_pred);
long double MAE(const std::vector<long double> &y_true, const std::vector<long double> &y_pred);
long double R2(const std::vector<long double> &y_true, const std::vector<long double> &y_pred);
long long factorial(int n);
long long ncr(int n, int r);
std::vector<long double> vec_mean(const std::vector<std::vector<long double>> &data);
std::vector<long double> vec_std(const std::vector<std::vector<long double>> &data, const std::vector<long double> &mean);
std::vector<long double> row(const std::vector<std::vector<long double>> &data, int row_index);
std::vector<long double> col(const std::vector<std::vector<long double>> &data, int col_index);
std::vector<std::vector<long double>> norm(const std::vector<std::vector<long double>> &data);
std::vector<std::vector<long double>> transpose(const std::vector<std::vector<long double>> &data);
std::vector<std::vector<long double>> matinv(std::vector<std::vector<long double>> A);
std::vector<std::vector<long double>> matmul(const std::vector<std::vector<long double>> &A, const std::vector<std::vector<long double>> &B);
std::vector<std::vector<long double>> add(const std::vector<std::vector<long double>> &A, const std::vector<std::vector<long double>> &B);
std::vector<std::vector<long double>> subtract(const std::vector<std::vector<long double>> &A, const std::vector<std::vector<long double>> &B);
void help();
void docs(const std::string &func_name);

} // namespace cml
