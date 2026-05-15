#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cml {

template <typename T>
std::pair<std::vector<T>, std::vector<T>>
train_test_split(std::vector<T> &data, double test_size = 0.2, unsigned int random_state = 42);

class StandardScaler {
public:
    std::vector<long double> mean_vec, std_vec;

    template <typename T> void fit(const std::vector<std::vector<T>> &data);
    template <typename T> std::vector<std::vector<T>> transform(const std::vector<std::vector<T>> &data) const;

private:
    template <typename T> std::vector<T> mean(const std::vector<std::vector<T>> &data) const;
    template <typename T> std::vector<T> stddev(const std::vector<std::vector<T>> &data, const std::vector<T> &mean) const;
};

class MinMaxScaler {
public:
    std::vector<long double> min_vec, max_vec;

    template <typename T> void fit(const std::vector<std::vector<T>> &data);
    template <typename T> std::vector<std::vector<T>> transform(const std::vector<std::vector<T>> &data) const;
};

template <typename T>
std::pair<std::vector<T>, std::vector<T>>
train_test_split(std::vector<T> &data, double test_size, unsigned int random_state) {
    if (test_size < 0.0 || test_size > 1.0) {
        throw std::invalid_argument("test_size must be in [0, 1]");
    }

    const size_t n = data.size();
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;

    std::mt19937 rng(random_state);
    std::shuffle(idx.begin(), idx.end(), rng);

    const size_t test_count = static_cast<size_t>(static_cast<double>(n) * test_size);
    std::vector<T> train_data, test_data;
    train_data.reserve(n - test_count);
    test_data.reserve(test_count);

    for (size_t i = 0; i < n; ++i) {
        if (i < test_count) test_data.push_back(data[idx[i]]);
        else train_data.push_back(data[idx[i]]);
    }
    return {train_data, test_data};
}

template <typename T>
std::vector<T> StandardScaler::mean(const std::vector<std::vector<T>> &data) const {
    if (data.empty() || data[0].empty()) throw std::invalid_argument("fit: empty data");
    std::vector<T> result(data[0].size(), 0.0);
    for (const auto &row : data) {
        if (row.size() != result.size()) throw std::invalid_argument("fit: ragged matrix");
        for (size_t i = 0; i < row.size(); ++i) result[i] += row[i];
    }
    for (auto &v : result) v /= static_cast<T>(data.size());
    return result;
}

template <typename T>
std::vector<T> StandardScaler::stddev(const std::vector<std::vector<T>> &data, const std::vector<T> &m) const {
    std::vector<T> result(data[0].size(), 0.0);
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            const T diff = row[i] - m[i];
            result[i] += diff * diff;
        }
    }
    for (auto &v : result) {
        v = std::sqrt(v / static_cast<T>(data.size()));
        if (v == 0) v = 1;
    }
    return result;
}

template <typename T>
void StandardScaler::fit(const std::vector<std::vector<T>> &data) {
    const auto m = mean(data);
    const auto s = stddev(data, m);
    mean_vec.assign(m.begin(), m.end());
    std_vec.assign(s.begin(), s.end());
}

template <typename T>
std::vector<std::vector<T>> StandardScaler::transform(const std::vector<std::vector<T>> &data) const {
    if (data.empty() || data[0].empty()) throw std::invalid_argument("transform: empty data");
    if (mean_vec.empty() || std_vec.empty() || data[0].size() != mean_vec.size()) {
        throw std::invalid_argument("transform: scaler is not fitted or dimensions mismatch");
    }
    std::vector<std::vector<T>> out(data.size(), std::vector<T>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != mean_vec.size()) throw std::invalid_argument("transform: ragged matrix");
        for (size_t j = 0; j < data[i].size(); ++j) {
            out[i][j] = static_cast<T>((data[i][j] - mean_vec[j]) / std_vec[j]);
        }
    }
    return out;
}

template <typename T>
void MinMaxScaler::fit(const std::vector<std::vector<T>> &data) {
    if (data.empty() || data[0].empty()) throw std::invalid_argument("fit: empty data");
    min_vec.assign(data[0].size(), std::numeric_limits<long double>::max());
    max_vec.assign(data[0].size(), std::numeric_limits<long double>::lowest());
    for (const auto &row : data) {
        if (row.size() != min_vec.size()) throw std::invalid_argument("fit: ragged matrix");
        for (size_t i = 0; i < row.size(); ++i) {
            if (row[i] < min_vec[i]) min_vec[i] = row[i];
            if (row[i] > max_vec[i]) max_vec[i] = row[i];
        }
    }
}

template <typename T>
std::vector<std::vector<T>> MinMaxScaler::transform(const std::vector<std::vector<T>> &data) const {
    if (data.empty() || data[0].empty()) throw std::invalid_argument("transform: empty data");
    if (min_vec.empty() || max_vec.empty() || data[0].size() != min_vec.size()) {
        throw std::invalid_argument("transform: scaler is not fitted or dimensions mismatch");
    }
    std::vector<std::vector<T>> out(data.size(), std::vector<T>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != min_vec.size()) throw std::invalid_argument("transform: ragged matrix");
        for (size_t j = 0; j < data[i].size(); ++j) {
            long double denom = max_vec[j] - min_vec[j];
            if (denom == 0) denom = 1;
            out[i][j] = static_cast<T>((data[i][j] - min_vec[j]) / denom);
        }
    }
    return out;
}

} // namespace cml
