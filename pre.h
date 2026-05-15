#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace cml {

// Randomly splits data into train/test subsets. test_size is fraction (e.g. 0.2 = 20% test).
template <typename T>
std::pair<std::vector<T>, std::vector<T>> train_test_split(
    std::vector<T> &data,
    double test_size = 0.2,
    unsigned int random_state = std::random_device{}());

// StandardScaler — zero mean, unit variance per feature (fit on train only).
class StandardScaler {
public:
    std::vector<long double> mean_vec, std_vec;

    template <typename T>
    void fit(const std::vector<std::vector<T>> &data);

    template <typename T>
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>> &data);

private:
    template <typename T>
    std::vector<T> __mean(const std::vector<std::vector<T>> &data);

    template <typename T>
    std::vector<T> __STD(const std::vector<std::vector<T>> &data, const std::vector<T> &mean);
};

// MinMaxScaler — scales features to [0, 1] using fit/transform.
class MinMaxScaler {
public:
    std::vector<long double> min_vec, max_vec;

    template <typename T>
    void fit(const std::vector<std::vector<T>> &data);

    template <typename T>
    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>> &data);
};

template <typename T>
std::pair<std::vector<T>, std::vector<T>>
train_test_split(std::vector<T> &data, double test_size, unsigned int random_state) {
    size_t N = data.size();
    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) idx[i] = i;
    std::mt19937 rng(random_state);
    std::shuffle(idx.begin(), idx.end(), rng);
    size_t test_count = static_cast<size_t>(N * test_size);
    std::vector<T> train_data, test_data;
    train_data.reserve(N - test_count);
    test_data.reserve(test_count);
    for (size_t i = 0; i < N; ++i) {
        if (i < test_count) test_data.push_back(data[idx[i]]);
        else train_data.push_back(data[idx[i]]);
    }
    return std::make_pair(train_data, test_data);
}

template <typename T>
std::vector<T> StandardScaler::__mean(const std::vector<std::vector<T>> &data) {
    std::vector<T> mean(data[0].size(), 0.0);
    for (const auto &row : data)
        for (size_t i = 0; i < row.size(); ++i) mean[i] += row[i];
    for (auto &val : mean) val /= data.size();
    return mean;
}

template <typename T>
std::vector<T> StandardScaler::__STD(const std::vector<std::vector<T>> &data, const std::vector<T> &mean) {
    std::vector<T> stdv(data[0].size(), 0.0);
    for (const auto &row : data)
        for (size_t i = 0; i < row.size(); ++i) stdv[i] += (row[i] - mean[i]) * (row[i] - mean[i]);
    for (auto &val : stdv) {
        val = std::sqrt(val / data.size());
        if (val == 0) val = 1;
    }
    return stdv;
}

template <typename T>
void StandardScaler::fit(const std::vector<std::vector<T>> &data) {
    auto m = __mean(data);
    auto s = __STD(data, m);
    mean_vec.assign(m.begin(), m.end());
    std_vec.assign(s.begin(), s.end());
}

template <typename T>
std::vector<std::vector<T>> StandardScaler::transform(const std::vector<std::vector<T>> &data) {
    std::vector<std::vector<T>> out(data.size(), std::vector<T>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < data[i].size(); ++j)
            out[i][j] = (data[i][j] - mean_vec[j]) / std_vec[j];
    return out;
}

template <typename T>
void MinMaxScaler::fit(const std::vector<std::vector<T>> &data) {
    min_vec.resize(data[0].size(), std::numeric_limits<T>::max());
    max_vec.resize(data[0].size(), std::numeric_limits<T>::lowest());
    for (const auto &row : data)
        for (size_t i = 0; i < row.size(); ++i) {
            if (row[i] < min_vec[i]) min_vec[i] = row[i];
            if (row[i] > max_vec[i]) max_vec[i] = row[i];
        }
}

template <typename T>
std::vector<std::vector<T>> MinMaxScaler::transform(const std::vector<std::vector<T>> &data) {
    std::vector<std::vector<T>> out(data.size(), std::vector<T>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < data[i].size(); ++j) {
            T denom = static_cast<T>(max_vec[j] - min_vec[j]);
            if (denom == 0) denom = 1;
            out[i][j] = (data[i][j] - static_cast<T>(min_vec[j])) / denom;
        }
    return out;
}

} // namespace cml
