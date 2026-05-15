
#include "type.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
#include <cmath>


// Randomly splits data into train/test subsets. test_size is fraction (e.g. 0.2 = 20% test).
template <typename T>
std::pair<std::vector<T>, std::vector<T>>
train_test_split(std::vector<T> &data,
                 double test_size = 0.2,
                 unsigned int random_state = std::random_device{}())
{
    size_t N = data.size();

    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i)
        idx[i] = i;

    std::mt19937 rng(random_state);
    std::shuffle(idx.begin(), idx.end(), rng);

    size_t test_count = static_cast<size_t>(N * test_size);

    std::vector<T> train_data, test_data;
    train_data.reserve(N - test_count);
    test_data.reserve(test_count);

    for (size_t i = 0; i < N; ++i)
    {
        if (i < test_count)
            test_data.push_back(data[idx[i]]);
        else
            train_data.push_back(data[idx[i]]);
    }

    return std::make_pair(train_data, test_data);
}


// StandardScaler
// Standardizes features to zero mean and unit variance.
// IMPORTANT: Always fit() on training data only. Never fit() on test data.
class StandardScaler
{

public:
    std::vector<long double> mean_vec, std_vec;

    template <typename T>
    std::vector<T> __mean(const std::vector<std::vector<T>> &data)
    {
        std::vector<T> mean(data[0].size(), 0.0);
        for (const auto &row : data)
            for (size_t i = 0; i < row.size(); ++i)
                mean[i] += row[i];

        for (auto &val : mean)
            val /= data.size();

        return mean;
    }

    template <typename T>
    std::vector<T> __STD(const std::vector<std::vector<T>> &data, const std::vector<T> &mean)
    {
        std::vector<T> std(data[0].size(), 0.0);

        for (const auto &row : data)
            for (size_t i = 0; i < row.size(); ++i)
                std[i] += (row[i] - mean[i]) * (row[i] - mean[i]);

        for (auto &val : std)
        {
            val = sqrt(val / data.size());
            if (val == 0)
                val = 1;
        }

        return std;
    }

    template <typename T>
    void fit(const std::vector<std::vector<T>> &data)
    {
        mean_vec = __mean(data);
        std_vec = __STD(data, mean_vec);
    }

    template <typename T>
    std::vector<std::vector<T>>
    transform(const std::vector<std::vector<T>> &data)
    {
        std::vector<std::vector<T>> out(
            data.size(), std::vector<T>(data[0].size()));

        for (size_t i = 0; i < data.size(); ++i)
            for (size_t j = 0; j < data[i].size(); ++j)
                out[i][j] = (data[i][j] - mean_vec[j]) / std_vec[j];

        return out;
    }
};

// MinMaxScaler
// Scales features to [0, 1] range.
// Same fit/transform pattern as StandardScaler.
class MinMaxScaler
{

public:
    std::vector<long double> min_vec, max_vec;

    template <typename T>
    void fit(const std::vector<std::vector<T>> &data)
    {
        min_vec.resize(data[0].size(), std::numeric_limits<T>::max());
        max_vec.resize(data[0].size(), std::numeric_limits<T>::lowest());

        for (const auto &row : data)
            for (size_t i = 0; i < row.size(); ++i)
            {
                if (row[i] < min_vec[i])
                    min_vec[i] = row[i];
                if (row[i] > max_vec[i])
                    max_vec[i] = row[i];
            }
    }

    template <typename T>
    std::vector<std::vector<T>>
    transform(const std::vector<std::vector<T>> &data)
    {
        std::vector<std::vector<T>> out(
            data.size(), std::vector<T>(data[0].size()));

        for (size_t i = 0; i < data.size(); ++i)
        {

            for (size_t j = 0; j < data[i].size(); ++j)
            {

                T denom = (T)(max_vec[j] - min_vec[j]);
                if (denom == 0)
                {
                    denom = 1;
                }

                out[i][j] = (data[i][j] - (T)min_vec[j]) / denom;
            }
        }

        return out;
    }
};