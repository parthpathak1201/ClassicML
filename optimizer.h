#pragma once
#include "logger.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace cml {

template<typename ParamType, typename DataType>
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(
        ParamType &params,
        const DataType &data,
        std::function<void(const DataType &, const size_t *, const size_t *, const ParamType &, ParamType &)> compute_gradients,
        size_t epoch) = 0;
    virtual void shuffle_indices() = 0;
};

template<typename ParamType, typename DataType>
class GradientDescent : public Optimizer<ParamType, DataType> {
public:
    enum Type { Batch, Stochastic, MiniBatch };

private:
    Type type;
    double learning_rate;
    size_t batch_size;
    double l2_lambda;
    size_t n_samples;
    std::vector<size_t> indices;
    std::mt19937 rng;
    bool verbose;

public:
    // Fixed seed keeps SGD/MiniBatch shuffling reproducible without adding public API surface.
    GradientDescent(Type t = Batch, double lr = 0.01, size_t bs = 32, double l2 = 0.0)
        : type(t), learning_rate(lr), batch_size(bs), l2_lambda(l2), n_samples(0), rng(42), verbose(false) {}

    void set_verbose(bool v) { verbose = v; }
    [[nodiscard]] bool get_verbose() const { return verbose; }

    void set_n_samples(size_t n) {
        assert(n > 0);
        n_samples = n;
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0);
    }

    void shuffle_indices() override {
        if (type == Stochastic || type == MiniBatch) std::shuffle(indices.begin(), indices.end(), rng);
    }

    void step(
        ParamType &params,
        const DataType &data,
        std::function<void(const DataType &, const size_t *, const size_t *, const ParamType &, ParamType &)> compute_gradients,
        size_t epoch) override {
        ParamType gradients;

        if (type == Batch) {
            gradients.zero();
            compute_gradients(data, indices.data(), indices.data() + n_samples, params, gradients);
            params = apply_update(params, gradients, n_samples);
        } else if (type == Stochastic) {
            for (size_t i = 0; i < n_samples; ++i) {
                gradients.zero();
                compute_gradients(data, indices.data() + i, indices.data() + i + 1, params, gradients);
                params = apply_update(params, gradients, 1);
            }
        } else if (type == MiniBatch) {
            for (size_t start = 0; start < n_samples; start += batch_size) {
                const size_t end = std::min(start + batch_size, n_samples);
                gradients.zero();
                compute_gradients(data, indices.data() + start, indices.data() + end, params, gradients);
                params = apply_update(params, gradients, end - start);
            }
        }

        if (verbose && (epoch == 0 || epoch % 10 == 0)) {
            Log::info("Optimizer step epoch=", epoch, " mode=", static_cast<int>(type),
                      " lr=", learning_rate, " batch_size=", batch_size);
        }
    }

    [[nodiscard]] double get_learning_rate() const { return learning_rate; }
    [[nodiscard]] double get_l2_lambda() const { return l2_lambda; }
    [[nodiscard]] Type get_type() const { return type; }
    [[nodiscard]] const std::vector<size_t> &get_indices() const { return indices; }

private:
    ParamType apply_update(const ParamType &params, const ParamType &gradients, size_t bs) {
        return params - learning_rate * (gradients / bs + params * l2_lambda);
    }
};

// TODO: Adam optimizer.

} // namespace cml
