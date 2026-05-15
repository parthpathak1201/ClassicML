#pragma once
#include "config.hpp"
#include "logger.hpp"
#include <vector>
#include <functional>
#include <random>
#include <numeric>
#include <algorithm>
#include <cassert>

namespace cml {

template<typename ParamType, typename DataType>
class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void step(
        ParamType &params,
        const DataType &data,
        std::function<void(const DataType &, const size_t *, const size_t *, const ParamType &, ParamType &)>
        compute_gradients,
        size_t epoch
    ) = 0;

    virtual void shuffle_indices() = 0;
};

// GradientDescent<ParamType, DataType>
// Generic gradient descent optimizer supporting Batch, Stochastic, and MiniBatch modes.
// ParamType must support: operator-, operator+, operator*, operator/, zero()
// DataType is the training data container passed to compute_gradients.
//
// Usage:
//   GradientDescent<LinearParams, Matrix> opt(GradientDescent<...>::MiniBatch, 0.01, 32, 0.0);
//   opt.set_n_samples(n);
//   opt.step(params, data, grad_func, epoch);
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
    GradientDescent(Type t = Batch, double lr = 0.01, size_t bs = 32, double l2 = 0.0)
        : type(t), learning_rate(lr), batch_size(bs), l2_lambda(l2), n_samples(0),
          rng(std::random_device{}()), verbose(false) {
    }

    void set_verbose(bool v) { verbose = v; }
    [[nodiscard]] bool get_verbose() const { return verbose; }

    // Resize and reset the index permutation to [0, 1, ..., n-1]
    void set_n_samples(size_t n) {
        assert(n > 0);
        n_samples = n;
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0);
    }

    // Shuffle sample indices each epoch (Stochastic and MiniBatch only)
    void shuffle_indices() override {
        if (type == Stochastic || type == MiniBatch) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
    }

    void step(
        ParamType &params,
        const DataType &data,
        std::function<void(const DataType &, const size_t *, const size_t *, const ParamType &, ParamType &)>
        compute_gradients,
        size_t epoch
    ) override {
        ParamType gradients;

        if (type == Batch) {
            gradients.zero();
            compute_gradients(data, indices.data(), indices.data() + n_samples, params, gradients);
            params = apply_update(params, gradients, n_samples);
        } else if (type == Stochastic) {
            for (size_t i = 0; i < n_samples; ++i) {
                gradients.zero();
                compute_gradients(data, &indices[i], &indices[i] + 1, params, gradients);
                params = apply_update(params, gradients, 1);
            }
        } else if (type == MiniBatch){
            for (size_t start = 0; start < n_samples; start += batch_size) {
                size_t end = std::min(start + batch_size, n_samples);
                gradients.zero();
                compute_gradients(data, &indices[start], &indices[end], params, gradients);
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
    Type get_type() const { return type; }
    [[nodiscard]] const std::vector<size_t> &get_indices() const { return indices; }

private:
    // Update rule (weight decay / L2):
    //   w = w - lr * (grad / batch_size + w * l2_lambda)
    // Equivalent to:
    //   w -= lr * grad / n          (standard gradient descent)
    //   w -= lr * l2_lambda * w     (weight decay)
    ParamType apply_update(const ParamType &params, const ParamType &gradients, size_t batch_size) {
        return params - learning_rate * (gradients / batch_size + params * l2_lambda);
    }
};

class Adam {
    //baad me
};

} // namespace cml
