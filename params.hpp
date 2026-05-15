#pragma once
#include "type.hpp"
#include <algorithm>

struct LinearParams {
    Vec weights{};
    double bias = 0.0;

    LinearParams operator-(const LinearParams &other) const {
        LinearParams result;
        result.weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            result.weights[i] = weights[i] - other.weights[i];
        }
        result.bias = bias - other.bias;
        return result;
    }

    LinearParams operator*(double scalar) const {
        LinearParams result;
        result.weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            result.weights[i] = weights[i] * scalar;
        }
        result.bias = bias * scalar;
        return result;
    }

    LinearParams operator/(double scalar) const {
        LinearParams result;
        result.weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            result.weights[i] = weights[i] / scalar;
        }
        result.bias = bias / scalar;
        return result;
    }

    LinearParams operator+(const LinearParams &other) const {
        LinearParams result;
        result.weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            result.weights[i] = weights[i] + other.weights[i];
        }
        result.bias = bias + other.bias;
        return result;
    }

    void zero() {
        std::fill(weights.begin(), weights.end(), 0.0);
        bias = 0.0;
    }
};

inline LinearParams operator*(double scalar, const LinearParams &params) {
    return params * scalar;
}
