#pragma once
#include "config.hpp"
#include "logger.hpp"
#include "type.hpp"
#include <stdexcept>

namespace cml {

// Validates feature matrix and label vector before fit().
inline void chk_fit(const Matrix &X, const Vec &y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        if (verbosity >= Verbosity::DEBUG) Log::error("fit: empty data or row mismatch");
        throw std::invalid_argument("fit: dimension mismatch");
    }
}

// Validates feature columns against weight vector before predict().
inline void chk_predict(const Matrix &X, size_t n_w) {
    if (X.empty() || X[0].size() != n_w) {
        if (verbosity >= Verbosity::DEBUG) Log::error("predict: feature/weight mismatch");
        throw std::invalid_argument("predict: dimension mismatch");
    }
}

// Validates unlabeled feature matrix before clustering fit().
inline void chk_X(const Matrix &X) {
    if (X.empty()) {
        if (verbosity >= Verbosity::DEBUG) Log::error("fit: empty feature matrix");
        throw std::invalid_argument("fit: empty feature matrix");
    }
}

} // namespace cml
