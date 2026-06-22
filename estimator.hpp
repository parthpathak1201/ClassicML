#pragma once
#include "type.hpp"

class Estimator {
public:
    virtual ~Estimator() = default;
    virtual void fit(const Matrix& X, const Vec& y) = 0;
    virtual Vec predict(const Matrix& X) = 0;
    virtual double score(const Vec& y_true, const Vec& y_pred) = 0;
};
