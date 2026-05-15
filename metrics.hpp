#pragma once
#include "logger.hpp"
#include "type.hpp"
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cml {

inline void validate_metric_sizes(const Vec& y_true, const Vec& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size()) throw std::invalid_argument("Size mismatch");
}

inline double accuracy(const Vec& y_true, const Vec& y_pred) {
    validate_metric_sizes(y_true, y_pred);
    double correct = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) if (y_true[i] == y_pred[i]) correct++;
    return correct / static_cast<double>(y_true.size());
}

inline double precision(const Vec& y_true, const Vec& y_pred, double pos = 1.0) {
    validate_metric_sizes(y_true, y_pred);
    double tp = 0.0, fp = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == pos) {
            if (y_true[i] == pos) tp++; else fp++;
        }
    }
    return (tp + fp == 0.0) ? 0.0 : tp / (tp + fp);
}

inline double recall(const Vec& y_true, const Vec& y_pred, double pos = 1.0) {
    validate_metric_sizes(y_true, y_pred);
    double tp = 0.0, fn = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == pos) {
            if (y_pred[i] == pos) tp++; else fn++;
        }
    }
    return (tp + fn == 0.0) ? 0.0 : tp / (tp + fn);
}

inline double f1_score(const Vec& y_true, const Vec& y_pred, double pos = 1.0) {
    const double p = precision(y_true, y_pred, pos);
    const double r = recall(y_true, y_pred, pos);
    return (p + r == 0.0) ? 0.0 : 2.0 * p * r / (p + r);
}

inline Matrix confusion_matrix(const Vec& y_true, const Vec& y_pred) {
    validate_metric_sizes(y_true, y_pred);
    std::set<double> cls_set(y_true.begin(), y_true.end());
    cls_set.insert(y_pred.begin(), y_pred.end());
    std::vector<double> classes(cls_set.begin(), cls_set.end());
    std::map<double, int> idx;
    for (size_t i = 0; i < classes.size(); ++i) idx[classes[i]] = static_cast<int>(i);

    Matrix cm(classes.size(), Vec(classes.size(), 0.0));
    for (size_t i = 0; i < y_true.size(); ++i) cm[idx[y_true[i]]][idx[y_pred[i]]]++;
    return cm;
}

inline double rmse(const Vec& y_true, const Vec& y_pred) {
    validate_metric_sizes(y_true, y_pred);
    double mse = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const double d = y_true[i] - y_pred[i];
        mse += d * d;
    }
    return std::sqrt(mse / static_cast<double>(y_true.size()));
}

inline double mae(const Vec& y_true, const Vec& y_pred) {
    validate_metric_sizes(y_true, y_pred);
    double total = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) total += std::abs(y_true[i] - y_pred[i]);
    return total / static_cast<double>(y_true.size());
}

inline double r2_score(const Vec& y_true, const Vec& y_pred) {
    validate_metric_sizes(y_true, y_pred);
    double mean = 0.0;
    for (double v : y_true) mean += v;
    mean /= static_cast<double>(y_true.size());
    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const double res = y_true[i] - y_pred[i];
        const double tot = y_true[i] - mean;
        ss_res += res * res;
        ss_tot += tot * tot;
    }
    return (ss_tot == 0.0) ? 0.0 : 1.0 - ss_res / ss_tot;
}

inline void classification_report(const Vec& y_true, const Vec& y_pred) {
    validate_metric_sizes(y_true, y_pred);
    std::set<double> cls_set(y_true.begin(), y_true.end());
    std::vector<double> classes(cls_set.begin(), cls_set.end());

    Log::header("Classification Report");
    Log::metric("Accuracy", accuracy(y_true, y_pred));

    if (classes.size() == 2) {
        const double pos = classes.back();
        Log::metric("Precision", precision(y_true, y_pred, pos));
        Log::metric("Recall", recall(y_true, y_pred, pos));
        Log::metric("F1", f1_score(y_true, y_pred, pos));
    } else {
        double avg_p = 0.0, avg_r = 0.0, avg_f = 0.0;
        for (double c : classes) {
            avg_p += precision(y_true, y_pred, c);
            avg_r += recall(y_true, y_pred, c);
            avg_f += f1_score(y_true, y_pred, c);
        }
        const double denom = classes.empty() ? 1.0 : static_cast<double>(classes.size());
        Log::metric("Precision (macro)", avg_p / denom);
        Log::metric("Recall (macro)", avg_r / denom);
        Log::metric("F1 (macro)", avg_f / denom);
    }
    Log::divider();

    Matrix cm = confusion_matrix(y_true, y_pred);
    std::set<double> all_classes(y_true.begin(), y_true.end());
    all_classes.insert(y_pred.begin(), y_pred.end());
    classes.assign(all_classes.begin(), all_classes.end());

    Log::info("Confusion Matrix");
    std::ostringstream hdr;
    hdr << "         ";
    for (double c : classes) hdr << std::setw(9) << ("Pred " + std::to_string(static_cast<int>(c)));
    Log::info(hdr.str());

    for (size_t i = 0; i < classes.size(); ++i) {
        std::ostringstream row;
        row << "Actual " << static_cast<int>(classes[i]);
        for (size_t j = 0; j < classes.size(); ++j) row << std::setw(9) << static_cast<int>(cm[i][j]);
        Log::info(row.str());
    }
}

inline void regression_report(const Vec& y_true, const Vec& y_pred) {
    Log::header("Regression Report");
    Log::metric("R2", r2_score(y_true, y_pred));
    Log::metric("RMSE", rmse(y_true, y_pred));
    Log::metric("MAE", mae(y_true, y_pred));
    Log::divider();
}

} // namespace cml
