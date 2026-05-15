// metrics.hpp
// Evaluation metrics for classification and regression.
// All functions operate on Vec (std::vector<double>).
//
// Classification: accuracy, precision, recall, f1_score, confusion_matrix
// Regression:     rmse
//
// precision/recall/f1 are binary only — pass pos= to specify positive class label.
// confusion_matrix supports multi-class — rows=actual, cols=predicted.
#pragma once
#include "config.hpp"
#include "type.hpp"
#include "logger.hpp"
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>

namespace cml {

// Classification accuracy: fraction of matching labels.
inline double accuracy(const Vec& y_true, const Vec& y_pred) {
    if (y_true.size() != y_pred.size()) throw std::invalid_argument("Size mismatch");
    double correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i]) correct++;
    return correct / y_true.size();
}

// binary only — specify positive class label
inline double precision(const Vec& y_true, const Vec& y_pred, double pos = 1.0) {
    double tp = 0, fp = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == pos) {
            if (y_true[i] == pos) tp++; else fp++;
        }
    }
    return (tp + fp == 0) ? 0.0 : tp / (tp + fp);
}

inline double recall(const Vec& y_true, const Vec& y_pred, double pos = 1.0) {
    double tp = 0, fn = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == pos) {
            if (y_pred[i] == pos) tp++; else fn++;
        }
    }
    return (tp + fn == 0) ? 0.0 : tp / (tp + fn);
}

inline double f1_score(const Vec& y_true, const Vec& y_pred, double pos = 1.0) {
    double p = precision(y_true, y_pred, pos);
    double r = recall(y_true, y_pred, pos);
    return (p + r == 0) ? 0.0 : 2 * p * r / (p + r);
}

// returns [n_classes x n_classes] matrix, rows=actual, cols=predicted
inline Matrix confusion_matrix(const Vec& y_true, const Vec& y_pred) {
    std::set<double> cls_set(y_true.begin(), y_true.end());
    std::vector<double> classes(cls_set.begin(), cls_set.end());
    int n = classes.size();
    std::map<double, int> idx;
    for (int i = 0; i < n; ++i) idx[classes[i]] = i;

    Matrix cm(n, Vec(n, 0.0));
    for (size_t i = 0; i < y_true.size(); ++i)
        cm[idx[y_true[i]]][idx[y_pred[i]]]++;
    return cm;
}

inline double rmse(const Vec& y_true, const Vec& y_pred) {
    double mse = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        mse += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    return std::sqrt(mse / y_true.size());
}

inline double mae(const Vec& y_true, const Vec& y_pred) {
    double total = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        total += std::abs(y_true[i] - y_pred[i]);
    return total / y_true.size();
}

inline double r2_score(const Vec& y_true, const Vec& y_pred) {
    double mean = 0;
    for (double v : y_true) mean += v;
    mean /= y_true.size();
    double ss_res = 0, ss_tot = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
    }
    return (ss_tot == 0) ? 0.0 : 1.0 - ss_res / ss_tot;
}

// Prints a full classification report: accuracy, precision, recall, f1, confusion matrix
inline void classification_report(const Vec& y_true, const Vec& y_pred) {
    Log::header("Classification Report");
    Log::metric("Accuracy", accuracy(y_true, y_pred));
    Log::metric("Precision", precision(y_true, y_pred));
    Log::metric("Recall", recall(y_true, y_pred));
    Log::metric("F1", f1_score(y_true, y_pred));
    Log::divider();

    Matrix cm = confusion_matrix(y_true, y_pred);
    std::set<double> cls_set(y_true.begin(), y_true.end());
    std::vector<double> classes(cls_set.begin(), cls_set.end());

    Log::info("Confusion Matrix");
    std::ostringstream hdr;
    hdr << "         ";
    for (double c : classes) hdr << std::setw(9) << ("Pred " + std::to_string(static_cast<int>(c)));
    Log::info(hdr.str());

    for (size_t i = 0; i < classes.size(); ++i) {
        std::ostringstream row;
        row << "Actual " << static_cast<int>(classes[i]);
        for (size_t j = 0; j < classes.size(); ++j) {
            row << std::setw(9) << static_cast<int>(cm[i][j]);
        }
        Log::info(row.str());
    }
}

// Prints regression report: R2, RMSE, MAE
inline void regression_report(const Vec& y_true, const Vec& y_pred) {
    Log::header("Regression Report");
    Log::metric("R²", r2_score(y_true, y_pred));
    Log::metric("RMSE", rmse(y_true, y_pred));
    Log::metric("MAE", mae(y_true, y_pred));
    Log::divider();
}

} // namespace cml