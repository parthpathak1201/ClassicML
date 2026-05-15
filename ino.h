#pragma once
#include "logger.hpp"
#include "type.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cml {

// Load a CSV file into Matrix (features) and Vec (labels).
std::pair<Matrix, Vec> load_csv(const std::string &path,
                                 bool last_col_is_label = true,
                                 bool has_header = true);

// Save Matrix + Vec labels to a CSV file.
void save_csv(const std::string &path, const Matrix &X, const Vec &y);

// Generate k Gaussian clusters (blobs) with n total points.
std::pair<Matrix, Vec> generate_blobs(int n, int k, int seed = 42, double cluster_std = 1.0);

// Generate n linearly separable 2D points (2 classes).
std::pair<Matrix, Vec> generate_linearly_separable(int n, int seed = 42);

// Generate n XOR-pattern points (2D, nonlinearly separable).
std::pair<Matrix, Vec> generate_xor(int n, int seed = 42, double noise = 0.1);

// Generate two interleaving half-circles (moons).
std::pair<Matrix, Vec> generate_moons(int n, int seed = 42, double noise = 0.1);

// Print first n_rows rows of a Matrix to stdout.
void print_matrix(const Matrix &X, int n_rows = 5);

// Print a Vec to stdout.
void print_vec(const Vec &v, int n_elems = 10);

namespace detail {

inline std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

inline void shuffle_dataset(Matrix &X, Vec &y, int seed) {
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    Matrix shuffled_X;
    Vec shuffled_y;
    shuffled_X.reserve(X.size());
    shuffled_y.reserve(y.size());
    for (size_t idx : indices) {
        shuffled_X.push_back(X[idx]);
        if (!y.empty()) shuffled_y.push_back(y[idx]);
    }
    X = std::move(shuffled_X);
    if (!y.empty()) y = std::move(shuffled_y);
}

} // namespace detail

inline std::pair<Matrix, Vec> load_csv(const std::string &path,
                                        bool last_col_is_label,
                                        bool has_header) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + path);
    std::string line;
    if (has_header) std::getline(file, line);
    Matrix X;
    Vec y;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto fields = detail::split_csv_line(line);
        if (fields.empty()) continue;
        if (last_col_is_label) {
            if (fields.size() < 2) continue;
            Vec row;
            row.reserve(fields.size() - 1);
            for (size_t j = 0; j + 1 < fields.size(); ++j) row.push_back(std::stod(fields[j]));
            X.push_back(row);
            y.push_back(std::stod(fields.back()));
        } else {
            Vec row;
            row.reserve(fields.size());
            for (const auto &field : fields) row.push_back(std::stod(field));
            X.push_back(row);
        }
    }
    return {X, y};
}

inline void save_csv(const std::string &path, const Matrix &X, const Vec &y) {
    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + path);
    if (X.size() != y.size()) throw std::invalid_argument("X and y must have the same number of rows");
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            if (j > 0) file << ',';
            file << X[i][j];
        }
        file << ',' << y[i] << '\n';
    }
}

inline std::pair<Matrix, Vec> generate_blobs(int n, int k, int seed, double cluster_std) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> center_dist(-10.0, 10.0);
    std::normal_distribution<double> point_dist(0.0, cluster_std);
    Matrix X;
    Vec y;
    int points_per_cluster = n / k;
    for (int c = 0; c < k; ++c) {
        double cx = center_dist(rng);
        double cy = center_dist(rng);
        for (int i = 0; i < points_per_cluster; ++i) {
            X.push_back({cx + point_dist(rng), cy + point_dist(rng)});
            y.push_back(static_cast<double>(c));
        }
    }
    detail::shuffle_dataset(X, y, seed);
    return {X, y};
}

inline std::pair<Matrix, Vec> generate_linearly_separable(int n, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.8);
    Matrix X;
    Vec y;
    int half = n / 2;
    for (int i = 0; i < half; ++i) {
        X.push_back({-2.0 + noise(rng), -2.0 + noise(rng)});
        y.push_back(0.0);
    }
    for (int i = 0; i < half; ++i) {
        X.push_back({2.0 + noise(rng), 2.0 + noise(rng)});
        y.push_back(1.0);
    }
    detail::shuffle_dataset(X, y, seed);
    return {X, y};
}

inline std::pair<Matrix, Vec> generate_xor(int n, int seed, double noise) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coord_dist(0.5, 2.0);
    std::normal_distribution<double> noise_dist(0.0, noise);
    Matrix X;
    Vec y;
    int per_quadrant = n / 4;
    auto add_quadrant = [&](double sx, double sy, double label) {
        for (int i = 0; i < per_quadrant; ++i) {
            X.push_back({sx * coord_dist(rng) + noise_dist(rng), sy * coord_dist(rng) + noise_dist(rng)});
            y.push_back(label);
        }
    };
    add_quadrant(1.0, 1.0, 1.0);
    add_quadrant(-1.0, -1.0, 1.0);
    add_quadrant(1.0, -1.0, 0.0);
    add_quadrant(-1.0, 1.0, 0.0);
    detail::shuffle_dataset(X, y, seed);
    return {X, y};
}

inline std::pair<Matrix, Vec> generate_moons(int n, int seed, double noise) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise_dist(0.0, noise);
    Matrix X;
    Vec y;
    int half = n / 2;
    for (int i = 0; i < half; ++i) {
        double angle = M_PI * static_cast<double>(i) / half;
        X.push_back({std::cos(angle) + noise_dist(rng), std::sin(angle) + noise_dist(rng)});
        y.push_back(0.0);
    }
    for (int i = 0; i < half; ++i) {
        double angle = M_PI * static_cast<double>(i) / half;
        X.push_back({std::cos(angle) + 0.5 + noise_dist(rng), std::sin(angle) - 0.2 + noise_dist(rng)});
        y.push_back(1.0);
    }
    detail::shuffle_dataset(X, y, seed);
    return {X, y};
}

inline void print_matrix(const Matrix &X, int n_rows) {
    int rows_to_print = static_cast<int>(std::min(static_cast<size_t>(n_rows), X.size()));
    for (int i = 0; i < rows_to_print; ++i) {
        std::ostringstream row;
        for (size_t j = 0; j < X[i].size(); ++j) {
            if (j > 0) row << ' ';
            row << X[i][j];
        }
        Log::info(row.str());
    }
    if (static_cast<size_t>(rows_to_print) < X.size())
        Log::info("... (", (X.size() - static_cast<size_t>(rows_to_print)), " more rows)");
}

inline void print_vec(const Vec &v, int n_elems) {
    int elems_to_print = static_cast<int>(std::min(static_cast<size_t>(n_elems), v.size()));
    std::ostringstream line;
    for (int i = 0; i < elems_to_print; ++i) {
        if (i > 0) line << ' ';
        line << v[i];
    }
    Log::info(line.str());
    if (static_cast<size_t>(elems_to_print) < v.size())
        Log::info("... (", (v.size() - static_cast<size_t>(elems_to_print)), " more)");
}

} // namespace cml
