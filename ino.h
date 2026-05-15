#pragma once
#include "type.hpp"
#include <string>
#include <utility>

namespace cml {

// Load a CSV file into Matrix (features) and Vec (labels).
// last_col_is_label: if true, last column is treated as label vector.
// has_header: if true, skip the first row.
std::pair<Matrix, Vec> load_csv(const std::string &path,
                                 bool last_col_is_label = true,
                                 bool has_header = true);

// Save Matrix + Vec labels to a CSV file.
// Writes features first, then label as last column.
void save_csv(const std::string &path,
              const Matrix &X,
              const Vec &y);

// Generate k Gaussian clusters (blobs) with n total points.
// Returns {X, labels} where labels are 0..k-1.
std::pair<Matrix, Vec> generate_blobs(int n, int k, int seed = 42,
                                       double cluster_std = 1.0);

// Generate n linearly separable 2D points (2 classes).
// Returns {X, labels} where labels are 0 or 1.
std::pair<Matrix, Vec> generate_linearly_separable(int n, int seed = 42);

// Generate n XOR-pattern points (2D, nonlinearly separable).
// Returns {X, labels} where labels are 0 or 1.
std::pair<Matrix, Vec> generate_xor(int n, int seed = 42, double noise = 0.1);

// Generate two interleaving half-circles (moons).
// Good for testing SVM and nonlinear boundaries.
// Returns {X, labels} where labels are 0 or 1.
std::pair<Matrix, Vec> generate_moons(int n, int seed = 42, double noise = 0.1);

// Print first n_rows rows of a Matrix to stdout.
// Useful for debugging.
void print_matrix(const Matrix &X, int n_rows = 5);

// Print a Vec to stdout.
void print_vec(const Vec &v, int n_elems = 10);

} // namespace cml
