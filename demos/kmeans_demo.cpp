// K-Means demo: 3 blobs, Random vs KMeans++ init, centroids and inertia.
#include "../logger.hpp"
#include "../ino.h"
#include "demo_common.hpp"
#include "../models.h"

using namespace cml;

int main() {
    Log::header("KMeans Demo");
    Log::info("300 points, 3 clusters; compare init strategies.");

    auto [X, y] = generate_blobs(300, 3, 42);
    print_matrix(X, 5);

    KMeans km_pp(3, 300, 5, KMeans::KMeansPP);
    km_pp.fit(X, true);
    Log::metric("KMeans++ inertia", km_pp.get_inertia());
    print_matrix(km_pp.get_centroids(), 3);

    auto labels = km_pp.predict(X);
    Log::info("Predicted cluster ids (first 10):");
    Vec lbl_vec(labels.begin(), labels.begin() + std::min(labels.size(), size_t(10)));
    print_vec(lbl_vec, 10);

    KMeans km_rand(3, 300, 5, KMeans::Random);
    km_rand.fit(X, true);
    Log::metric("Random init inertia", km_rand.get_inertia());

    KMeans::docs();
    Log::divider();
    Log::success("KMeans demo finished.");
    return 0;
}
