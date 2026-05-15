// K-Means demo: 3 blobs, compare init strategies.
#include "demo_common.hpp"

int main() {
    report_header("KMeans");
    auto [X, y] = generate_blobs(300, 3, 42);
    report_dataset(X, y);
    KMeans km_pp(3, 300, 5, KMeans::KMeansPP);
    km_pp.fit(X, true);
    Log::metric("KMeans++ inertia", km_pp.get_inertia());
    KMeans km_rand(3, 300, 5, KMeans::Random);
    km_rand.fit(X, true);
    Log::metric("Random init inertia", km_rand.get_inertia());
    auto pred = km_pp.predict(X);
    Vec lbl;
    for (size_t i = 0; i < std::min(pred.size(), size_t(10)); ++i) lbl.push_back(static_cast<double>(pred[i]));
    Log::info("Sample cluster ids (first 10)");
    print_vec(lbl, 10);
    Log::success("KMeans demo finished.");
    return 0;
}
