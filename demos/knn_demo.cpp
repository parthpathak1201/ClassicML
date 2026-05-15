// KNN demo on moons dataset.
#include "demo_common.hpp"

int main() {
    report_header("KNearestNeighbors");
    auto [X, y] = generate_moons(200, 42, 0.15);
    report_dataset(X, y);
    auto [X_tr, X_te, y_tr, y_te] = split_xy(X, y);
    KNearestNeighbors knn(5);
    knn.fit(X_tr, y_tr, true);
    report_classification_metrics(y_te, knn.predict(X_te));
    Log::success("KNN demo finished.");
    return 0;
}
