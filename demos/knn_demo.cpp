// KNN demo: 2-class blobs; compare k=1,5,15.
#include "../logger.hpp"
#include "../ino.h"
#include "../metrics.hpp"
#include "demo_common.hpp"
#include "../models.h"

using namespace cml;

int main() {
    Log::header("KNearestNeighbors Demo");
    Log::info("k=1 memorizes training set (often 100% train accuracy).");

    auto [X, y] = generate_blobs(200, 2, 42);
    auto [X_train, X_test, y_train, y_test] = split_xy(X, y);

    for (int k : {1, 5, 15}) {
        KNearestNeighbors knn(k, KNearestNeighbors::Euclidean);
        knn.fit(X_train, y_train, true);
        Vec train_preds = knn.predict(X_train);
        Vec test_preds = knn.predict(X_test);
        Log::metric("k train acc", k, knn.score(y_train, train_preds));
        Log::metric("k test acc", k, knn.score(y_test, test_preds));
    }

    KNearestNeighbors::docs();
    Log::divider();
    Log::success("KNN demo finished.");
    return 0;
}
