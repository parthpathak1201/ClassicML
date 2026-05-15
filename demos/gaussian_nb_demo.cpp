// Gaussian Naive Bayes demo on blobs.
#include "demo_common.hpp"

int main() {
    report_header("GaussianNB");
    auto [X, y] = generate_blobs(240, 3, 42);
    report_dataset(X, y);
    auto [X_tr, X_te, y_tr, y_te] = split_xy(X, y);
    GaussianNB gnb;
    gnb.fit(X_tr, y_tr, true);
    report_classification_metrics(y_te, gnb.predict(X_te));
    Log::success("GaussianNB demo finished.");
    return 0;
}
