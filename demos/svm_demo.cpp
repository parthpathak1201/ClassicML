// SVM demo on moons dataset.
#include "demo_common.hpp"

int main() {
    report_header("SVM");
    auto [X, y] = generate_moons(180, 42, 0.1);
    report_dataset(X, y);
    auto [X_tr, X_te, y_tr, y_te] = split_xy(X, y);
    StandardScaler sc;
    sc.fit(X_tr);
    X_tr = sc.transform(X_tr);
    X_te = sc.transform(X_te);
    SVM svm(1.0, 100);
    svm.fit(X_tr, y_tr, true);
    report_classification_metrics(y_te, svm.predict(X_te));
    Log::success("SVM demo finished.");
    return 0;
}
