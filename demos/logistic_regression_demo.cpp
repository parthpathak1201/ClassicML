// Logistic regression demo: linearly separable blobs.
#include "demo_common.hpp"

int main() {
    report_header("LogisticRegression");
    auto [X, y] = generate_linearly_separable(200, 42);
    report_dataset(X, y);
    auto [X_train, X_test, y_train, y_test] = split_xy(X, y, 0.2, 42);
    StandardScaler scaler;
    scaler.fit(X_train);
    X_train = scaler.transform(X_train);
    X_test = scaler.transform(X_test);
    LogisticRegression lr(GradientDescent<LinearParams, Matrix>::MiniBatch, 0.1, 32, 0.01);
    lr.fit(X_train, y_train, 300, true);
    print_loss_bars(lr.costs);
    report_classification_metrics(y_test, lr.predict(X_test));
    Log::success("LogisticRegression demo finished.");
    return 0;
}
