// Logistic regression demo: linearly separable blobs, probabilities, classification report.
#include "../logger.hpp"
#include "../ino.h"
#include "../metrics.hpp"
#include "../pre.h"
#include "demo_common.hpp"
#include "../models.h"

using namespace cml;

int main() {
    Log::header("LogisticRegression Demo");
    Log::info("Binary classification on generate_linearly_separable(200).");

    auto [X, y] = generate_linearly_separable(200, 42);
    print_matrix(X, 5);
    print_vec(y, 5);

    auto [X_train, X_test, y_train, y_test] = split_xy(X, y, 0.2, 42);
    StandardScaler scaler;
    scaler.fit(X_train);
    X_train = scaler.transform(X_train);
    X_test = scaler.transform(X_test);

    LogisticRegression lr(GradientDescent<LinearParams, Matrix>::MiniBatch, 0.1, 32, 0.01);
    lr.fit(X_train, y_train, 300, true);
    print_loss_bars(lr.costs);

    Vec probs = lr.predict_prob(X_test);
    Vec preds = lr.predict(X_test);
    Log::info("Sample probabilities (first 5):");
    print_vec(probs, 5);

    classification_report(y_test, preds);
    LogisticRegression::docs();
    Log::divider();
    Log::success("LogisticRegression demo finished.");
    return 0;
}
