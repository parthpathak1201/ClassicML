// SVM demo: linearly separable data; compare lambda=0.001 vs 0.1.
#include "../logger.hpp"
#include "../ino.h"
#include "../metrics.hpp"
#include "demo_common.hpp"
#include "../models.h"

using namespace cml;

int main() {
    Log::header("SVM Demo");
    Log::info("Labels are mapped to {-1,+1} internally; predictions shown as 0/1.");

    auto [X, y] = generate_linearly_separable(200, 42);
    auto [X_train, X_test, y_train, y_test] = split_xy(X, y);

    StandardScaler scaler;
    scaler.fit(X_train);
    X_train = scaler.transform(X_train);
    X_test = scaler.transform(X_test);

    for (double lambda : {0.001, 0.1}) {
        Log::info("Training SVM with lambda=", lambda);
        SVM svm(0.001, lambda, 500);
        svm.fit(X_train, y_train, true);
        Vec preds = svm.predict(X_test);
        Log::metric("test accuracy", svm.score(y_test, preds));
    }

    SVM::docs();
    Log::divider();
    Log::success("SVM demo finished.");
    return 0;
}
