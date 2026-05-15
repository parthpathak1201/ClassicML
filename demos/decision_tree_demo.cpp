// Decision tree demo on XOR data.
#include "demo_common.hpp"

int main() {
    report_header("DecisionTree");
    auto [X, y] = generate_xor(200, 42, 0.1);
    report_dataset(X, y);
    auto [X_tr, X_te, y_tr, y_te] = split_xy(X, y);
    DecisionTree tree(5);
    tree.fit(X_tr, y_tr, true);
    report_classification_metrics(y_te, tree.predict(X_te));
    Log::success("DecisionTree demo finished.");
    return 0;
}
