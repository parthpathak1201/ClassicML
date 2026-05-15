// Decision tree demo: 3-class blobs, print_tree, shallow vs deep trees.
#include "../logger.hpp"
#include "../ino.h"
#include "../metrics.hpp"
#include "demo_common.hpp"
#include "../models.h"

using namespace cml;

int main() {
    Log::header("DecisionTree Demo");
    Log::info("3-class blobs; compare max_depth=2 vs max_depth=10.");

    auto [X, y] = generate_blobs(150, 3, 42);
    print_matrix(X, 5);
    print_vec(y, 5);

    auto [X_train, X_test, y_train, y_test] = split_xy(X, y);

    DecisionTree shallow(2, 2);
    shallow.fit(X_train, y_train, true);
    shallow.print_tree();
    Vec p1 = shallow.predict(X_test);
    Log::metric("depth=2 accuracy", shallow.score(y_test, p1));

    DecisionTree deep(10, 2);
    deep.fit(X_train, y_train, true);
    deep.print_tree();
    Vec p2 = deep.predict(X_test);
    Log::metric("depth=10 accuracy", deep.score(y_test, p2));

    DecisionTree::docs();
    Log::divider();
    Log::success("DecisionTree demo finished.");
    return 0;
}
