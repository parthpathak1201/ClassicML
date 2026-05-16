#pragma once

namespace detail {
inline bool log_epoch_milestone(size_t epoch, size_t total, bool verbose) {
    if (!verbose) return false;
    const size_t step = std::max(size_t(1), total / 10);
    return epoch % step == 0 || epoch == total - 1;
}
}
