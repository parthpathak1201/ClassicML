#pragma once
#include "config.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace Color {
inline constexpr const char *Red = "\033[31m";
inline constexpr const char *Green = "\033[32m";
inline constexpr const char *Yellow = "\033[33m";
inline constexpr const char *Blue = "\033[34m";
inline constexpr const char *Magenta = "\033[35m";
inline constexpr const char *Cyan = "\033[36m";
inline constexpr const char *White = "\033[37m";
inline constexpr const char *Gray = "\033[90m";
inline constexpr const char *BoldRed = "\033[1;31m";
inline constexpr const char *BoldGreen = "\033[1;32m";
inline constexpr const char *BoldYellow = "\033[1;33m";
inline constexpr const char *BoldBlue = "\033[1;34m";
inline constexpr const char *BoldCyan = "\033[1;36m";
inline constexpr const char *Reset = "\033[0m";
} // namespace Color

namespace Style {
inline constexpr const char *Bold = "\033[1m";
inline constexpr const char *Dim = "\033[2m";
inline constexpr const char *Underline = "\033[4m";
inline constexpr const char *Reset = "\033[0m";
} // namespace Style

namespace Log {
namespace detail {

inline int digit_count(int n) {
    if (n <= 0) return 1;
    int d = 0;
    while (n > 0) {
        ++d;
        n /= 10;
    }
    return d;
}

template<typename T>
void print_arg(const T &v) {
    if constexpr (std::is_floating_point_v<T>) {
        std::cout << std::fixed << std::setprecision(4) << v;
    } else {
        std::cout << v;
    }
}

template<typename... Args>
void print_impl(const char *color, const char *prefix, Args &&...args) {
    std::cout << color << prefix;
    (print_arg(std::forward<Args>(args)), ...);
    std::cout << Color::Reset << '\n';
}

inline void print_raw(const char *color, const std::string &line) {
    std::cout << color << line << Color::Reset << '\n';
}

inline bool log_enabled() { return cml::verbosity != cml::Verbosity::SILENT; }
inline bool log_debug() { return cml::verbosity == cml::Verbosity::DEBUG; }

} // namespace detail

// Emits an [INFO] line when verbosity is BASIC or DEBUG.
template<typename... Args>
void info(Args &&...args) {
    if (!detail::log_enabled()) return;
    detail::print_impl(Color::Gray, "[INFO] ", std::forward<Args>(args)...);
}

// Emits an [OK] line when verbosity is BASIC or DEBUG.
template<typename... Args>
void success(Args &&...args) {
    if (!detail::log_enabled()) return;
    detail::print_impl(Color::Green, "[OK]   ", std::forward<Args>(args)...);
}

// Emits a [WARN] line when verbosity is BASIC or DEBUG.
template<typename... Args>
void warn(Args &&...args) {
    if (!detail::log_enabled()) return;
    detail::print_impl(Color::Yellow, "[WARN] ", std::forward<Args>(args)...);
}

// Emits an [ERR] line (always shown unless SILENT; DEBUG adds detail elsewhere).
template<typename... Args>
void error(Args &&...args) {
    if (cml::verbosity == cml::Verbosity::SILENT) return;
    detail::print_impl(Color::Red, "[ERR]  ", std::forward<Args>(args)...);
}

// Prints a bold section header when verbosity is BASIC or DEBUG.
template<typename... Args>
void header(Args &&...args) {
    if (!detail::log_enabled()) return;
    detail::print_impl(Color::BoldCyan, "", std::forward<Args>(args)...);
}

// Prints a horizontal rule when verbosity is BASIC or DEBUG.
inline void divider() {
    if (!detail::log_enabled()) return;
    detail::print_raw(Color::Gray, "────────────────────────────────────────────────────────────");
}

// Prints training epoch progress when verbosity is BASIC or DEBUG.
template<typename... Args>
void epoch(int e, int total, Args &&...args) {
    if (!detail::log_enabled()) return;
    const int width = detail::digit_count(total);
    std::cout << Color::BoldCyan << "[Epoch " << std::setw(width) << e << "/" << total << "]"
              << Color::Reset;
    (std::cout << ... << std::forward<Args>(args));
    std::cout << Color::Reset << '\n';
}

// Prints a named metric value when verbosity is BASIC or DEBUG.
template<typename... Args>
void metric(const std::string &name, Args &&...values) {
    if (!detail::log_enabled()) return;
    std::cout << Style::Bold << "  " << name << Style::Reset << "  →  " << Color::Green;
    (detail::print_arg(std::forward<Args>(values)), ...);
    std::cout << Color::Reset << '\n';
}

template<typename... Args>
void tree_node(int depth, bool is_leaf, Args &&...content) {
    std::cout << std::string(static_cast<size_t>(depth) * 2, ' ');
    if (is_leaf) {
        std::cout << Color::Green;
    } else {
        std::cout << Color::Cyan;
    }
    (std::cout << ... << std::forward<Args>(content));
    std::cout << Color::Reset << '\n';
}

template<typename... Args>
void table_header(Args &&...cols) {
    std::cout << Style::Bold;
    (std::cout << ... << (std::setw(10) << std::forward<Args>(cols)));
    std::cout << Style::Reset << '\n';
}

template<typename... Args>
void table_row(Args &&...cols) {
    (std::cout << ... << (std::setw(10) << std::forward<Args>(cols)));
    std::cout << '\n';
}

} // namespace Log
