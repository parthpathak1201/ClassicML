#pragma once

namespace cml {

// Library-wide console verbosity levels (SILENT, BASIC, DEBUG).
enum class Verbosity { SILENT, BASIC, DEBUG };

// Global log level — set via classicml.h before training.
inline Verbosity verbosity = Verbosity::BASIC;

} // namespace cml
