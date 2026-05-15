#pragma once

namespace cml {

enum class Verbosity { SILENT, BASIC, DEBUG };
inline Verbosity g_verbosity = Verbosity::BASIC;

} // namespace cml
