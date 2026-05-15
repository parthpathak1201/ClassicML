// Shared library translation unit for ClassicML (Python bridge / linking).
#include "classicml.h"

namespace cml {
// Ensures the shared library exports at least one symbol.
void lib_anchor() {}
}
