#include <cassert>
#include <limits>

#include "inspire/contract.hpp"

int main() {
  inspire::HandVector values{0.F, 0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
  assert(inspire::finite(values));
  for (std::size_t i = 0; i < inspire::kHandDof; ++i) {
    const float saved = values[i];
    values[i] = std::numeric_limits<float>::quiet_NaN();
    assert(!inspire::finite(values));
    values[i] = std::numeric_limits<float>::infinity();
    assert(!inspire::finite(values));
    values[i] = -std::numeric_limits<float>::infinity();
    assert(!inspire::finite(values));
    values[i] = saved;
  }
  assert(inspire::finite(values));
  return 0;
}
