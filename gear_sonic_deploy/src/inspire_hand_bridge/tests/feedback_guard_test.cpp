#include <cassert>

#include "inspire/feedback_guard.hpp"

int main() {
  inspire::HandVector measured{0.1F, 0.2F, 0.3F, 0.4F, 0.2F, 0.5F};
  inspire::HandVector left = measured;
  inspire::HandVector right = measured;
  inspire::HandVector tolerance{0.06F, 0.06F, 0.06F, 0.06F, 0.06F, 0.06F};

  left[0] += 0.02F;
  left[1] -= 0.03F;
  left[2] += 0.04F;
  left[3] -= 0.05F;
  left[4] += 0.02F;
  left[5] -= 0.04F;
  assert(!inspire::feedback_violation(left, right, measured, measured, tolerance));

  right[4] += 0.061F;
  const auto violation =
      inspire::feedback_violation(left, right, measured, measured, tolerance);
  assert(violation);
  assert(!violation->left);
  assert(violation->index == 4);
  assert(violation->delta > violation->limit);
  return 0;
}
