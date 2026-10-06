#include <cmath>
#include <stdexcept>
#include <string>

#include "inspire/real_hardware_codec.hpp"

namespace {

bool close(float left, float right) { return std::abs(left - right) < 1e-6F; }

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  inspire::HandVector closed_normalized{};
  const auto closed = inspire::real::normalized_to_canonical(closed_normalized);
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    require(close(closed[index], inspire::real::kCanonicalHardwareClosed[index]),
            "normalized zero did not map to the canonical hardware closed endpoint");
  }

  inspire::HandVector open_normalized{};
  open_normalized.fill(1.F);
  const auto open = inspire::real::normalized_to_canonical(open_normalized);
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    require(close(open[index], inspire::real::kCanonicalHardwareOpen[index]),
            "normalized one did not map to the canonical hardware open endpoint");
  }
  const inspire::HandVector lab_closed{1.7F, 1.7F, 1.7F, 1.7F, 0.6F, 1.3F};
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    require(close(open[index], 0.F), "lab hardware open endpoint must be canonical zero");
    require(close(closed[index], lab_closed[index]),
            "canonical closed endpoint differs from the lab hardware interface");
  }

  inspire::HandVector sample_normalized{0.F, 0.25F, 0.5F, 0.75F, 1.F, 0.755F};
  const auto canonical = inspire::real::normalized_to_canonical(sample_normalized);
  const auto round_trip = inspire::real::canonical_to_normalized(canonical);
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    require(close(sample_normalized[index], round_trip[index]),
            "normalized/canonical round trip changed a value");
  }

  inspire::HandVector normalized_velocity{1.F, -1.F, 0.5F, -0.5F, 1.F, -1.F};
  const auto canonical_velocity =
      inspire::real::normalized_velocity_to_canonical(normalized_velocity);
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    const float expected =
        -normalized_velocity[index] * (inspire::real::kCanonicalHardwareClosed[index] -
                                       inspire::real::kCanonicalHardwareOpen[index]);
    require(close(canonical_velocity[index], expected),
            "normalized velocity conversion has the wrong scale or sign");
  }

  bool rejected = false;
  try {
    auto invalid = sample_normalized;
    invalid[2] = 1.01F;
    (void)inspire::real::normalized_to_canonical(invalid);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "out-of-range normalized state was not rejected");

  rejected = false;
  try {
    auto below_lab_endpoint = open;
    below_lab_endpoint[5] = -0.01F;
    (void)inspire::real::canonical_to_normalized(below_lab_endpoint);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "real codec accepted a thumb command below the hardware open endpoint");

  inspire::HandVector measured{};
  inspire::HandVector target{};
  target[2] = 0.01F;
  require(inspire::real::within_canonical_tolerance(target, measured, 0.01F),
          "hold gate rejected its exact tolerance boundary");
  target[2] = 0.0101F;
  require(!inspire::real::within_canonical_tolerance(target, measured, 0.01F),
          "hold gate accepted a displacement above tolerance");
  return 0;
}
