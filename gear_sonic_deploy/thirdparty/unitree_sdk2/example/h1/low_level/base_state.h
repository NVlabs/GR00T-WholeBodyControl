#pragma once

#include <array>

struct BaseState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
};
