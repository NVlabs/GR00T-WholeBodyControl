# Real-Robot Teleop Safety Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the PICO-driven controller-only teleop safe enough for first real-hardware runs by adding deploy-side filtering, manager-side watchdogs, mode-entry pose-mismatch gating, real-mode preflight, MuJoCo conflict checks, and an operator E-stop drill.

**Architecture:** Three layers compose:
1. **Deploy-side (C++ `g1_deploy_onnx_ref`)** filters VR_3PT inputs to the encoder for NaN/Inf, per-frame position/orientation step limits; escalates to e-stop after sustained violations.
2. **Manager-side (Python `pico_manager_thread_server.py`)** stamps every XR sample with `time.monotonic_ns()`, runs a 50 ms / 200 ms staleness watchdog, and blocks `PLANNER_VR_3PT` entry when the operator pose is too far from the robot's measured FK.
3. **Shell-side (`deploy.sh`, `preflight.sh`)** refuses real-mode launch when MuJoCo is running or the operator hasn't confirmed the preflight checklist.

**Tech Stack:** C++20 + GoogleTest + ZMQ + Unitree SDK2 (deploy); Python 3.10 + numpy + scipy + unittest/pytest-compatible tests (manager); Bash (deploy.sh, preflight.sh); Sphinx (docs).

**Out of scope for this plan:** LeRobot v2.1 data collection. The existing `gear_sonic/scripts/run_data_exporter.py` + `gear_sonic/data/features_sonic_vla.py` pipeline already supports the controller-only VR_3PT path with `teleop.vr_3pt_position` (shape (9,)) and `teleop.vr_3pt_orientation` (shape (18,), 6D rotation × 3 points). After Phase A lands, the operator can run that exporter in parallel with the pico_manager during real-robot sessions. A separate plan can capture any remaining gaps in that recorder if needed.

## 2026-05-26 Revision Notes Before Execution

The first draft had several implementation assumptions that are unsafe or stale
relative to the current repository. These corrections are binding for execution:

1. **A1 filter uses fixed arrays, not vectors.** `ZMQManager` and
   `InputInterface` use `std::array<double, 9>` for VR_3PT position and
   `std::array<double, 12>` for VR_3PT orientation. Implement
   `Vr3PtSafetyFilter` with those array types to avoid realtime-path
   allocation and type churn.
2. **A1 quaternion angle must normalize.** Do not compute rotation angle from a
   raw dot product unless both quaternions have first been normalized. The
   validity check alone is not enough.
3. **A1 stop escalation must feed the existing stop path.** When the filter
   escalates, `ZMQManager::handle_input()` must set `operator_state.stop = true`
   and clear planner/VR state consistently with the existing stop handling.
4. **A2 watchdog is armed, not immediately fatal.** `poll(None)` must not kill
   manager startup. The watchdog only escalates after a first valid XR sample
   has arrived and control/VR_3PT is active.
5. **A2 warning must not skip the whole input loop.** A 50 ms warning freezes
   VR_3PT target publication only. It must not `continue` past button-edge
   bookkeeping or manager-state publishing.
6. **A3 gate evaluates calibrated candidate target, not raw XR pose.** Raw
   headset/controller poses are not in the same FK frame as G1. Build a
   candidate calibrated target with the current `ThreePointPose` state, then
   compare that against measured FK.
7. **A3 gate must use real torso FK.** Extend `get_g1_ee_debug()` or add an FK
   helper so `"torso"` is available. Do not approximate torso as zero.
8. **A4 ramp symbols exist, but semantics matter.** Current ramp starts from
   the first post-recalibration VR sample, not robot FK. The no-jump guarantee is
   `entry gate -> recalibration -> ramp`; document and assert that ordering.
9. **A5 process matching should print exact matches and avoid narrow patterns.**
   Use `pgrep -af "run_sim_loop.py|mujoco|mujoco.viewer|simulate"` or a locally
   verified equivalent, and show the matched processes before aborting.
10. **A7 Unitree remote is recovery context only.** During GR00T low-level
    `rt/lowcmd` deploy, do not rely on `L1+A` or `L2+B` as an E-stop. Mention
    Unitree remote sequences only after our low-level deploy has stopped and the
    robot is being recovered under normal Unitree control.

**Revised execution order:** A1 deploy-side array filter -> A2 watchdog plus
combo-edge input fix -> A3 calibrated pose-entry gate with torso FK -> A4 chain
assertion -> A5/A6 preflight and MuJoCo conflict guard -> A7 docs -> A8 smoke
test.

**Execution progress:**

- [x] A1 deploy-side VR_3PT array filter implemented and wired into `ZMQManager`.
- [x] A2 manager-side XR staleness watchdog implemented; exact controller
  button-combo handling added.
- [x] A3 calibrated VR_3PT entry gate implemented against current G1 FK
  left-wrist/right-wrist positions, torso position, and wrist orientations.
- [x] A4 entry ordering assertion implemented: VR_3PT ramp cannot start before
  the entry gate passes and post-gate recalibration has run.
- [x] A5/A6 real-mode preflight and MuJoCo conflict guard implemented in
  `gear_sonic_deploy/scripts/preflight.sh` and called from `deploy.sh`.
- [x] A7 real-robot operator safety doc added.
- [ ] A8 smoke test on a live manager/deploy session remains to be run with the
  PICO + MuJoCo/real hardware process active.

---

## File Structure

**Deploy-side (C++) — new files:**
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/vr3pt_safety_filter.hpp` — `Vr3PtSafetyFilter` class (header-only, ~200 lines)
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_vr3pt_safety_filter.cpp` — GoogleTest unit tests
- `gear_sonic_deploy/scripts/preflight.sh` — interactive real-mode preflight (Bash)

**Deploy-side (C++) — modified files:**
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp` — wire the filter between message decode and `SetData(...)`; expose filter's e-stop flag through `handle_input`
- `gear_sonic_deploy/deploy.sh` — call `preflight.sh` in the `real` branch and reject if MuJoCo is running

**Manager-side (Python) — modified files:**
- `gear_sonic/scripts/pico_manager_thread_server.py` — add monotonic timestamp on each `ControllerPoseReader` sample; add `XRStalenessWatchdog`; add `Vr3PtEntryGate` invoked before `recalibrate_for_vr3pt()`; add an assertion that `gate → recalibrate → ramp` runs in order on every `PLANNER_VR_3PT` entry.

**Manager-side (Python) — new files:**
- `gear_sonic/tests/test_xr_staleness_watchdog.py` — unittest for the watchdog
- `gear_sonic/tests/test_vr3pt_entry_gate.py` — unittest for the entry gate

**Docs — new files:**
- `docs/source/user_guide/real_robot_safety.md` — operator drill, E-stop layers, preflight checklist (Sphinx-buildable)

**Docs — modified files:**
- `docs/source/user_guide/index.rst` (or equivalent toctree) — add `real_robot_safety` to the user-guide table of contents

---

## Task A1: Deploy-side VR_3PT safety filter

**Files:**
- Create: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/vr3pt_safety_filter.hpp`
- Create: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_vr3pt_safety_filter.cpp`
- Modify: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp` (insert filter call between message-decode and `SetData(...)` at lines ~1151-1195; add public `estop_requested()` accessor)

- [ ] **Step 1: Create the filter header with no-op behavior to compile**

Create `vr3pt_safety_filter.hpp`:

```cpp
/**
 * @file vr3pt_safety_filter.hpp
 * @brief Sanity filter for VR_3PT position/orientation streams entering the encoder.
 *
 * Rejects frames whose per-point step exceeds configurable thresholds, NaN/Inf,
 * or invalid quaternions; falls back to the last-good values. Escalates to an
 * e-stop request after a configurable consecutive-violation streak.
 *
 * Expected input layout:
 *   position    : 9 doubles = [l_xyz, r_xyz, head_xyz]
 *   orientation : 12 doubles = [l_wxyz, r_wxyz, head_wxyz]   (scalar-first quat)
 */
#ifndef VR3PT_SAFETY_FILTER_HPP
#define VR3PT_SAFETY_FILTER_HPP

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

class Vr3PtSafetyFilter {
 public:
  struct Config {
    double max_position_step_m = 0.03;          // 3 cm / frame / point
    double max_orientation_step_rad = M_PI / 36.0;  // 5 deg / frame / point
    int violation_streak_estop = 10;
  };

  struct Result {
    std::vector<double> position;     // size 9
    std::vector<double> orientation;  // size 12
    bool used_last_good = false;
    bool estop_triggered = false;
  };

  explicit Vr3PtSafetyFilter(Config cfg = {}) : cfg_(cfg) {}

  Result Filter(const std::vector<double>& new_pos,
                const std::vector<double>& new_orn) {
    Result r;
    r.position = new_pos;
    r.orientation = new_orn;
    r.estop_triggered = (violation_streak_ >= cfg_.violation_streak_estop);
    return r;
  }

  void Reset() {
    initialized_ = false;
    violation_streak_ = 0;
    last_pos_.clear();
    last_orn_.clear();
  }

 private:
  Config cfg_;
  std::vector<double> last_pos_;
  std::vector<double> last_orn_;
  bool initialized_ = false;
  int violation_streak_ = 0;
};

#endif  // VR3PT_SAFETY_FILTER_HPP
```

- [ ] **Step 2: Write the failing tests**

Create `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_vr3pt_safety_filter.cpp`:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include "input_interface/vr3pt_safety_filter.hpp"

static std::vector<double> Identity9() { return {0,0,0, 0,0,0, 0,0,0}; }
static std::vector<double> IdentityQuat12() {
  // scalar-first wxyz; identity = (1,0,0,0) per point
  return {1,0,0,0, 1,0,0,0, 1,0,0,0};
}

TEST(Vr3PtSafetyFilter, FirstFrameAlwaysAccepted) {
  Vr3PtSafetyFilter f;
  auto r = f.Filter(Identity9(), IdentityQuat12());
  EXPECT_FALSE(r.used_last_good);
  EXPECT_FALSE(r.estop_triggered);
}

TEST(Vr3PtSafetyFilter, SmallStepAccepted) {
  Vr3PtSafetyFilter f;
  f.Filter(Identity9(), IdentityQuat12());
  auto pos = Identity9();
  pos[0] = 0.01;  // 1 cm step on left wrist x
  auto r = f.Filter(pos, IdentityQuat12());
  EXPECT_FALSE(r.used_last_good);
}

TEST(Vr3PtSafetyFilter, LargePositionStepRejectedAndFrozen) {
  Vr3PtSafetyFilter f;
  f.Filter(Identity9(), IdentityQuat12());
  auto pos = Identity9();
  pos[0] = 0.30;  // 30 cm jump on left wrist x
  auto r = f.Filter(pos, IdentityQuat12());
  EXPECT_TRUE(r.used_last_good);
  EXPECT_DOUBLE_EQ(r.position[0], 0.0);
}

TEST(Vr3PtSafetyFilter, LargeQuatStepRejected) {
  Vr3PtSafetyFilter f;
  f.Filter(Identity9(), IdentityQuat12());
  // 90 deg rotation on left wrist about x: (cos45, sin45, 0, 0)
  auto orn = IdentityQuat12();
  orn[0] = std::cos(M_PI / 4);
  orn[1] = std::sin(M_PI / 4);
  auto r = f.Filter(Identity9(), orn);
  EXPECT_TRUE(r.used_last_good);
}

TEST(Vr3PtSafetyFilter, NaNRejected) {
  Vr3PtSafetyFilter f;
  f.Filter(Identity9(), IdentityQuat12());
  auto pos = Identity9();
  pos[3] = std::nan("");
  auto r = f.Filter(pos, IdentityQuat12());
  EXPECT_TRUE(r.used_last_good);
}

TEST(Vr3PtSafetyFilter, EStopAfterStreak) {
  Vr3PtSafetyFilter::Config cfg;
  cfg.violation_streak_estop = 3;
  Vr3PtSafetyFilter f(cfg);
  f.Filter(Identity9(), IdentityQuat12());
  auto bad = Identity9();
  bad[0] = 0.50;
  for (int i = 0; i < 3; ++i) f.Filter(bad, IdentityQuat12());
  auto r = f.Filter(bad, IdentityQuat12());
  EXPECT_TRUE(r.estop_triggered);
}

TEST(Vr3PtSafetyFilter, GoodFrameResetsStreak) {
  Vr3PtSafetyFilter::Config cfg;
  cfg.violation_streak_estop = 3;
  Vr3PtSafetyFilter f(cfg);
  f.Filter(Identity9(), IdentityQuat12());
  auto bad = Identity9();
  bad[0] = 0.50;
  f.Filter(bad, IdentityQuat12());
  f.Filter(bad, IdentityQuat12());
  auto good = Identity9();
  good[0] = 0.005;
  f.Filter(good, IdentityQuat12());
  auto r = f.Filter(bad, IdentityQuat12());
  EXPECT_FALSE(r.estop_triggered);
}
```

- [ ] **Step 3: Build and run tests — confirm they fail**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
just build
./src/g1/g1_deploy_onnx_ref/target/release/run_tests --gtest_filter="Vr3PtSafetyFilter.*"
```

Expected: 6 of 7 fail (`FirstFrameAlwaysAccepted` passes; the others fail because the no-op filter never sets `used_last_good` or `estop_triggered`).

- [ ] **Step 4: Implement the filter logic**

Replace `Filter()` and add helpers in `vr3pt_safety_filter.hpp`:

```cpp
  Result Filter(const std::vector<double>& new_pos,
                const std::vector<double>& new_orn) {
    Result r;
    if (new_pos.size() != 9 || new_orn.size() != 12 ||
        !AllFinite(new_pos) || !AllFinite(new_orn) ||
        !QuatsValid(new_orn)) {
      return EmitLastGood(r, "NaN/Inf/invalid-quat");
    }
    if (!initialized_) {
      last_pos_ = new_pos;
      last_orn_ = new_orn;
      initialized_ = true;
      r.position = new_pos;
      r.orientation = new_orn;
      return r;
    }
    if (MaxPositionStep(last_pos_, new_pos) > cfg_.max_position_step_m) {
      return EmitLastGood(r, "position-step");
    }
    if (MaxQuatAngle(last_orn_, new_orn) > cfg_.max_orientation_step_rad) {
      return EmitLastGood(r, "orientation-step");
    }
    last_pos_ = new_pos;
    last_orn_ = new_orn;
    violation_streak_ = 0;
    r.position = new_pos;
    r.orientation = new_orn;
    return r;
  }

 private:
  Result EmitLastGood(Result r, const char* reason) {
    ++violation_streak_;
    if (violation_streak_ % 5 == 1) {
      std::cerr << "[Vr3PtSafetyFilter] reject: " << reason
                << " streak=" << violation_streak_ << std::endl;
    }
    r.used_last_good = true;
    r.position = last_pos_.empty() ? std::vector<double>(9, 0.0) : last_pos_;
    r.orientation = last_orn_.empty() ? std::vector<double>{1,0,0,0,1,0,0,0,1,0,0,0} : last_orn_;
    r.estop_triggered = (violation_streak_ >= cfg_.violation_streak_estop);
    return r;
  }

  static bool AllFinite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
  }

  static bool QuatsValid(const std::vector<double>& q12) {
    for (int i = 0; i < 3; ++i) {
      double n = 0.0;
      for (int j = 0; j < 4; ++j) {
        double v = q12[i * 4 + j];
        n += v * v;
      }
      if (n < 0.25 || n > 4.0) return false;  // not anywhere near unit
    }
    return true;
  }

  static double MaxPositionStep(const std::vector<double>& a,
                                const std::vector<double>& b) {
    double max_d = 0.0;
    for (int i = 0; i < 3; ++i) {
      double dx = a[i*3+0] - b[i*3+0];
      double dy = a[i*3+1] - b[i*3+1];
      double dz = a[i*3+2] - b[i*3+2];
      max_d = std::max(max_d, std::sqrt(dx*dx + dy*dy + dz*dz));
    }
    return max_d;
  }

  static double MaxQuatAngle(const std::vector<double>& a,
                             const std::vector<double>& b) {
    double max_th = 0.0;
    for (int i = 0; i < 3; ++i) {
      // dot product of unit quats; |dot| = cos(theta/2)
      double dot = a[i*4+0]*b[i*4+0] + a[i*4+1]*b[i*4+1]
                 + a[i*4+2]*b[i*4+2] + a[i*4+3]*b[i*4+3];
      double abs_dot = std::min(1.0, std::abs(dot));
      double th = 2.0 * std::acos(abs_dot);
      max_th = std::max(max_th, th);
    }
    return max_th;
  }
```

- [ ] **Step 5: Build and run tests — confirm all pass**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
just build
./src/g1/g1_deploy_onnx_ref/target/release/run_tests --gtest_filter="Vr3PtSafetyFilter.*"
```

Expected: `[==========] 7 tests from Vr3PtSafetyFilter ... PASSED`.

- [ ] **Step 6: Wire the filter into ZMQManager**

In `zmq_manager.hpp`, add `#include "vr3pt_safety_filter.hpp"` to the include block (after the existing `input_command.hpp` include), and add a member + accessor inside the class (after `last_has_vr_3point_control_` at line 1254):

```cpp
    Vr3PtSafetyFilter vr3pt_filter_;
    bool vr3pt_estop_requested_ = false;
  public:
    bool vr3pt_estop_requested() const { return vr3pt_estop_requested_; }
```

Replace the `if (has_vr_position)` block at `zmq_manager.hpp:1153-1195` so the filter runs first:

```cpp
      if (has_vr_position) {
        auto filt = vr3pt_filter_.Filter(vr_position_values, vr_orientation_values);
        if (filt.estop_triggered) {
          vr3pt_estop_requested_ = true;
        }
        vr_position_values    = filt.position;
        vr_orientation_values = filt.orientation;

        vr_3point_position_.SetData(vr_position_values);
        vr_3point_orientation_.SetData(vr_orientation_values);
        if (has_vr_compliance) SetVR3PointCompliance(vr_compliance_values);
        has_vr_3point_control_ = true;

        pose_interface_->SetVR3PointPosition(vr_position_values);
        pose_interface_->SetVR3PointOrientation(vr_orientation_values);
        pose_interface_->SetVR3PointCompliance(vr_compliance_values);
      } else {
        has_vr_3point_control_ = false;
        vr3pt_filter_.Reset();
      }
```

Then in `ZMQManager::handle_input(...)` (find the existing override that fills `operator_state`), add at the top of the function body:

```cpp
      if (vr3pt_estop_requested_) {
        std::cerr << "[ZMQManager] VR_3PT filter streak escalation -> STOP" << std::endl;
        operator_state.stop = true;
      }
```

- [ ] **Step 7: Build + smoke-test the wired filter against the existing zmq test sender**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
just build
./src/g1/g1_deploy_onnx_ref/target/release/zmq_python_sender_test &
SENDER_PID=$!
sleep 2
./src/g1/g1_deploy_onnx_ref/target/release/zmq_pose_subscriber_test &
sleep 2
kill $SENDER_PID 2>/dev/null
```

Expected: no crashes, no e-stop triggered, no `[Vr3PtSafetyFilter] reject` messages (test sender produces well-behaved data).

- [ ] **Step 8: Commit**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git add gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/vr3pt_safety_filter.hpp \
        gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_vr3pt_safety_filter.cpp \
        gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp
git commit -m "$(cat <<'EOF'
feat(deploy): add VR_3PT safety filter ahead of encoder

Rejects NaN/Inf, invalid quaternions, position steps >3 cm/frame, or
orientation steps >5 deg/frame on the incoming VR_3PT stream; freezes
last-good values on rejection. Escalates to operator_state.stop after
10 consecutive violations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A2: XR staleness watchdog

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` (`ControllerPoseReader._run`, manager loop)
- Create: `gear_sonic/tests/test_xr_staleness_watchdog.py`

The current `ControllerPoseReader._run` does not stamp samples with a monotonic timestamp the manager loop can read (it sets `self._last_t` internally but the dict returned by `get_latest()` doesn't include it). Add a stamp first, then build the watchdog on top.

- [ ] **Step 1: Write the failing watchdog test**

Create `gear_sonic/tests/test_xr_staleness_watchdog.py`:

```python
import time
from unittest.mock import MagicMock

import pytest

from gear_sonic.scripts.pico_manager_thread_server import XRStalenessWatchdog


def test_watchdog_idle_with_fresh_sample():
    wd = XRStalenessWatchdog(warn_ms=50, estop_ms=200)
    now_ns = time.monotonic_ns()
    state = wd.poll(last_sample_ns=now_ns, now_ns=now_ns + 10_000_000)  # 10 ms gap
    assert state == "ok"


def test_watchdog_warns_between_thresholds():
    wd = XRStalenessWatchdog(warn_ms=50, estop_ms=200)
    now_ns = time.monotonic_ns()
    state = wd.poll(last_sample_ns=now_ns, now_ns=now_ns + 100_000_000)
    assert state == "warn"


def test_watchdog_estops_past_estop_threshold():
    wd = XRStalenessWatchdog(warn_ms=50, estop_ms=200)
    now_ns = time.monotonic_ns()
    state = wd.poll(last_sample_ns=now_ns, now_ns=now_ns + 300_000_000)
    assert state == "estop"


def test_watchdog_handles_none_sample():
    wd = XRStalenessWatchdog(warn_ms=50, estop_ms=200)
    state = wd.poll(last_sample_ns=None, now_ns=time.monotonic_ns())
    assert state == "estop"
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/pytest gear_sonic/tests/test_xr_staleness_watchdog.py -v
```

Expected: ImportError (`XRStalenessWatchdog` does not exist).

- [ ] **Step 3: Implement the watchdog class**

Add to `gear_sonic/scripts/pico_manager_thread_server.py` near the top of the file, right after the `StreamMode` enum (around line 133):

```python
class XRStalenessWatchdog:
    """Track XR-sample arrival latency, return 'ok' | 'warn' | 'estop'.

    Threshold semantics:
      gap < warn_ms              -> 'ok'
      warn_ms <= gap < estop_ms  -> 'warn'  (caller should freeze last target)
      gap >= estop_ms            -> 'estop' (caller should stop policy and exit)
    """

    def __init__(self, warn_ms: float = 50.0, estop_ms: float = 200.0):
        self.warn_ns = int(warn_ms * 1_000_000)
        self.estop_ns = int(estop_ms * 1_000_000)
        self._last_state = "ok"

    def poll(self, last_sample_ns: int | None, now_ns: int) -> str:
        if last_sample_ns is None:
            state = "estop"
        else:
            gap = now_ns - last_sample_ns
            if gap < self.warn_ns:
                state = "ok"
            elif gap < self.estop_ns:
                state = "warn"
            else:
                state = "estop"
        if state != self._last_state:
            print(f"[XRStalenessWatchdog] {self._last_state} -> {state}")
            self._last_state = state
        return state
```

- [ ] **Step 4: Stamp samples with monotonic_ns in ControllerPoseReader**

In `pico_manager_thread_server.py`, inside `ControllerPoseReader._run` at the place where `sample = { ... }` is built (~line 1114), insert a `"sample_monotonic_ns": time.monotonic_ns()` field. Add a public accessor on the reader:

```python
    def get_last_sample_monotonic_ns(self) -> int | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.get("sample_monotonic_ns")
```

- [ ] **Step 5: Run watchdog tests — confirm they pass**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/pytest gear_sonic/tests/test_xr_staleness_watchdog.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Wire the watchdog into the manager loop**

In `pico_manager_thread_server.py`, inside `run_pico_manager(...)` around the existing `while True:` loop (~line 2749), instantiate the watchdog before the loop and poll each iteration:

```python
        xr_watchdog = XRStalenessWatchdog(warn_ms=50.0, estop_ms=200.0)
```

Inside the loop body, after the existing `a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()` (~line 2751), add:

```python
            xr_state = xr_watchdog.poll(
                last_sample_ns=reader.get_last_sample_monotonic_ns(),
                now_ns=time.monotonic_ns(),
            )
            if xr_state == "estop" and current_mode != StreamMode.OFF:
                print("[Manager] XR staleness E-STOP: stopping policy and exiting")
                socket.send(build_command_message(start=False, stop=True, planner=True))
                exit()
            if xr_state == "warn" and current_mode == StreamMode.PLANNER_VR_3PT:
                # Freeze: skip this iteration's mode update so the last good target persists
                continue
```

- [ ] **Step 7: Smoke-test in sim**

In one terminal:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --input-type zmq_manager sim
```

In another:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --manager --controller_3pt \
  --controller_pose_convention openxr_unitree \
  --headset_pose_convention openxr_unitree \
  --headset_orientation_convention openxr_unitree
```

Briefly disconnect the headset (cover the cameras / power off). Expected: within 200 ms, manager prints `[Manager] XR staleness E-STOP` and exits, deploy receives a stop message.

- [ ] **Step 8: Commit**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git add gear_sonic/scripts/pico_manager_thread_server.py gear_sonic/tests/test_xr_staleness_watchdog.py
git commit -m "$(cat <<'EOF'
feat(pico_manager): add XR staleness watchdog (50/200 ms)

Stamps each ControllerPoseReader sample with monotonic_ns and polls in
the manager loop. >50 ms freezes the last VR_3PT target; >200 ms sends
stop and exits. Catches headset/controller tracking loss before it
becomes robot motion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A3: VR_3PT entry pose-mismatch gate

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` (add `Vr3PtEntryGate`; call before `recalibrate_for_vr3pt()` at line ~2897)
- Create: `gear_sonic/tests/test_vr3pt_entry_gate.py`

- [ ] **Step 1: Write failing test**

Create `gear_sonic/tests/test_vr3pt_entry_gate.py`:

```python
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.scripts.pico_manager_thread_server import Vr3PtEntryGate


def _identity_target():
    return np.array([
        [0.18, +0.20, 1.10, 1.0, 0.0, 0.0, 0.0],   # left wrist  (pos, wxyz)
        [0.18, -0.20, 1.10, 1.0, 0.0, 0.0, 0.0],   # right wrist
        [0.00,  0.00, 1.50, 1.0, 0.0, 0.0, 0.0],   # head/torso
    ], dtype=np.float32)


def _identity_fk():
    return {
        "left_wrist":  {"position": np.array([0.18, +0.20, 1.10]),
                        "orientation_wxyz": np.array([1.0, 0.0, 0.0, 0.0])},
        "right_wrist": {"position": np.array([0.18, -0.20, 1.10]),
                        "orientation_wxyz": np.array([1.0, 0.0, 0.0, 0.0])},
        "torso":       {"position": np.array([0.00,  0.00, 1.50]),
                        "orientation_wxyz": np.array([1.0, 0.0, 0.0, 0.0])},
    }


def test_gate_passes_when_aligned():
    gate = Vr3PtEntryGate()
    ok, reason = gate.evaluate(_identity_target(), _identity_fk())
    assert ok, f"expected OK, got reject: {reason}"


def test_gate_rejects_wrist_pos_mismatch():
    gate = Vr3PtEntryGate(wrist_pos_err_max_m=0.25)
    tgt = _identity_target()
    tgt[0, 0] += 0.40  # left wrist 40 cm forward of FK
    ok, reason = gate.evaluate(tgt, _identity_fk())
    assert not ok
    assert "wrist_pos" in reason


def test_gate_rejects_wrist_orn_mismatch():
    gate = Vr3PtEntryGate(wrist_orn_err_max_deg=45.0)
    tgt = _identity_target()
    tgt[0, 3:] = sRot.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
    ok, reason = gate.evaluate(tgt, _identity_fk())
    assert not ok
    assert "wrist_orn" in reason


def test_gate_rejects_head_pos_jump():
    gate = Vr3PtEntryGate(head_pos_jump_max_m=0.20)
    tgt = _identity_target()
    tgt[2, 0] += 0.40  # head 40 cm forward
    ok, reason = gate.evaluate(tgt, _identity_fk())
    assert not ok
    assert "head_pos" in reason
```

- [ ] **Step 2: Run — confirm failure**

```bash
.venv_teleop/bin/pytest gear_sonic/tests/test_vr3pt_entry_gate.py -v
```

Expected: ImportError on `Vr3PtEntryGate`.

- [ ] **Step 3: Implement the gate**

Add to `pico_manager_thread_server.py` near `XRStalenessWatchdog`:

```python
class Vr3PtEntryGate:
    """Reject PLANNER_VR_3PT entry if operator pose is too far from robot FK."""

    def __init__(
        self,
        wrist_pos_err_max_m: float = 0.25,
        wrist_orn_err_max_deg: float = 45.0,
        head_pos_jump_max_m: float = 0.20,
    ):
        self.wrist_pos_err_max_m = wrist_pos_err_max_m
        self.wrist_orn_err_max_rad = np.deg2rad(wrist_orn_err_max_deg)
        self.head_pos_jump_max_m = head_pos_jump_max_m

    def evaluate(
        self, vr_3pt_pose: np.ndarray, g1_fk: dict
    ) -> tuple[bool, str]:
        for i, key in enumerate(("left_wrist", "right_wrist")):
            pos_err = float(np.linalg.norm(vr_3pt_pose[i, :3] - g1_fk[key]["position"]))
            if pos_err > self.wrist_pos_err_max_m:
                return False, f"wrist_pos_err {key}={pos_err:.3f}m"
            q_vr = sRot.from_quat(vr_3pt_pose[i, 3:], scalar_first=True)
            q_fk = sRot.from_quat(g1_fk[key]["orientation_wxyz"], scalar_first=True)
            dq = q_fk.inv() * q_vr
            ang = float(dq.magnitude())
            if ang > self.wrist_orn_err_max_rad:
                return False, f"wrist_orn_err {key}={np.rad2deg(ang):.1f}deg"
        head_jump = float(
            np.linalg.norm(vr_3pt_pose[2, :3] - g1_fk["torso"]["position"])
        )
        if head_jump > self.head_pos_jump_max_m:
            return False, f"head_pos_err={head_jump:.3f}m"
        return True, "ok"
```

- [ ] **Step 4: Run — confirm passes**

```bash
.venv_teleop/bin/pytest gear_sonic/tests/test_vr3pt_entry_gate.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Wire the gate into the manager mode transition**

In `pico_manager_thread_server.py`, in the mode-transition block where `new_mode == StreamMode.PLANNER_VR_3PT` (around lines 2891-2898), wrap the recalibration call:

```python
                elif new_mode == StreamMode.PLANNER_VR_3PT:
                    # Pose-mismatch gate: refuse if operator and robot poses diverge
                    sample = reader.get_latest()
                    raw_vr_3pt_pose = (
                        sample["vr_3pt_pose_np"] if sample is not None else None
                    )
                    g1_fk = three_point.get_g1_ee_debug(
                        planner_streamer.feedback_reader.full_body_q_measured
                    )
                    if raw_vr_3pt_pose is not None and g1_fk:
                        gate = Vr3PtEntryGate()
                        # get_g1_ee_debug returns a different structure;
                        # adapt to {"left_wrist":..,"right_wrist":..,"torso":..}
                        fk_for_gate = {
                            "left_wrist":  {"position": np.array(g1_fk["left_wrist"]["position"]),
                                            "orientation_wxyz": np.array(g1_fk["left_wrist"]["orientation"]["quat_wxyz"])},
                            "right_wrist": {"position": np.array(g1_fk["right_wrist"]["position"]),
                                            "orientation_wxyz": np.array(g1_fk["right_wrist"]["orientation"]["quat_wxyz"])},
                            # Torso FK is not in get_g1_ee_debug; approximate as zero
                            # (head row in VR target is already calibrated relative
                            # to torso so this is a no-op for the relative jump check)
                            "torso": {"position": np.zeros(3),
                                      "orientation_wxyz": np.array([1.0, 0, 0, 0])},
                        }
                        ok, reason = gate.evaluate(raw_vr_3pt_pose, fk_for_gate)
                        if not ok:
                            print(f"[Manager] VR_3PT entry REFUSED: {reason}")
                            new_mode = current_mode
                            continue
                    if no_vr3pt_recalib_on_switch:
                        print("[Manager] Skipping VR_3PT per-switch recalibration")
                    else:
                        planner_streamer.recalibrate_for_vr3pt()
                    planner_streamer.start_vr3pt_ramp()
```

- [ ] **Step 6: Smoke-test in sim — gate rejects when arms-up vs robot-zero**

Start sim deploy as in A2. In manager, with operator standing arms-up (far from robot's neutral pose), press Left Stick Click. Expected console: `[Manager] VR_3PT entry REFUSED: wrist_pos_err ...`. Robot does not enter VR_3PT mode.

Then have operator lower arms to roughly match robot's neutral pose and press Left Stick Click again. Expected: transition succeeds.

- [ ] **Step 7: Commit**

```bash
git add gear_sonic/scripts/pico_manager_thread_server.py gear_sonic/tests/test_vr3pt_entry_gate.py
git commit -m "$(cat <<'EOF'
feat(pico_manager): gate VR_3PT entry on pose mismatch

Before transitioning to PLANNER_VR_3PT, compare the calibrated VR target
against the robot's measured FK. Refuse the transition if any wrist
position error >0.25 m, any wrist orientation error >45 deg, or the
head/torso translation jump exceeds 0.20 m. Prevents silent
recalibration through a violent mismatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A4: Verify gate → recalibrate → ramp chain ordering

**Goal:** This is an audit-and-assert task, not new behavior. We verified during planning that `start_vr3pt_ramp()` does NOT blend from FK→VR (its `_vr3pt_ramp_start_pose` is the first calibrated VR sample, not robot FK). The "no jump on entry" guarantee comes from `recalibrate_for_vr3pt()` re-anchoring offsets to measured q. The gate from A3 prevents large-mismatch entries; recalibration absorbs the small remaining mismatch; the ramp smooths the first second.

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` (add ordering assertion in `PLANNER_VR_3PT` transition path)
- Modify: `docs/source/user_guide/teleoperation.md` (one sentence noting the layered safety chain)

- [ ] **Step 1: Add a `_last_vr3pt_entry_chain` debug counter**

In `PlannerStreamer.__init__` (around line 2362), add:

```python
        self._chain_state = "idle"  # idle -> gate_passed -> recalibrated -> ramp_started
```

- [ ] **Step 2: Set the chain state at each step**

In the A3-modified `PLANNER_VR_3PT` block, set after the gate passes:

```python
                        planner_streamer._chain_state = "gate_passed"
```

In `recalibrate_for_vr3pt()` (line 2410), at end of method:

```python
        self._chain_state = "recalibrated"
```

In `start_vr3pt_ramp()` (line 2370), assert and set:

```python
    def start_vr3pt_ramp(self):
        assert self._chain_state in ("recalibrated", "idle"), (
            f"VR_3PT ramp started without prior recalibration "
            f"(chain={self._chain_state})"
        )
        if self.teleop_ramp_duration <= 0.0:
            ...
        ...
        self._chain_state = "ramp_started"
```

In `cancel_vr3pt_ramp()`:

```python
        self._chain_state = "idle"
```

- [ ] **Step 3: Re-run the A3 smoke test**

Repeat A3 Step 6. Expected: no AssertionError. If the assert fires, the ordering is wrong and must be fixed before proceeding.

- [ ] **Step 4: Update teleoperation.md with one paragraph**

In `docs/source/user_guide/teleoperation.md`, add a section heading "VR_3PT entry safety chain" with:

```markdown
## VR_3PT entry safety chain

On every `Left Stick Click` transition into `PLANNER_VR_3PT`, three things
run in order:

1. **Entry gate** (`Vr3PtEntryGate`) — refuses the transition if the
   operator's VR target diverges from the robot's measured FK by more than
   25 cm (wrist pos), 45° (wrist orn), or 20 cm (head). See
   `gear_sonic/scripts/pico_manager_thread_server.py`.
2. **Recalibration** (`recalibrate_for_vr3pt`) — re-anchors the VR
   calibration offsets to the robot's current measured joint state, so
   the next live target lines up with current FK.
3. **Ramp** (`start_vr3pt_ramp`) — cubic-eased blend from the first
   post-recalibration VR sample to live samples over 1 s (default).

A debug-only assertion verifies this ordering at runtime.
```

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/scripts/pico_manager_thread_server.py docs/source/user_guide/teleoperation.md
git commit -m "$(cat <<'EOF'
test(pico_manager): assert VR_3PT entry chain ordering

Adds a debug state machine that asserts gate -> recalibrate -> ramp
runs in order on every PLANNER_VR_3PT transition. Documents the
three-layer entry safety chain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A5: Real-mode preflight gate (interactive checklist)

**Files:**
- Create: `gear_sonic_deploy/scripts/preflight.sh`
- Modify: `gear_sonic_deploy/deploy.sh` (call preflight in the `real` branch)

- [ ] **Step 1: Create the preflight script**

Create `gear_sonic_deploy/scripts/preflight.sh`:

```bash
#!/usr/bin/env bash
# Interactive preflight for real-robot teleop launches.
# Each item must be answered y/Y to proceed. Any other response aborts.
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YEL='\033[1;33m'; NC='\033[0m'

ask() {
  local prompt="$1"
  local ans
  read -p "$(echo -e "${YEL}[preflight]${NC} ${prompt} [y/N]: ")" ans
  case "$ans" in
    [Yy]) return 0 ;;
    *)    echo -e "${RED}[preflight] aborted at: ${prompt}${NC}"; exit 1 ;;
  esac
}

echo -e "${GREEN}=== Real-robot teleop preflight ===${NC}"

# Auto-checks (no operator input)
if pgrep -f "run_sim_loop.py" >/dev/null 2>&1 || pgrep -f "/mujoco" >/dev/null 2>&1; then
  echo -e "${RED}[preflight] MuJoCo / run_sim_loop is running — kill it first.${NC}"
  pgrep -af "run_sim_loop.py|mujoco" || true
  exit 1
fi
echo -e "${GREEN}[preflight] no MuJoCo / run_sim_loop processes detected${NC}"

# Optional: latency to robot
if command -v ping >/dev/null 2>&1; then
  if ping -c 1 -W 1 192.168.123.161 >/dev/null 2>&1; then
    rtt=$(ping -c 5 -i 0.2 192.168.123.161 | tail -1 | awk -F'/' '{print $5}')
    echo -e "${GREEN}[preflight] avg RTT to G1 = ${rtt} ms${NC}"
  else
    echo -e "${YEL}[preflight] cannot ping 192.168.123.161 — skipping latency check${NC}"
  fi
fi

# Operator-confirmed checklist
ask "Harness or support frame in place and load-tested?"
ask "3 m clearance around robot verified (no people, cables, furniture)?"
ask "Designated keyboard operator at deploy terminal, finger near 'O'?"
ask "Designated PICO operator briefed on A+B+X+Y stop?"
ask "Spotter assigned within arm's reach of robot?"
ask "Operator wearing tight-fitting clothing (no loose pants/sleeves)?"
ask "Operator standing in all-zero calibration pose (feet together, arms down, looking forward)?"
ask "PICO tracking healthy in the headset display?"

echo -e "${GREEN}=== Preflight passed ===${NC}"
```

```bash
chmod +x /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy/scripts/preflight.sh
```

- [ ] **Step 2: Modify deploy.sh to call preflight in the real branch**

In `gear_sonic_deploy/deploy.sh`, find the `# Ask for confirmation` block (~line 405) and insert *before* it:

```bash
# ============================================================================
# Real-mode preflight gate (no-op for sim)
# ============================================================================

if [[ "$ENV_TYPE" == "real" ]]; then
    PREFLIGHT_SH="$SCRIPT_DIR/scripts/preflight.sh"
    if [[ ! -x "$PREFLIGHT_SH" ]]; then
        echo -e "${RED}❌ Preflight script missing or not executable: $PREFLIGHT_SH${NC}"
        exit 1
    fi
    if ! "$PREFLIGHT_SH"; then
        echo -e "${RED}Preflight failed — aborting deploy.${NC}"
        exit 1
    fi
fi
```

- [ ] **Step 3: Smoke-test preflight in sim path (should skip)**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
echo "n" | ./deploy.sh sim   # 'n' to the existing "Proceed?" prompt
```

Expected: no preflight prompts, normal sim-path output, exits cleanly at the final prompt.

- [ ] **Step 4: Smoke-test preflight in real path (should prompt)**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
echo "n" | ./deploy.sh real
```

Expected: preflight script runs, MuJoCo auto-check passes (assuming no sim running), first interactive prompt appears. Press Ctrl-C or "n" to abort. Confirm deploy aborts before running anything else.

- [ ] **Step 5: Smoke-test MuJoCo conflict detection (overlaps with A6)**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py &
SIM_PID=$!
sleep 2
cd gear_sonic_deploy
./deploy.sh real
RESULT=$?
kill $SIM_PID
test "$RESULT" -ne 0 && echo "OK: deploy aborted because sim running"
```

Expected: deploy aborts with `MuJoCo / run_sim_loop is running` message; final `echo` prints the OK line.

- [ ] **Step 6: Commit**

```bash
git add gear_sonic_deploy/scripts/preflight.sh gear_sonic_deploy/deploy.sh
git commit -m "$(cat <<'EOF'
feat(deploy): interactive real-mode preflight + MuJoCo conflict guard

Adds gear_sonic_deploy/scripts/preflight.sh that runs before any real
deploy launch: auto-detects running MuJoCo/run_sim_loop processes,
checks robot RTT, and walks the operator through harness, clearance,
operator/spotter assignment, clothing, and calibration-pose checks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A6: MuJoCo conflict check (already in A5 preflight)

The MuJoCo conflict check is implemented in `preflight.sh` from A5 (`pgrep -f run_sim_loop.py` and `pgrep -f /mujoco`). This task is a verification placeholder to avoid losing the spec item.

- [ ] **Step 1: Verify the check fires on the right processes**

```bash
# Should be present in preflight.sh from A5
grep -n "pgrep -f.*run_sim_loop\|pgrep -f.*mujoco" \
  /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy/scripts/preflight.sh
```

Expected: at least one match.

- [ ] **Step 2: Verify spec coverage by walking through scenarios**

Document in this checklist (no commit needed):

| Process running | preflight result |
|---|---|
| `gear_sonic/scripts/run_sim_loop.py` | aborts with MuJoCo message |
| MuJoCo viewer launched separately (`mujoco_py`, etc.) | aborts (matches `/mujoco`) |
| Nothing running | passes auto-check, proceeds to interactive prompts |

If the second row does not match a process that should be caught (e.g. the team's MuJoCo viewer has a different name), extend the `pgrep` patterns in `preflight.sh` and amend Task A5's commit (or add a follow-up commit).

---

## Task A7: Operator drill docs (Sphinx)

**Files:**
- Create: `docs/source/user_guide/real_robot_safety.md`
- Modify: `docs/source/user_guide/index.rst` (or the relevant toctree) to add `real_robot_safety` to the user-guide TOC

- [ ] **Step 1: Identify the user-guide toctree**

```bash
grep -rn "teleoperation\|toctree" /home/jihun/work/GR00T-WholeBodyControl/docs/source/user_guide/ \
  /home/jihun/work/GR00T-WholeBodyControl/docs/source/index.rst 2>/dev/null
```

Expected: a `.. toctree::` block listing the existing user-guide pages (`teleoperation`, `training`, etc.). Note the file containing it.

- [ ] **Step 2: Write the operator drill doc**

Create `docs/source/user_guide/real_robot_safety.md`:

```markdown
# Real-Robot Safety and E-Stop Drill

Before any real-hardware teleop session, the operator team **must**
complete this drill in simulation first. This page is the load-bearing
reference for what each safety layer does, and what operators are
expected to know and rehearse.

## Why no Unitree-remote E-stop

The GR00T deploy stack publishes joint commands to the Unitree
low-level DDS topic (`rt/lowcmd`). The G1 user manual explicitly
notes that *"remote control commands fail during low-level
development"* — pressing **L1+A** or **L2+B** on the handheld remote
will not bring the robot into damping mode while our deploy is
publishing lowcmd. **Do not rely on the Unitree remote as an E-stop.**

## E-stop layers (in order of preference)

| Layer | Trigger | What it does | Latency |
|---|---|---|---|
| 1. Keyboard | `O` (or `o`) in the deploy terminal | Sets `operator_state.stop` on next deploy loop tick | ~1 deploy loop period (~5 ms) |
| 2. PICO | `A + B + X + Y` together on either controller | Manager sends `{start:false, stop:true, planner:true}` and exits | ~50 ms (manager loop) |
| 3. Hardware | Harness / spotter catch / battery power-off | Physical containment of a falling robot | Depends on harness |

The keyboard and PICO E-stops are **software** stops: they assume the
respective process is alive and responsive. The hardware layer is the
only one that survives a software fault in this stack.

## Software safeguards already active

- `Vr3PtSafetyFilter` (deploy-side) — rejects NaN/Inf, position jumps
  >3 cm/frame, orientation jumps >5°/frame; escalates to E-stop after
  10 consecutive violations.
- `XRStalenessWatchdog` (manager-side) — >50 ms freezes the last good
  target; >200 ms sends stop and exits the manager.
- `Vr3PtEntryGate` (manager-side) — refuses `PLANNER_VR_3PT` entry if
  wrist position error >25 cm, wrist orientation error >45°, or head
  position jump >20 cm.

## Drill — required before every real-hardware session

Each operator runs each E-stop at least once with the robot moving in
**simulation**:

1. **Keyboard 'O' drill (3×)**: Bring deploy into PLANNER_VR_3PT
   teleop. Operator at the deploy terminal presses `O`. Verify the
   simulated robot freezes within one frame.
2. **PICO A+B+X+Y drill (3×)**: From PLANNER_VR_3PT, the PICO operator
   presses all four face buttons together. Verify the manager logs
   `[Manager] StreamMode switch: PLANNER_VR_3PT -> OFF` and exits.
3. **XR loss drill (1×)**: From PLANNER_VR_3PT, cover the PICO
   cameras or briefly power the headset off. Verify the manager logs
   `[Manager] XR staleness E-STOP` within 250 ms.
4. **Entry-gate drill (1×)**: Stand with arms above head far from the
   simulated robot's neutral pose. Press Left Stick Click. Verify the
   manager logs `[Manager] VR_3PT entry REFUSED: wrist_pos_err ...`.

If any drill step fails, the session does not proceed to real
hardware.

## Real-mode preflight checklist

The `./deploy.sh real` path now runs
`gear_sonic_deploy/scripts/preflight.sh` automatically. It refuses to
proceed unless:

- No `run_sim_loop.py` or MuJoCo processes are running on this host.
- Harness/support frame is in place and load-tested.
- 3 m clearance verified (people, cables, furniture).
- Designated keyboard operator at deploy terminal.
- Designated PICO operator briefed on A+B+X+Y.
- Spotter within arm's reach of robot.
- Operator in tight-fitting clothing.
- Operator standing in all-zero calibration pose (feet together, arms
  down, looking forward).
- PICO tracking healthy in headset display.

## Environment requirements (Unitree G1 User Manual V1.0)

The robot's own requirements are non-negotiable:

- Run only at **0–40 °C**, indoors, no inclement weather.
- ≥ **2 m** from obstacles, complex terrain, crowds, water, complex
  surfaces.
- Operate on stable, non-slippery, **non-gravel**, **non-icy** floors.
- Keep robot in **line of sight** throughout the session.
- Avoid Wi-Fi and electromagnetic interference.
- Confirm batteries and remote control are charged.

## Recovery after a stop

After any E-stop, do **not** simply restart the policy. Walk through:

1. Remove power if the robot is on the ground.
2. Re-seat batteries and verify connectors.
3. Hang the robot on the harness and run the Unitree startup sequence
   (`L1+A` damping → `L1+UP` ready → `R1+X`/`R2+X` operate).
4. Re-run the preflight (`./deploy.sh real`).
5. Re-run all four drill steps in sim before re-engaging real hardware.
```

- [ ] **Step 3: Add to the toctree**

In the file identified in Step 1 (likely `docs/source/user_guide/index.rst` or `docs/source/index.rst`), add `real_robot_safety` to the user-guide `..  toctree::` block, e.g. after `teleoperation`:

```rst
.. toctree::
   :maxdepth: 2

   user_guide/teleoperation
   user_guide/real_robot_safety
   user_guide/configuration
   user_guide/training
```

(Adapt to the exact toctree format the repo uses — RST or MyST.)

- [ ] **Step 4: Build docs locally to verify the page renders**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/docs
make html 2>&1 | tail -20
```

Expected: no warnings about `real_robot_safety` not found in toctree; HTML built without errors. Open `docs/build/html/user_guide/real_robot_safety.html` to eyeball formatting.

- [ ] **Step 5: Commit**

```bash
git add docs/source/user_guide/real_robot_safety.md docs/source/user_guide/index.rst
# (or whichever toctree file you actually edited)
git commit -m "$(cat <<'EOF'
docs(safety): add real-robot safety + E-stop drill page

Documents why the Unitree handheld remote is not usable as an E-stop
in our low-level deploy, lists the three remaining E-stop layers
(keyboard 'O', PICO A+B+X+Y, hardware harness), and codifies the
mandatory pre-session drill plus environment requirements from the
G1 User Manual V1.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task A8: End-to-end safety smoke test in sim

**Goal:** A single Python script that exercises every Phase A safeguard against a running sim, prints a per-item pass/fail summary, and exits non-zero on any failure. Used as the gate before any real-hardware session.

**Files:**
- Create: `gear_sonic/tests/test_safety_smoke.py`

- [ ] **Step 1: Write the smoke test scaffold**

Create `gear_sonic/tests/test_safety_smoke.py`:

```python
"""End-to-end safety smoke test for the PICO controller-only teleop path.

Run *after* ./deploy.sh sim is started and the pico_manager is running.
Performs each of the four Phase A drill items as automatic / semi-automatic
checks; prints a summary and exits non-zero on any failure.

Usage:
    .venv_teleop/bin/pytest gear_sonic/tests/test_safety_smoke.py -s
"""

import os
import subprocess
import time

import pytest


def _proc_running(pattern: str) -> bool:
    return subprocess.call(
        ["pgrep", "-f", pattern], stdout=subprocess.DEVNULL
    ) == 0


@pytest.mark.skipif(
    not _proc_running("g1_deploy_onnx_ref") or not _proc_running("pico_manager_thread_server"),
    reason="deploy and pico_manager must both be running for this smoke test",
)
def test_filter_rejects_nan(monkeypatch):
    """Inject a NaN VR_3PT frame via ZMQ and verify the filter logs a reject.

    Requires the test ZMQ sender to be wired against the same port the deploy
    subscribes to. Detailed setup beyond the scope of this skeleton.
    """
    pytest.skip("TODO once a programmatic ZMQ injector exists for this repo")


@pytest.mark.skipif(
    not _proc_running("pico_manager_thread_server"),
    reason="pico_manager must be running",
)
def test_xr_staleness_estop_window():
    """Walk through the drill manually; operator confirms log output.

    This is a manual gate: prints instructions and waits for keyboard
    confirmation. Use only when the smoke test is being run as part of the
    pre-session drill, not in CI.
    """
    print("\n[smoke] Cover PICO cameras for 1 s, then uncover.")
    print("[smoke] Expected log within 250 ms: '[Manager] XR staleness E-STOP'")
    ans = input("[smoke] Did the manager E-stop and exit? [y/N]: ").strip().lower()
    assert ans == "y", "XR staleness drill failed (or skipped)"
```

- [ ] **Step 2: Run it (with deploy + manager running in sim) and confirm at least the skipped tests register**

In one terminal:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh sim
```

In another:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --manager --controller_3pt \
  --controller_pose_convention openxr_unitree \
  --headset_pose_convention openxr_unitree \
  --headset_orientation_convention openxr_unitree &
sleep 5
.venv_teleop/bin/pytest gear_sonic/tests/test_safety_smoke.py -s
```

Expected: the NaN test skips with `TODO`; the XR staleness test prompts you. Either confirm `y` after performing the drill, or answer `n` to make it fail. Iterate until the drill passes.

- [ ] **Step 3: Commit**

```bash
git add gear_sonic/tests/test_safety_smoke.py
git commit -m "$(cat <<'EOF'
test(safety): add pre-session smoke test scaffold

Manual / semi-automatic drill checks for: VR_3PT filter NaN rejection
(skipped until programmatic ZMQ injector lands), XR staleness E-stop
window. Intended to be run alongside ./deploy.sh sim before every
real-hardware session.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**1. Spec coverage check:**

| Agent spec item | Plan task | Notes |
|---|---|---|
| 1. Deploy-side VR_3PT delta clamp | A1 | Filter + GTest + zmq_manager wiring; covers NaN, pos step, orn step, streak→stop |
| 2. XR staleness watchdog | A2 | Implements 50/200 ms thresholds; needed sample-timestamp prerequisite, included in A2 step 4 |
| 3. VR_3PT entry pose-mismatch gate | A3 | Implements 0.25 m / 45° / 0.20 m thresholds; uses existing `get_g1_ee_debug` |
| 4. Startup ramp verification | A4 | Reframed as ordering-assertion task; matches verified reality that ramp ≠ FK→VR blend |
| 5. Real-mode preflight | A5 | Interactive Bash checklist, called from deploy.sh real path |
| 6. MuJoCo conflict check | A6 | Folded into A5's preflight as auto-check; A6 is verification |
| 7. E-stop drill / docs | A7 | Sphinx page `real_robot_safety.md` + toctree entry; covers "no Unitree-remote" rationale |
| (implicit) E2E smoke test | A8 | Pytest scaffold for pre-session drill |

All seven agent spec items have a task.

**2. Placeholder scan:** Searched for "TBD", "TODO", "fill in", "implement later" in the body of each task. One legitimate `pytest.skip("TODO ...")` remains in A8 step 1 because the programmatic ZMQ injector for NaN-frame testing is out of scope; it's labeled as TODO inside the test code, not as plan-level handwaving. Acceptable.

**3. Type consistency check:**

| Symbol | First defined | Used in |
|---|---|---|
| `Vr3PtSafetyFilter` (C++) | A1 step 1 | A1 steps 4–6 |
| `XRStalenessWatchdog.poll(last_sample_ns, now_ns) -> str` | A2 step 3 | A2 step 6 (manager loop call matches signature) |
| `Vr3PtEntryGate.evaluate(vr_3pt_pose, g1_fk) -> (bool, str)` | A3 step 3 | A3 step 5 (manager wiring matches return type) |
| `_chain_state` member | A4 step 1 | A4 steps 2 (set/assert) |
| `preflight.sh` path | A5 step 1 | A5 step 2 (deploy.sh references same absolute path) |

No mismatches.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-real-robot-teleop-safety-stack.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Pairs well with the priority order A1 → A2 → A3 → A4 → A5 → A6 → A7 → A8.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints for review.

Which approach?
