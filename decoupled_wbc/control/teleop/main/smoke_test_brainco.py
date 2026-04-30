"""Stage 0 / Stage 1 smoke test for the brainco hand pipeline.

Hardware-free. Runs three checks in order, fails loudly on the first to break:

  1. Imports — every module the runtime needs is loadable.
  2. Asset wiring — dex_retargeting can build BraincoHandRetargeting from the
     in-tree URDFs and YAML.
  3. Bridge math — synthetic xrobotoolkit (26, 7) input flows through
     hand_state_to_unitree_keypoints to produce a finite, wrist-anchored
     (25, 3) keypoint cloud.

Run BEFORE plugging in a headset:

    python -m decoupled_wbc.control.teleop.main.smoke_test_brainco

Exit code 0 on full pass; non-zero on any failure.
"""

import sys
import traceback

import numpy as np


def _hr(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def stage0_imports():
    """Probe imports. We classify failures as either:

      - 'critical' (the brainco source files themselves don't parse / import)
      - 'runtime-dep' (the source files load but a third-party dep — dex_retargeting,
        unitree_sdk2py, xrobotoolkit_sdk — is unavailable; only matters on the
        deployment box).

    On a dev machine without the teleop venv, runtime-dep failures are expected
    and don't break the smoke test. On a deployment box they should all pass.
    """
    _hr("Stage 0: imports")

    runtime_deps = {"dex_retargeting", "unitree_sdk2py", "xrobotoolkit_sdk"}
    critical_failures = []
    runtime_dep_failures = []

    def _try(label, fn):
        try:
            fn()
            print(f"  [OK]   {label}")
        except ModuleNotFoundError as e:
            missing = (e.name or "").split(".")[0]
            if missing in runtime_deps:
                print(f"  [DEP]  {label}: missing third-party dep {missing!r}")
                runtime_dep_failures.append((label, missing))
            else:
                print(f"  [FAIL] {label}: ModuleNotFoundError: {e}")
                critical_failures.append(label)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
            critical_failures.append(label)

    _try(
        "decoupled_wbc...brainco.brainco_bridge",
        lambda: __import__(
            "decoupled_wbc.control.teleop.device.pico.brainco.brainco_bridge",
            fromlist=["hand_state_to_unitree_keypoints"],
        ),
    )
    _try(
        "decoupled_wbc...brainco.hand_retargeting",
        lambda: __import__(
            "decoupled_wbc.control.teleop.device.pico.brainco.hand_retargeting",
            fromlist=["BraincoHandRetargeting"],
        ),
    )
    _try(
        "decoupled_wbc...brainco.robot_hand_brainco",
        lambda: __import__(
            "decoupled_wbc.control.teleop.device.pico.brainco.robot_hand_brainco",
            fromlist=["BraincoController"],
        ),
    )
    _try("xrobotoolkit_sdk", lambda: __import__("xrobotoolkit_sdk"))

    if critical_failures:
        print(f"\n  -> Critical import failures: {critical_failures}")
        return False

    if runtime_dep_failures:
        missing = sorted({m for _, m in runtime_dep_failures})
        print(f"\n  -> Runtime deps missing (deploy-blocking, OK on dev): {missing}")
        print("     Install via:  bash install_scripts/install_pico.sh")
    return True


def stage1_assets():
    _hr("Stage 1a: dex_retargeting + URDF + YAML wiring")

    try:
        from decoupled_wbc.control.teleop.device.pico.brainco.hand_retargeting import (
            BraincoHandRetargeting,
        )
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "dex_retargeting":
            print(
                "  [SKIP] dex_retargeting not installed. "
                "Run `pip install dex-retargeting` to enable this stage."
            )
            return True
        print(f"  [FAIL] import BraincoHandRetargeting: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] import BraincoHandRetargeting: {e}")
        return False

    try:
        retargeter = BraincoHandRetargeting()
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] BraincoHandRetargeting() raised: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print(f"  [OK] left_joint_names  ({len(retargeter.left_joint_names)}): {retargeter.left_joint_names}")
    print(f"  [OK] right_joint_names ({len(retargeter.right_joint_names)})")
    print(f"  [OK] left_dex_to_hardware:  {retargeter.left_dex_to_hardware}")
    print(f"  [OK] right_dex_to_hardware: {retargeter.right_dex_to_hardware}")
    print(f"  [OK] left_indices.shape:    {retargeter.left_indices.shape}")

    expected_motor_count = 6
    if len(retargeter.left_dex_to_hardware) != expected_motor_count:
        print(
            f"  [FAIL] expected {expected_motor_count} brainco motors per hand, "
            f"got {len(retargeter.left_dex_to_hardware)}"
        )
        return False

    return True


def stage1_bridge():
    _hr("Stage 1b: bridge math on synthetic input")

    from decoupled_wbc.control.teleop.device.pico.brainco.brainco_bridge import (
        NUM_KEEP_JOINTS,
        NUM_RAW_JOINTS,
        WRIST_INDEX_AFTER_DROP,
        hand_state_to_unitree_keypoints,
        make_brainco_shared_arrays,
        push_keypoints_to_shared,
    )

    # Synthetic state: identity rotation everywhere, joints stretched along +X.
    state = np.zeros((NUM_RAW_JOINTS, 7))
    state[:, 0] = np.arange(NUM_RAW_JOINTS) * 0.02  # 0 cm, 2 cm, 4 cm, ...
    state[:, 6] = 1.0  # qw = 1

    out = hand_state_to_unitree_keypoints(state)

    if out.shape != (NUM_KEEP_JOINTS, 3):
        print(f"  [FAIL] keypoints shape: {out.shape} != ({NUM_KEEP_JOINTS}, 3)")
        return False
    print(f"  [OK] keypoints shape: {out.shape}")

    if not np.all(np.isfinite(out)):
        print("  [FAIL] keypoints contain NaN/Inf")
        return False
    print("  [OK] all entries finite")

    if not np.allclose(out[WRIST_INDEX_AFTER_DROP], 0.0):
        print(f"  [FAIL] wrist (kept index 0) not at origin: {out[WRIST_INDEX_AFTER_DROP]}")
        return False
    print("  [OK] wrist anchored at origin")

    # Sanity: index fingertip (kept index 9) magnitude in unitree frame should
    # match its world-frame distance from wrist (rigid transform = isometry).
    expected = np.linalg.norm(state[10, :3] - state[1, :3])  # raw 10 (kept 9) - raw 1 (kept 0)
    actual = np.linalg.norm(out[9])
    if not np.isclose(expected, actual):
        print(f"  [FAIL] index fingertip distance: expected {expected:.4f} m, got {actual:.4f} m")
        return False
    print(f"  [OK] index fingertip distance preserved: {actual:.4f} m")

    # Shared-array roundtrip
    left, right = make_brainco_shared_arrays()
    push_keypoints_to_shared(left, out)
    if not np.allclose(np.array(left[:]).reshape(25, 3), out):
        print("  [FAIL] shared-array roundtrip mismatch")
        return False
    print("  [OK] shared-array roundtrip")

    return True


def main():
    print("Brainco hand pipeline smoke test (hardware-free).")

    results = {
        "imports": stage0_imports(),
        "assets":  stage1_assets(),
        "bridge":  stage1_bridge(),
    }

    _hr("Summary")
    for name, ok in results.items():
        marker = "PASS" if ok else "FAIL"
        print(f"  {marker:4s}  {name}")

    if all(results.values()):
        print("\nAll stages passed. Safe to proceed to Stage 2 (PICO connected, brainco off).")
        return 0
    else:
        print("\nFix failures above before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
