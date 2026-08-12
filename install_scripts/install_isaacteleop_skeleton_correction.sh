#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Make the isaacteleop skeleton correction visible to .venv_teleop.
#
# --skeleton-source quest (and the quest branch of auto) call
# isaacteleop.retargeting_engine.utilities.correct_body_orientations. That
# function is added by NVIDIA/IsaacTeleop#921 and ships in isaacteleop 1.5+,
# but .venv_teleop pins 1.3.131. Reinstalling the venv against 1.5 would drag in
# a C++ rebuild and a minor-version jump, so instead this bridges the one
# module across.
#
# TEMPORARY. Delete this script, and re-create .venv_teleop from a real 1.5+
# isaacteleop, once #921 has merged and GR00T-WholeBodyControl has been moved to
# that version.
#
# Usage:
#   install_isaacteleop_skeleton_correction.sh            apply (idempotent)
#   install_isaacteleop_skeleton_correction.sh --check    report, change nothing
#   install_isaacteleop_skeleton_correction.sh --revert   undo
#
# Override the defaults with ISAACTELEOP_DIR and VENV_TELEOP if your checkout
# lives elsewhere.

set -euo pipefail

ISAACTELEOP_DIR="${ISAACTELEOP_DIR:-$HOME/IsaacTeleop}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_TELEOP="${VENV_TELEOP:-$REPO_ROOT/.venv_teleop}"

MODE="apply"
case "${1:-}" in
    --check) MODE="check" ;;
    --revert) MODE="revert" ;;
    --help|-h) sed -n '5,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    "") ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

fail() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "  $*"; }

# ---------------------------------------------------------------------------
# Locate everything before touching anything.
# ---------------------------------------------------------------------------

SRC="$ISAACTELEOP_DIR/src/python/isaacteleop/retargeting_engine/utilities/full_body_transform.py"
SRC_WRIST="$ISAACTELEOP_DIR/src/python/isaacteleop/retargeters/G1/wrist_bias.py"

PKG="$(echo "$VENV_TELEOP"/lib/python*/site-packages/isaacteleop/retargeting_engine)"
[ -d "$PKG" ] || fail "no isaacteleop in $VENV_TELEOP -- is the venv created? (looked for lib/python*/site-packages/isaacteleop)"

DST="$PKG/utilities/full_body_transform.py"
PKG_G1="$(dirname "$PKG")/retargeters/G1"
DST_WRIST="$PKG_G1/wrist_bias.py"
INIT="$PKG/utilities/__init__.py"
TYPES="$PKG/tensor_types/standard_types.py"

echo "isaacteleop skeleton-correction bridge [$MODE]"
note "source : $SRC"
note "target : $PKG"

if [ "$MODE" != "revert" ] && { [ ! -f "$SRC" ] || [ ! -f "$SRC_WRIST" ]; }; then
    # Overwhelmingly the cause: the checkout is on a branch that predates #921.
    BRANCH="$(git -C "$ISAACTELEOP_DIR" branch --show-current 2>/dev/null || echo '<not a git repo>')"
    fail "$SRC does not exist.
  $ISAACTELEOP_DIR is on branch '$BRANCH'.
  The correction only exists on jsepulveda/quest_remapping (NVIDIA/IsaacTeleop#921)
  until that PR merges. Check that branch out, then re-run:
      git -C $ISAACTELEOP_DIR checkout jsepulveda/quest_remapping"
fi

# ---------------------------------------------------------------------------
# Revert.
# ---------------------------------------------------------------------------

if [ "$MODE" = "revert" ]; then
    [ -L "$DST" ] && { rm "$DST"; note "removed symlink"; } || note "no symlink to remove"
    [ -L "$DST_WRIST" ] && { rm "$DST_WRIST"; note "removed wrist-bias symlink"; } || note "no wrist-bias symlink to remove"
    if grep -q "^from .wrist_bias import" "$PKG_G1/__init__.py" 2>/dev/null; then
        sed -i "/^from .wrist_bias import/d;/^__all__ = \[\"WRIST_BIAS_RAD\"/d" "$PKG_G1/__init__.py"; note "removed wrist-bias exports"
    fi
    if grep -q "bridge alias for quest_remapping" "$TYPES" 2>/dev/null; then
        sed -i '/bridge alias for quest_remapping/d' "$TYPES"; note "removed NUM_BODY_JOINTS alias"
    else
        note "no alias to remove"
    fi
    if grep -q "^from .full_body_transform import" "$INIT" 2>/dev/null; then
        python3 - "$INIT" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = re.sub(r"from \.full_body_transform import \([^)]*\)\n", "", s)
for name in ('"FullBodyTransform",', '"correct_body_orientations",', '"SKELETON_PROFILES",'):
    s = re.sub(r"[ \t]*" + re.escape(name) + r"\n", "", s)
s = s.replace("    # Data\n]", "]")
p.write_text(s)
PY
        note "removed exports"
    else
        note "no exports to remove"
    fi
    echo "reverted. .venv_teleop is back to stock 1.3.131 and --skeleton-source quest will not work."
    exit 0
fi

# ---------------------------------------------------------------------------
# Check / apply. Each step reports one of: ok (already done), applied, MISSING.
# ---------------------------------------------------------------------------

STATUS=0
step() { # name, is_present, apply_cmd
    local name="$1" present="$2"
    if [ "$present" = "yes" ]; then note "ok       $name"; return; fi
    if [ "$MODE" = "check" ]; then note "MISSING  $name"; STATUS=1; return; fi
    shift 2; "$@"; note "applied  $name"
}

# 1. The module itself, symlinked so edits in the IsaacTeleop checkout are live.
present_link="no"
[ -L "$DST" ] && [ "$(readlink -f "$DST")" = "$(readlink -f "$SRC")" ] && present_link="yes"
do_link() { rm -f "$DST"; ln -s "$SRC" "$DST"; }
step "full_body_transform.py symlink" "$present_link" do_link

# 2. 1.5 renamed NUM_BODY_JOINTS_PICO to NUM_BODY_JOINTS; the module imports the
#    new name, so alias it rather than editing the module.
present_alias="no"
grep -q "^NUM_BODY_JOINTS = " "$TYPES" && present_alias="yes"
do_alias() {
    grep -q "^NUM_BODY_JOINTS_PICO = " "$TYPES" \
        || fail "NUM_BODY_JOINTS_PICO not found in $TYPES -- unexpected isaacteleop version"
    sed -i 's/^\(NUM_BODY_JOINTS_PICO = .*\)$/\1\nNUM_BODY_JOINTS = NUM_BODY_JOINTS_PICO  # 1.5 name; bridge alias for quest_remapping/' "$TYPES"
}
step "NUM_BODY_JOINTS alias" "$present_alias" do_alias

# 3. Re-export so consumers can import from the package, as they do on 1.5.
present_exports="no"
grep -q "^from .full_body_transform import" "$INIT" && present_exports="yes"
do_exports() {
    python3 - "$INIT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
imp = ("from .full_body_transform import (\n"
       "    FullBodyTransform,\n    correct_body_orientations,\n    SKELETON_PROFILES,\n)\n")
anchor = "from .hand_transform import HandTransform\n"
s = s.replace(anchor, anchor + imp, 1) if anchor in s else imp + s
s = s.replace('__all__ = [\n',
              '__all__ = [\n    "FullBodyTransform",\n    "correct_body_orientations",\n    "SKELETON_PROFILES",\n', 1)
p.write_text(s)
PY
}
step "utilities/__init__ exports" "$present_exports" do_exports

# 4. The G1 wrist bias, the robot-side companion to the skeleton correction.
present_wrist="no"
[ -L "$DST_WRIST" ] && [ "$(readlink -f "$DST_WRIST")" = "$(readlink -f "$SRC_WRIST")" ] && present_wrist="yes"
do_wrist() { rm -f "$DST_WRIST"; ln -s "$SRC_WRIST" "$DST_WRIST"; }
step "G1 wrist_bias.py symlink" "$present_wrist" do_wrist

present_wrist_exports="no"
grep -q "^from .wrist_bias import" "$PKG_G1/__init__.py" 2>/dev/null && present_wrist_exports="yes"
do_wrist_exports() {
    printf '%s\n' 'from .wrist_bias import WRIST_BIAS_RAD, wrist_bias_for' \
                   '__all__ = ["WRIST_BIAS_RAD", "wrist_bias_for"]' >> "$PKG_G1/__init__.py"
}
step "G1 __init__ exports" "$present_wrist_exports" do_wrist_exports

# ---------------------------------------------------------------------------
# Prove it actually imports. A green checklist above means nothing on its own:
# a stale symlink target or a version skew still fails here.
# ---------------------------------------------------------------------------

if [ "$MODE" = "check" ] && [ "$STATUS" -ne 0 ]; then
    echo "bridge INCOMPLETE -- re-run without --check to apply it."
    exit 1
fi

echo "verifying:"
if "$VENV_TELEOP/bin/python" - <<'PY'
import sys
try:
    from isaacteleop.retargeting_engine.utilities import (
        correct_body_orientations, SKELETON_PROFILES)
    from isaacteleop.retargeters.G1 import wrist_bias_for
    import numpy as np
    q = np.tile([0.0, 0.0, 0.0, 1.0], (24, 1))
    assert np.array_equal(correct_body_orientations(q, "pico"), q), "pico must be identity"
    assert not np.allclose(correct_body_orientations(q, "quest"), q), "quest must change the input"
    assert wrist_bias_for("pico") == ((0.0,)*3, (0.0,)*3), "pico wrist bias must be zero"
    assert any(any(s) for s in wrist_bias_for("quest")), "quest wrist bias must be non-zero"
    print("  import ok; profiles:", sorted(SKELETON_PROFILES), "+ G1 wrist bias")
except Exception as exc:  # noqa: BLE001
    print(f"  FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1)
PY
then
    echo "bridge is in place. --skeleton-source quest will use the isaacteleop correction."
else
    fail "the bridge is in place but isaacteleop still will not import the correction.
  Most likely $ISAACTELEOP_DIR moved off jsepulveda/quest_remapping, which leaves
  the symlink dangling. Check it out again and re-run."
fi
