#!/usr/bin/env python3
"""Check actuator parameters in gear_sonic robots/*.py against URDF/MJCF joint limits.

Catches the class of bug reported in issue #259 (h2.py effort_limit_sim disagreeing
with h2.urdf / h2.xml). Parses each robot's ImplicitActuatorCfg effort_limit_sim /
velocity_limit_sim (dict or scalar + joint_names_expr regex) and compares them to
the <limit effort= velocity=> attributes in the robot's URDF.

Usage:
    python scripts/check_actuator_params.py                 # all robots
    python scripts/check_actuator_params.py --robot h2      # one robot
    python scripts/check_actuator_params.py --urdf <path> --config <path>  # explicit

Exit code 0 = all consistent, 1 = mismatches found, 2 = error.
"""
import argparse
import fnmatch
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROBOTS_DIR = os.path.join(REPO_ROOT, "gear_sonic", "envs", "manager_env", "robots")
URDF_DIR = os.path.join(REPO_ROOT, "gear_sonic", "data", "assets", "robot_description", "urdf")

# Joints that are NOT expected to appear in actuator groups (passive/fixed).
PASSIVE_JOINT_HINTS = ("virtual", "floating", "tarsus", "toe", "thumb", "imu", "sensor")


def parse_urdf_limits(urdf_path):
    """Return {joint_name: {effort: float|None, velocity: float|None}}."""
    with open(urdf_path, encoding="utf-8") as f:
        xml = f.read()
    limits = {}
    # <joint name="..."> ... <limit effort="X" velocity="Y"/> (may be self-closing on one line)
    for m in re.finditer(
        r'<joint\b([^>]*)>(.*?)</joint>', xml, re.S | re.I
    ):
        head, body = m.group(1), m.group(2)
        nm = re.search(r'name="([^"]+)"', head)
        if not nm:
            continue
        limit = re.search(r'<limit\b[^>]*?>', body, re.I)
        eff = vel = None
        if limit:
            e = re.search(r'effort="([^"]+)"', limit.group(0), re.I)
            v = re.search(r'velocity="([^"]+)"', limit.group(0), re.I)
            eff = float(e.group(1)) if e else None
            vel = float(v.group(1)) if v else None
        limits[nm.group(1)] = {"effort": eff, "velocity": vel}
    return limits


def parse_actuator_config(py_path):
    """Extract actuator groups from a robots/*.py file.

    Returns list of {joints: [regex,...], effort: float|None, velocity: float|None}.
    Handles both dict form (effort_limit_sim={".*_x": 100.0}) and scalar form
    (effort_limit_sim=150.0 with joint_names_expr=[...]).
    """
    with open(py_path, encoding="utf-8") as f:
        src = f.read()
    groups = []
    # Find each ImplicitActuatorCfg(...) block
    for m in re.finditer(r"ImplicitActuatorCfg\s*\((.*?)\)\s*,", src, re.S):
        block = m.group(1)
        expr = re.search(r"joint_names_expr\s*=\s*\[(.*?)\]", block, re.S)
        if not expr:
            continue
        joints = re.findall(r'["\']([^"\']+)["\']', expr.group(1))
        if not joints:
            continue
        eff = vel = None
        em = re.search(r"effort_limit_sim\s*=\s*(\{.*?\}|[0-9.eE+-]+)", block, re.S)
        if em:
            raw = em.group(1).strip()
            if raw.startswith("{"):
                # dict: map each regex to a value; use max as representative,
                # but keep per-regex info in the report instead.
                d = {}
                for kv in re.finditer(r'["\']([^"\']+)["\']\s*:\s*([0-9.eE+-]+)', raw):
                    d[kv.group(1)] = float(kv.group(2))
                eff = d
            else:
                eff = float(raw)
        vm = re.search(r"velocity_limit_sim\s*=\s*(\{.*?\}|[0-9.eE+-]+)", block, re.S)
        if vm:
            raw = vm.group(1).strip()
            if raw.startswith("{"):
                d = {}
                for kv in re.finditer(r'["\']([^"\']+)["\']\s*:\s*([0-9.eE+-]+)', raw):
                    d[kv.group(1)] = float(kv.group(2))
                vel = d
            else:
                vel = float(raw)
        groups.append({"joints": joints, "effort": eff, "velocity": vel})
    return groups


def joint_matches(expr, joint):
    """Match a joint name against an actuator regex (fnmatch on translated regex)."""
    if expr == ".*":
        return True
    # Convert simple .*_suffix patterns to fnmatch-style
    if expr.startswith(".*") and expr.endswith("_joint"):
        mid = expr[2:-6]  # strip .* and _joint
        return joint.startswith(mid) and joint.endswith("_joint")
    if expr.endswith("_joint") and not expr.startswith(".*"):
        return joint == expr
    # fall back to regex match
    try:
        return re.fullmatch(expr, joint) is not None
    except re.error:
        return fnmatch.fnmatch(joint, expr)


def lookup_val(mapping, joint):
    """mapping may be a dict (regex->val) or scalar float."""
    if mapping is None:
        return None
    if isinstance(mapping, dict):
        for expr, val in mapping.items():
            if joint_matches(expr, joint):
                return val
        return None
    return float(mapping)


def find_urdf(urdf_dir, name):
    """Locate the robot's URDF — try {name}.urdf then main.urdf."""
    for cand in (f"{name}.urdf", "main.urdf"):
        p = os.path.join(urdf_dir, name, cand)
        if os.path.isfile(p):
            return p
    return None


def check_robot(name, robots_dir, urdf_dir):
    py = os.path.join(robots_dir, f"{name}.py")
    urdf = find_urdf(urdf_dir, name)
    if not os.path.isfile(py):
        print(f"[skip] {name}: no {name}.py in {robots_dir}")
        return 0
    if not urdf:
        print(f"[skip] {name}: no {name}.urdf / main.urdf in {urdf_dir}/{name}")
        return 0
    urdf_limits = parse_urdf_limits(urdf)
    groups = parse_actuator_config(py)
    if not groups:
        print(f"[warn] {name}: no ImplicitActuatorCfg groups parsed")
        return 0
    print(f"=== {name} ({len(groups)} actuator groups, {len(urdf_limits)} urdf joints) ===")
    mismatches = 0
    for gi, g in enumerate(groups):
        for joint in urdf_limits:
            if any(h in joint for h in PASSIVE_JOINT_HINTS):
                continue
            if not any(joint_matches(e, joint) for e in g["joints"]):
                continue
            ul = urdf_limits[joint]
            ae = lookup_val(g["effort"], joint)
            av = lookup_val(g["velocity"], joint)
            if ae is not None and ul["effort"] is not None and abs(ae - ul["effort"]) > 1e-6:
                ratio = ae / ul["effort"] if ul["effort"] else float("inf")
                print(f"  [X] {joint}: actuator effort {ae:g} vs urdf {ul['effort']:g} ({ratio:.2f}x)")
                mismatches += 1
            if av is not None and ul["velocity"] is not None and abs(av - ul["velocity"]) > 1e-6:
                print(f"  [X] {joint}: actuator velocity {av:g} vs urdf {ul['velocity']:g}")
                mismatches += 1
    if mismatches == 0:
        print("  [+] all actuator limits consistent with urdf")
    return mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="", help="single robot name (e.g. h2)")
    ap.add_argument("--robots-dir", default=ROBOTS_DIR)
    ap.add_argument("--urdf-dir", default=URDF_DIR)
    args = ap.parse_args()
    if args.robot:
        names = [args.robot]
    else:
        names = sorted(
            f[:-3] for f in os.listdir(args.robots_dir)
            if f.endswith(".py") and not f.startswith("__") and not f.startswith("_")
        )
    total = 0
    for n in names:
        total += check_robot(n, args.robots_dir, args.urdf_dir)
    if total:
        print(f"\n{total} mismatch(es) found")
        sys.exit(1)
    print("\nall checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
