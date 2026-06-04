import numpy as np

isaaclab = "dump_step0/isaaclab_step0.npz"
mujoco = "dump_step0/mujoco_step0_from_isaac_state.npz"
report = "dump_step0/comparison_report.txt"

il = np.load(isaaclab)
mj = np.load(mujoco)

common = sorted(set(il.keys()) & set(mj.keys()))
print(f"IsaacLab keys: {len(il.keys())}")
print(f"MuJoCo keys:   {len(mj.keys())}")
print(f"Common keys:   {len(common)}")
print("Common key list:")
for k in common:
    print(" ", k)
print()
lines = []
lines.append("COMPARISON: IsaacLab vs MuJoCo step-0")
lines.append(f"IsaacLab: {isaaclab}")
lines.append(f"MuJoCo:   {mujoco}")
lines.append(f"Common keys: {len(common)}\n")

max_err = 0.0
for k in common:
    a, b = il[k], mj[k]

    if isinstance(a, np.ndarray) and a.ndim > 0 and a.shape[0] == 1:
        a = a[0]
    if isinstance(b, np.ndarray) and b.ndim > 0 and b.shape[0] == 1:
        b = b[0]

    if a.shape != b.shape:
        line = f"{k:40s} SHAPE MISMATCH {a.shape} vs {b.shape}"
    elif a.dtype.kind in {"U", "S", "O"}:
        line = f"{k:40s} STRING/OBJECT TYPE (Skipped diff)"
    else:
        d = a - b
        ma = float(np.max(np.abs(d)))
        rms = float(np.sqrt(np.mean(d ** 2)))
        max_err = max(max_err, ma)
        line = f"{k:40s} max_abs={ma:.6e} rms={rms:.6e}"

        if ma > 1e-4:
            flat_a = a.reshape(-1)
            flat_b = b.reshape(-1)
            flat_d = np.abs(d).reshape(-1)
            worst = np.argsort(flat_d)[-5:][::-1]
            line += "\n"
            for idx in worst:
                line += (
                    f"    idx={idx:4d} "
                    f"il={flat_a[idx]: .6f} "
                    f"mj={flat_b[idx]: .6f} "
                    f"diff={flat_d[idx]:.6e}\n"
                )

    print(line)
    lines.append(line)

def find_subvector(vec, sub, name):
    vec = vec.reshape(-1)
    sub = sub.reshape(-1)
    best_i = None
    best_err = 1e9
    for i in range(len(vec) - len(sub) + 1):
        err = np.max(np.abs(vec[i:i+len(sub)] - sub))
        if err < best_err:
            best_err = err
            best_i = i
    line = f"{name:25s} best_i={best_i:4d} best_err={best_err:.6e}"
    print(line)
    lines.append(line)

print("\nIsaacLab policy_obs block search:")
lines.append("\nIsaacLab policy_obs block search:")

po = il["policy_obs"]
if po.ndim > 1 and po.shape[0] == 1:
    po = po[0]
find_subvector(po, il["obs_term_base_ang_vel"], "base_ang_vel")
find_subvector(po, il["obs_term_joint_pos_rel"], "joint_pos_rel")
find_subvector(po, il["obs_term_joint_vel"], "joint_vel")
find_subvector(po, il["obs_term_last_action"], "last_action")
find_subvector(po, il["obs_term_gravity_dir"], "gravity_dir")

summary = f"\nMAX error: {max_err:.6e}"
print(summary)
lines.append(summary)

with open(report, "w") as f:
    f.write("\n".join(lines))

print(f"\nReport saved to {report}")
