# Adding Your Own Dataset

Our open-sourced datasets are available at <https://bones.studio/datasets>. The full training code, within the GR00T whole-body control framework, will be open-sourced soon at <https://github.com/NVlabs/GR00T-WholeBodyControl>. This guide covers bringing your own motion data into MotionBricks training.

## Two paths

You have two choices when bringing your own data:

1. **Bring your own training pipeline without the dependency on motionbricks' motion-representation handler and data loader (highly recommended).** There's no restriction on the motion representation used in our tokenizer, root, and pose module training. It is highly recommended to build your own training pipeline by referencing the current synthetic training path.

2. **Reuse the MotionBricks motion data representation.** In this path, you assume the same G1 Skeleton34 representation and only work with a different dataset processed in the same way. Full spec of the representation is in [`motion_representation.md`](./motion_representation.md).

---

## Reusing the MotionBricks Representation

Write a PyTorch `Dataset` whose `__getitem__` returns `{"keyid": int, "motion": Tensor[T, feature_dim]}`, where `motion` is the **already computed and normalized** global motion feature tensor for a single clip. See [`motion_representation.md`](./motion_representation.md) for the feature layout and how to produce it from raw joint rotations + global positions.

You can reuse the provided normalization stats, but you are also encouraged to compute your own.

See `motionbricks/data/synthetic_dataset.py` for the minimal dataset interface.

## Rolling Your Own

For most non-G1 datasets, we **highly recommend** writing your own dataset loader and motion-representation handler rather than forcing the existing ones to fit.

### Pointers in the code

| What you're writing | Start from |
|---------------------|------------|
| A new motion-rep class | `motionbricks/motionlib/core/motion_reps/motion_reps_base/motion_rep_base.py` (`MotionRepBase`) — minimal base that only wires normalization |
| A representation with a separate root / body split | `motionbricks/motionlib/core/motion_reps/motion_reps_base/seperate_root_local_body.py` |
| A full dual-root example | `motionbricks/motionlib/core/motion_reps/dual_root_global_joints.py` (builds `GlobalRootGlobalJoints`, `LocalRootGlobalJoints`, `DualRootGlobalJoints`) |
| The actual feature math (FK, velocity, heading) | `motionbricks/motionlib/core/motion_reps/tools/motion_features.py` — in particular `compute_motion_features` and the per-feature helpers it dispatches to |
| A minimal dataset loader | `motionbricks/data/synthetic_dataset.py` (`SyntheticMotionDataset`, `collate_batch`) |
| Stats handling | `motionbricks/motionlib/core/utils/stats.py` (`Stats`) |
