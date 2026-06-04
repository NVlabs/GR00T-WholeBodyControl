# XR Upper-Body Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-repo bridge that replays upper-body trajectory frames into GEAR-SONIC `zmq_manager` while GEAR-SONIC remains the only Unitree body `LowCmd` writer.

**Architecture:** Add a focused Python module for frame loading, validation, rate limiting, and ZMQ publishing. Reuse `gear_sonic.utils.teleop.zmq.zmq_planner_sender` so the bridge emits the same packed `command` and `planner` messages already decoded by C++ `ZMQManager`.

**Tech Stack:** Python 3.10, NumPy, pyzmq, pytest, existing GEAR-SONIC ZMQ planner message builders.

---

### Task 1: Trajectory Frame Model And Loaders

**Files:**
- Create: `gear_sonic/utils/teleop/xr_upperbody_bridge.py`
- Test: `gear_sonic/tests/test_xr_upperbody_bridge.py`

- [ ] **Step 1: Write tests for JSONL and NPZ loading**

Create tests that write temporary JSONL/NPZ files with `upper_body_position` `[17]`, optional hand arrays `[7]`, and timestamps. Assert loaded frame count, default velocity behavior, and validation errors for bad dimensions.

- [ ] **Step 2: Implement `UpperBodyFrame`, `load_jsonl_frames`, and `load_npz_frames`**

Use dataclasses and NumPy arrays. Required array shapes are `upper_body_position=(17,)`; optional shapes are `upper_body_velocity=(17,)`, `left_hand_joints=(7,)`, `right_hand_joints=(7,)`.

- [ ] **Step 3: Run targeted loader tests**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_upperbody_bridge.py -q`

Expected: loader tests pass.

### Task 2: Filtering And Planner Message Conversion

**Files:**
- Modify: `gear_sonic/utils/teleop/xr_upperbody_bridge.py`
- Test: `gear_sonic/tests/test_xr_upperbody_bridge.py`

- [ ] **Step 1: Write tests for clipping, rate limiting, and packed field names**

Decode the existing packed message header and assert `mode`, `movement`, `facing`, `speed`, `height`, `upper_body_position`, `upper_body_velocity`, and hand fields are emitted with expected shapes.

- [ ] **Step 2: Implement `BridgeConfig`, `UpperBodyFilter`, and `build_frame_planner_message`**

Default lower-body command must be idle: `mode=0`, `movement=[0,0,0]`, `facing=[1,0,0]`, `speed=-1`, `height=-1`. Clip upper-body positions to configurable radians and limit per-frame position deltas.

- [ ] **Step 3: Run targeted conversion tests**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_upperbody_bridge.py -q`

Expected: conversion tests pass.

### Task 3: CLI Publisher

**Files:**
- Create: `gear_sonic/scripts/xr_upperbody_bridge.py`
- Modify: `gear_sonic/utils/teleop/xr_upperbody_bridge.py`
- Test: `gear_sonic/tests/test_xr_upperbody_bridge.py`

- [ ] **Step 1: Write dry-run CLI test**

Exercise `main()` with `--dry-run --input <jsonl> --hz 50 --once` and assert it reports loaded/published frame counts without opening a ZMQ socket.

- [ ] **Step 2: Implement CLI arguments and publisher loop**

Support `--input`, `--format auto|jsonl|npz`, `--bind-host`, `--port`, `--hz`, `--loop`, `--once`, `--dry-run`, `--start-control`, `--send-stop-on-exit`, `--max-abs-joint`, and `--max-joint-step`. In non-dry-run mode bind a ZMQ PUB socket and send initial `command(start=<flag>, stop=False, planner=True)`.

- [ ] **Step 3: Run targeted tests and py_compile**

Run:
`PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_upperbody_bridge.py -q`

Run:
`PYTHONDONTWRITEBYTECODE=1 python -B -m py_compile gear_sonic/utils/teleop/xr_upperbody_bridge.py gear_sonic/scripts/xr_upperbody_bridge.py`

Expected: tests and compilation pass.

### Task 4: Documentation

**Files:**
- Create: `docs/source/tutorials/xr_upperbody_bridge.md`
- Modify: `docs/source/tutorials/vr_wholebody_teleop.md` or relevant index only if needed.

- [ ] **Step 1: Document sim-first launch**

Explain that GEAR-SONIC remains the only `LowCmd` writer, then show a sim launch sequence using `run_sim_loop.py`, `deploy.sh --input-type zmq_manager sim`, and the bridge CLI.

- [ ] **Step 2: Document real-robot safety boundary**

State that real robot use is manual, requires existing safety confirmation, and the bridge must not run another body command writer.

- [ ] **Step 3: Run doc-relevant smoke checks**

Run py_compile again and report that Sphinx was not run unless requested.

### Self-Review

- Spec coverage: the plan covers frame input, ZMQ planner publication, safety filters, dry-run/sim-first validation, and docs.
- Placeholder scan: no implementation step relies on unnamed behavior; field names and dimensions are explicit.
- Type consistency: Python arrays use NumPy, frame fields match existing `build_planner_message` parameters and C++ `PlannerMessage` field names.
