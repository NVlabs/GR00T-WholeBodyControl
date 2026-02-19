# Whole-body Teleoperation Guide

This guide covers best practices during whole-body teleoperation.

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart), [PICO Setup](../getting_started/vr_teleop_setup), and [Teleop Setup](../tutorials/vr_wholebody_teleop)
```

## Overview

Whole-body teleoperation is very very hard to get right and there are a lot of moving parts. This guide will walk you through the details that are required during whole-body teleoperation. During whole-body teleoperation, the GEAR-SONIC policy will try to copy your motion **as much as possible**, including your foot movements! Thus, it can be quite **demanding and dangerous** if proper precautions are not taken.

```{admonition} Safety Warning
:class: danger
The robot will track your full-body movements in real-time. Always maintain a clear 3-meter safety zone around the robot, keep a safety operator at the keyboard ready to press **`O`** for emergency stop and always be prepared to emergecy stop on the PICO controller on your own. Practice extensively in simulation before attempting on real hardware!
```

## Sample Teleoperation Session

A typical whole-body teleoperation session follows this workflow:


```{video} ../_static/teleop/teleop_session_overview.mp4
:width: 100%
```
*Video: Full startup sequence — calibrating the PICO headset, engaging the policy, and the robot starting to balance independently on the gantry.*

**Terminal 1 — MuJoCo Simulator** (or skip for real robot):
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2 — C++ Deployment**:
```bash
cd gear_sonic_deploy
# For simulation:
bash deploy.sh sim --input-type zmq_manager
# For real robot:
# bash deploy.sh real --input-type zmq_manager
```

**Terminal 3 — PICO Teleop Streamer**:
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager 
```

**Operator Actions**:
1. **Put on PICO headset and controllers** — Ensure foot trackers are securely attached. 
2. **Stand in calibration pose** — Upright, feet together, arms in down. Recalibrate often!!!
3. **Make robot stand loose but standing** - Put the G1 somehow slack on gantry (the policy will start and start balancing on its own). 
4. **Press A+B+X+Y** on controllers — Initializes the policy and calibrates (enters Planner mode)
5. **Press A+X** — Switches to Pose mode (whole-body teleoperation active)
6. **Teleoperate** — Your movements are now mirrored by the robot
7. **Press A+B+X+Y** when done — Emergency stop and exit. Policy will stop!!!

## Clothing Requirements

**Critical:** You **must** wear **tight-fitting pants or leggings** during teleoperation.

**Why this matters:**
- The PICO foot trackers use visual tracking and need a clear view of your leg/foot movements
- Loose, baggy, or flowing clothing (sweatpants, wide-leg pants, long skirts) will occlude the trackers
- Even brief tracking loss causes the robot to receive incorrect foot positions, leading to stumbling or dangerous motions

**Recommended:**
- ✅ Athletic leggings or compression tights
- ✅ Fitted jeans or slim-fit pants

**Not recommended:**
- ❌ Baggy sweatpants or cargo pants
- ❌ Wide-leg or flared pants
- ❌ Long dresses or skirts
- ❌ Any loose or flowing fabric around the legs

**Upper body:** Normal fitted clothing is fine. Avoid very baggy sleeves that might interfere with controller tracking. 

**❗️❗️❗️Recalibrate Often** The tracking quality of PICO may get worse overtime. When seeing performance drop, always recalibrate! Also, when the PICO controller loses track, it may get stuck in a sitting/weird pose. Always make sure your tracked motion is correct (by viewing the PICO avatar) before starting the teleoperation policy! 

## WiFi Delays

```{video} ../_static/teleop/teleop_natural_movement.mp4
:width: 100%
```
*Video: Demonstrating how natural vs. hesitant walking affects robot stability — stumbling caused by WiFi delays or unnatural movement.*

**Network latency should be kept as low as possible.** We provide tools to detect delays. Network delays can significantly affect the GEAR-SONIC policy's performance, as the flow of movement for the robot will be interrupted mid-movement (say mid-stride during walking) and the robot can stumble or lose balance.

**Best practices:**
- **Use Private WiFi Routers** Public and school wifis can easily have large delays. 
- **Minimize WiFi hops** — Ideally the PICO and deployment machine are on the same local network
- **Check latency** — Monitor ZMQ message delays in the terminal output


**Expected latency:**
- **Good:** < 10ms (wired or strong local WiFi)
- **Acceptable:** 10-30ms (may notice slight lag)
- **Poor:** > 30ms (robot will struggle to track smooth motions, increased stumbling risk)

**Checking network performance:**
The deployment terminal will show warnings if message delays exceed thresholds. Watch for messages like:
```
WARNING: High ZMQ latency detected: 45ms
```

If you see persistent high latency warnings, improve your network setup before continuing.

## Movement Patterns

```{video} ../_static/teleop/teleop_walking.mp4
:width: 100%
```
*Video: Walking demo — forward, backward, running, and sideways movement with natural gait.*

**Try to be as natural as possible.** The GEAR-SONIC policy is trained on natural human motion, so moving naturally gives the best results. Hesitating when moving actually lead to more stumbling! 

**Good movement practices:**

1. **Walk naturally** — Use your normal gait with natural arm swing. Don't exaggerate or try to "help" the robot.

2. **Transfer weight smoothly** — When stepping, shift your weight from one foot to the other just as you normally would. Be confident! 

3. **Use moderate speeds** — The robot can track fast motions, but start with walking at a comfortable pace. You can speed up/starts running once you're comfortable.

**Common mistakes:**

- ❌ **Overtly slow movements** — Don't slow down too much! 
- ❌ **Trying to match the robot's movement** — Don't try to match the robot movement, as the robot will then try to match your movement and starts stumbling. Be natural!

**Advanced movements:**

```{video} ../_static/teleop/teleop_advanced.mp4
:width: 100%
```
*Video: Advanced movements — kneeling, dynamic leg motions, and challenging poses.*

Once comfortable with basic walking:
- **Turning** — Turn naturally by rotating your torso and stepping in the new direction
- **Reaching** — Extend your arms smoothly to grab objects 
- **Squatting** — Bend your knees and lower your body naturally
- **Sidestepping** — Step sideways with natural weight transfer

## Calibration Best Practices

The initial calibration is **critical** to successful teleoperation.

**The calibration pose:**
1. **Stand upright** — Look straight ahead (not down at controllers)
2. **Feet together** — Foot parallel with no gaps. 
3. **Upper arms down** — Hang straight down beside your torso


**Tips:**
- Hold the pose steady for 1-2 seconds after pressing A+B+X+Y
- If the robot seems offset throughout the session, recalibrate (stop with A+B+X+Y, then restart)


## Mode Switching Safety

When switching between modes, **always match the robot's current pose first**.

**Dangerous scenario:**
1. Robot is in Planner mode standing upright
2. You're crouching or reaching in a different pose
3. You press **A+X** to switch to Pose mode
4. **Robot violently tries to match your crouched pose** ⚠️

**Safe procedure:**
1. Before pressing **A+X**, look at the robot (or visualization)
2. Move your body to approximately match the robot's current pose
3. Then press **A+X** — transition will be smooth

**Pause feature (Menu button):**

```{video} ../_static/teleop/teleop_pause.mp4
:width: 100%
```
*Video: Using the Menu button to pause and resume pose streaming during teleoperation.*

- Holding **Menu** pauses pose streaming
- **Before releasing Menu**, move your body back to match the robot's current pose
- Releasing Menu while in a very different pose causes sudden dangerous motions

## Troubleshooting

### Robot is not tracking my movements

**Possible causes:**
- Foot trackers are not securely attached or have low battery
- Loose clothing is occluding the trackers
- Poor lighting conditions
- XRoboToolKit not running on PICO or configured incorrectly

**Solutions:**
1. Check foot tracker placement and battery level
2. Verify you're wearing tight-fitting pants
3. Improve lighting (avoid very bright or very dark areas)
4. Restart XRoboToolKit on the PICO headset
5. Recalibrate 

### Robot makes sudden aggressive motions

**Possible causes:**
- Switched modes while poses didn't match
- Tracking glitch or foot tracker occlusion
- Network packet loss causing delayed frames

**Solutions:**
1. Recalibrate carefully before resuming
2. Always match robot's pose before switching modes
3. Check network latency and improve WiFi/wired connection

### Tracking is jittery or stumbles

**Possible causes:**
- Wireless delays
- IMU drift 
- Joint encoder drift


**Solutions:**
1. Reduce WiFi interference (move away from other wireless devices)
2. Reclibrate robot 

   
## Emergency Procedures

### Emergency stop methods

**Keyboard (deployment terminal):**
- Press **`O`** for immediate stop

**PICO controllers:**
- Press **A + B + X + Y** simultaneously

Both methods immediately halt the policy and exit control mode.


## Next Steps

- **Understand input interfaces** — See tutorials for [Keyboard](../tutorials/keyboard.md), [Gamepad](../tutorials/gamepad.md), [ZMQ](../tutorials/zmq.md), [Manager](../tutorials/manager.md)
- **Learn about deployment** — See [Deployment Code & Program Flow](../references/deployment_code)
- **General troubleshooting** — See [Troubleshooting Guide](troubleshooting) 