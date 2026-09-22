# Tau viewer presets

`tau_presets.json` maps each preset name to the joints shown on the tau plot.
The JSON object order is also the order used by the viewer: press `A` for the
previous preset and `D` for the next one. Joint names must come from the list
below; unknown names and duplicates are rejected at startup.

The viewer keeps all 29 displayed `tau_est` channels in `TauHistory`,
regardless of the active preset. The 17 waist/arm channels come from indices
`53:70` of `g1_upper_body_state` in the pico manager `pose`/`manager_state`
stream (port `5556` by default). The 12 finger channels come from
`rt/brainco/{left,right}/state` in BrainCo motor order.

## Upper-body joints (17)

1. `waist_yaw_joint`
2. `waist_roll_joint`
3. `waist_pitch_joint`
4. `left_shoulder_pitch_joint`
5. `left_shoulder_roll_joint`
6. `left_shoulder_yaw_joint`
7. `left_elbow_joint`
8. `left_wrist_roll_joint`
9. `left_wrist_pitch_joint`
10. `left_wrist_yaw_joint`
11. `right_shoulder_pitch_joint`
12. `right_shoulder_roll_joint`
13. `right_shoulder_yaw_joint`
14. `right_elbow_joint`
15. `right_wrist_roll_joint`
16. `right_wrist_pitch_joint`
17. `right_wrist_yaw_joint`

## Left-hand motors (6)

1. `left_brainco_thumb`
2. `left_brainco_thumb_aux`
3. `left_brainco_index`
4. `left_brainco_middle`
5. `left_brainco_ring`
6. `left_brainco_pinky`

## Right-hand motors (6)

1. `right_brainco_thumb`
2. `right_brainco_thumb_aux`
3. `right_brainco_index`
4. `right_brainco_middle`
5. `right_brainco_ring`
6. `right_brainco_pinky`

An alternative preset file can be selected with `--tau-presets-path`. The
initial preset can be selected with `--tau-preset`.
