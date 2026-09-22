# Interaction Phase Annotation

`annotate_phases.py` opens a local window for video annotation. It saves phase
timestamps to `meta/interaction_metadata.jsonl` in the selected dataset. The interface
has a simple dark design with a video player, four phase fields, a timeline with orange
diamond markers, and automatic saving.

## Installation

You need Linux, Python **3.10 or newer**, and a desktop session. The only required
Python library is **PySide6**, which provides Qt Widgets and Qt Multimedia.

### Create a virtual environment with venv

First, open a terminal and move to the script folder:

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection/gear_sonic/scripts/process_dataset
```

Check that Python 3.10 or newer is available:

```bash
python3 --version
```

Create a virtual environment named `.venv-annotator`:

```bash
python3 -m venv .venv-annotator
```

Activate the environment:

```bash
source .venv-annotator/bin/activate
```

After activation, the terminal usually shows `(.venv-annotator)` before the command
prompt. Install the required library inside this environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-annotator.txt
```

Run the annotation tool while the environment is active:

```bash
python annotate_phases.py /home/nikita/Skoltech/MWS/Teleop-Data-Collection/outputs/sep-10-2026-cleaned-merged
```

When you open a new terminal, activate the same environment again before running the
tool:

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection/gear_sonic/scripts/process_dataset
source .venv-annotator/bin/activate
```

To leave the virtual environment, run:

```bash
deactivate
```

On Ubuntu, install these system packages if the `venv` command or Qt system libraries
are missing:

```bash
sudo apt install python3-venv libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0
```

The annotation tool does not need OpenCV, pandas, LeRobot, a server, or a web browser.
It plays videos through Qt Multimedia. See the
[PySide6 Multimedia documentation](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/index.html)
for more information.

## Running the tool

Run the following command:

```bash
python annotate_phases.py /home/nikita/Skoltech/MWS/Teleop-Data-Collection/outputs/sep-10-2026-cleaned-merged
```

Use `--episode` to open a specific episode at startup:

```bash
python annotate_phases.py /path/to/dataset --episode 42
```

The tool supports this dataset structure:

```text
dataset/
├── meta/interaction_metadata.jsonl
└── videos/chunk-000/observation.images.external_view_camera/
    ├── episode_000000.mp4
    ├── episode_000001.mp4
    └── ...
```

The tool also searches for videos in other `chunk-*` folders. It reads the episode ID
from `episode_index`, not from the line number. Missing numbers in the episode sequence
are allowed. If an episode has no video, annotation is disabled and the interface shows
an error message.

## How to annotate videos

1. Enter an **Episode ID** and press **Enter**. Use the arrow buttons next to the ID
   field to open the previous or next episode from the metadata file. When the ID field
   is not active, press **Enter** to open the next episode.
2. Start the video with the **Play** button, the **R** key, or the **K** key. You can select a playback
   speed from 0.25× to 2×. A new episode opens in the paused state. Audio is disabled.
3. Press **Space** at the start of each phase in this order:
   `approach_start` → `contact_active_start` → `release_start` → `idle_start`.
   The current phase has an orange border. You can add a marker while the video is
   playing or paused. If the current time is already at a diamond marker, press Space
   again to remove that marker. The cleared timestamp is saved as `0.0`.
4. Each marker appears in its phase field and on the timeline. It is saved immediately.
   After the fourth marker, the video pauses, the interface shows
   **"All timestamps are marked and saved"**, and the system plays a notification sound.
   A Space key press at an existing marker removes it. A Space key press at another time
   only shows the notification again when all four phases are complete.
5. Click a completed phase field or a diamond marker to move the video to that time.
   Drag a diamond marker to adjust its timestamp. The video frame and the value in the
   phase field follow the marker. The new value is saved when you release the marker.
   A marker cannot move past the previous or next phase. Hover over a marker to see its
   phase name and timestamp. Markers with the same time are placed at different heights
   so that you can select each one.

The current video time and all phase fields use the **seconds:milliseconds** format:
`05:240` means 5.240 seconds, and `125:007` means 125.007 seconds. The interface shows
milliseconds, but the real frame accuracy depends on the video FPS and decoder seeking.

| Action | Control |
| --- | --- |
| Add the next phase marker | Space or the orange button |
| Remove a marker at the current time | Space at its diamond marker |
| Pause or play | R, K, or the button under the video |
| Seek while paused | Click or drag on the timeline |
| Adjust a phase timestamp | Drag an orange diamond marker |
| Decrease or increase playback speed | Left Arrow or Right Arrow |
| Open the next episode | Enter outside the Episode ID field |
| Open the previous or next episode | Alt+Left Arrow or Alt+Right Arrow |
| Remove the last completed phase | Ctrl+Z or **Undo last marker** |
| Clear all four phase markers | **Reset phases**, then confirm |
| Close the application | Ctrl+C in the application or its terminal |

When you enter an episode ID, the Space and arrow keys do not change the video. Holding
Space does not create several markers. Phase timestamps must be in non-decreasing order.
If you try to add an earlier timestamp, the interface shows an error. To correct an early
phase, undo the later phases or reset all four phases.

## Data format and saving

Timestamps are stored in JSONL **in seconds**. New timestamps are rounded to three
decimal places:

```json
{"episode_index":42,"phase_timestamps":{"approach_start":1.24,"contact_active_start":2.86,"release_start":3.5,"idle_start":4.12}}
```

This example is shorter than a real record. The tool keeps all other fields in the
record. It does not change the formatting of other JSONL lines. The updated file is
written safely through a temporary file in the same folder. At the first change, the
tool creates `meta/interaction_metadata.jsonl.bak` with the original data. It does not
replace an existing backup.

- Four `0.0` values mean that an episode has no annotation. Empty and cleared phase
  timestamps are also stored as `0.0`. The tool does not write `null` for the four phases.
- The JSONL output does not contain extra progress fields. Every `0.0` timestamp is read
  as an empty phase, so partial annotations may contain gaps after you remove a marker.
  A real marker at `00:000` is not allowed because it looks the same as an empty value.
- After a restart, Space continues from the first unfinished phase. A complete annotation
  is not replaced unless you undo a marker or reset the phases.
- The tool saves after every change, including undo and reset. There are no unsaved
  changes when you open another episode or close the window. If saving fails, the tool
  shows an error and does not apply the new value in the interface.
- If another program changes the JSONL file, saving is blocked until you restart this
  tool. The `interaction_metadata.jsonl.lock` file coordinates writes between several
  annotation tool windows. Do not run another program that writes this JSONL file without
  using the same lock.

## Tests

The storage tests use the standard `unittest` library and do not need Qt:

```bash
python -m unittest discover -s . -p 'test_annotate_phases.py' -v
```

If a video does not open, check the dataset path and the error message below the phase
fields. Run the application in a desktop session, not through SSH without a graphical
display.
