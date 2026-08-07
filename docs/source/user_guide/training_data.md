# Training Data

## BONES-SEED

[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) is an open dataset of **142,220 annotated human motion animations** for humanoid robotics, created by [Bones Studio](https://bones.studio/datasets). It provides motion capture data in SOMA and Unitree G1 formats with natural language descriptions, temporal segmentation labels, and detailed skeletal metadata.

| | |
|---|---|
| **Total motions** | 142,220 (71,132 original + 71,088 mirrored) |
| **Total duration** | ~288 hours (@ 120 fps) |
| **Performers** | 522 actors (253 F / 269 M) |
| **Age range** | 17–71 years |
| **Height range** | 145–199 cm |
| **Weight range** | 38–145 kg |
| **Output formats** | SOMA Uniform · SOMA Proportional · Unitree G1 MuJoCo-compatible |
| **Annotations** | Up to 6 NL descriptions per motion + temporal segmentation + skeletal metadata |

### Relevance to SONIC

BONES-SEED a large subset of SONIC training data:

- **Unitree G1 joint trajectories** — retargeted for MuJoCo, directly usable for motion tracking training
- **Broad motion coverage** — locomotion, manipulation, dance, sports, communication, and everyday activities across 8 categories and 20 sub-categories
- **Rich language annotations** — up to 6 natural language descriptions per motion, enabling language-conditioned policy learning
- **Temporal segmentation** — per-motion phase labels with timestamps for structured skill decomposition
- **Performer diversity** — 522 actors spanning a wide range of body types, ages, and movement styles

### Motion Categories

| Package       | Motions | Description                                                             |
|---------------|---------|-------------------------------------------------------------------------|
| Locomotion    | 74,488  | Walking, jogging, jumping, climbing, crawling, turning, and transitions |
| Communication | 21,493  | Gestures, pointing, looking, and communicative body language            |
| Interactions  | 14,643  | Object manipulation, pick-and-place, carrying, and tool use             |
| Dances        | 11,006  | Full-body dance performances across multiple styles                     |
| Gaming        | 8,700   | Game-inspired actions and dynamic movements                             |
| Everyday      | 5,816   | Household tasks, consuming, sitting, reading, and daily activities      |
| Sport         | 3,993   | Athletic movements and sports-specific actions                          |
| Other         | 2,081   | Stunts, martial arts, and edge-case motions                             |

### Data Formats

Every motion is available in three formats:

- **SOMA Proportional (BVH)** — per-actor skeleton preserving original body proportions
- **SOMA Uniform (BVH)** — standardized skeleton shared across all motions for batch processing
- **Unitree G1 (CSV)** — joint-angle trajectories retargeted to the Unitree G1 humanoid

### Download

```bash
# Using the Hugging Face CLI
pip install huggingface_hub
huggingface-cli download bones-studio/seed --repo-type dataset --local-dir ./bones-seed
```

```python
# Using Python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bones-studio/seed",
    repo_type="dataset",
    local_dir="./bones-seed"
)
```

After downloading, extract the motion archives:

