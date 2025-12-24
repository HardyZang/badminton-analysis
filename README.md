# Badminton Overhead Clear Analysis (MVP)

Python CLI to extract pose with MediaPipe Pose, detect overhead clear swing segments, compute basic metrics, and export overlay videos, JSON reports, CSV keypoints, and summary plots.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

```bash
python analyze_clear.py --input path/to/video_or_folder --out outputs --hand auto --debug
```

- `--input` can be a single video (`.mp4/.mov/.avi`) or a folder containing videos.
- `--out` output root directory; each video gets its own subfolder.
- `--hand` choose `auto|left|right`; `auto` picks the wrist that peaks higher more often.
- `--debug` prints extra info.

Outputs per video:

- `keypoints.csv`: per-frame 2D joints (pixels, smoothed)
- `report.json`: metrics, scores, suggestions
- `overlay.mp4`: skeleton and strike markers
- `summary.png`: wrist height curves with strike lines

Example output layout for `sample.mp4` (no video included):

```
outputs/
  sample/
    keypoints.csv
    report.json
    overlay.mp4
    summary.png
```

## Filming tips

- Fixed camera, full body in frame, bright and stable lighting.
- Record in landscape, allow space above the head to capture full swing.
- Single athlete performing overhead clears; tripod height roughly waist to chest level.

## Metric notes

- **Strike height**: wrist vs shoulder/nose height at strike.
- **Contact position**: wrist offset vs torso midline (前/合适/后).
- **Trunk rotation**: shoulder/hip line change from prep to strike.
- **Non-racket arm**: max abduction angle/time window before strike.
- **Elbow timing**: elbow angle curve; early straightening flagged.
- **Footwork hint**: hip center speed pre/post strike for recovery cue.

Thresholds and weights are in code (`clear_metrics.py`) for quick tuning.
