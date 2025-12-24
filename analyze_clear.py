import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

from clear_events import SwingEvent, choose_hand, detect_swings
from clear_metrics import EventMetrics, compute_event_metrics
from clear_pose import PoseExtractor, save_pose_csv
from render_overlay import render_overlay_video, render_summary_plot


def gather_videos(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = []
        for ext in ("*.mp4", "*.mov", "*.avi"):
            files.extend(Path(input_path).glob(ext))
        return [str(f) for f in sorted(files)]
    return [input_path]


def summarize_feedback(all_metrics: List[EventMetrics]) -> Dict[str, List[str]]:
    issues = Counter()
    for m in all_metrics:
        for s in m.suggestions:
            issues[s] += 1
    top_issues = [s for s, _ in issues.most_common(3)]
    top_fixes = top_issues[:3]
    return {"issues": top_issues, "fixes": top_fixes}


def process_video(video_path: str, out_dir: str, hand_pref: str = "auto", debug: bool = False):
    video_name = Path(video_path).stem
    video_out_dir = Path(out_dir) / video_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    extractor = PoseExtractor(smooth_alpha=0.25)
    sequence = extractor.extract(video_path)
    save_pose_csv(sequence, video_out_dir / "keypoints.csv")

    hand = choose_hand(sequence.frames, hand_pref)
    events = detect_swings(sequence.frames, sequence.fps, hand)
    metrics: List[EventMetrics] = [compute_event_metrics(sequence.frames, ev) for ev in events]

    overlay_path = video_out_dir / "overlay.mp4"
    render_overlay_video(video_path, sequence.frames, events, metrics, str(overlay_path))
    summary_plot = video_out_dir / "summary.png"
    render_summary_plot(sequence.frames, events, str(summary_plot))

    report = {
        "video": video_path,
        "fps": sequence.fps,
        "hand": hand,
        "events": [],
    }
    for idx, m in enumerate(metrics):
        report["events"].append(
            {
                "index": idx,
                "frames": {"start": m.event.start_idx, "strike": m.event.strike_idx, "end": m.event.end_idx},
                "score": m.score,
                "wrist_height_shoulder_ratio": m.wrist_height_shoulder_ratio,
                "wrist_height_nose_ratio": m.wrist_height_nose_ratio,
                "above_head": m.above_head,
                "contact_front_class": m.contact_front_class,
                "trunk_rotation_deg": m.trunk_rotation_deg,
                "non_racket_abduction_deg": m.non_racket_abduction_deg,
                "non_racket_abduction_time": m.non_racket_abduction_time,
                "elbow_extension_timing": m.elbow_extension_timing,
                "elbow_angle_at_contact": m.elbow_angle_at_contact,
                "hip_velocity_pre": m.hip_velocity_pre,
                "hip_velocity_post": m.hip_velocity_post,
                "suggestions": m.suggestions,
            }
        )

    with open(video_out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = summarize_feedback(metrics)
    print(f"=== {video_name} Summary ===")
    print("Top 3 issues:")
    for item in summary["issues"]:
        print("-", item)
    print("Top 3 improvements:")
    for item in summary["fixes"]:
        print("-", item)
    if debug:
        print(f"Detected {len(events)} swing events, hand={hand}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Badminton overhead clear analysis")
    parser.add_argument("--input", required=True, help="Path to video or folder")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--hand", default="auto", choices=["auto", "left", "right"], help="Racket hand")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    return parser.parse_args()


def main():
    args = parse_args()
    videos = gather_videos(args.input)
    if not videos:
        raise SystemExit("No video files found")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    all_reports = []
    for vid in videos:
        all_reports.append(process_video(vid, args.out, hand_pref=args.hand, debug=args.debug))
    # Save combined manifest
    with open(Path(args.out) / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
