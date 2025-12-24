import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from clear_events import SwingEvent
from clear_metrics import EventMetrics
from utils import LANDMARK_NAMES, PoseFrame, get_keypoint


POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _draw_skeleton(frame_bgr, pose_frame: PoseFrame, color=(0, 255, 0)):
    for joint_a, joint_b in POSE_CONNECTIONS:
        a = get_keypoint(pose_frame, joint_a)
        b = get_keypoint(pose_frame, joint_b)
        if a and b:
            cv2.line(frame_bgr, (int(a.x), int(a.y)), (int(b.x), int(b.y)), color, 2)
    for name in LANDMARK_NAMES:
        kp = get_keypoint(pose_frame, name)
        if kp:
            cv2.circle(frame_bgr, (int(kp.x), int(kp.y)), 4, color, -1)


def render_overlay_video(
    input_path: str,
    frames: List[PoseFrame],
    events: List[SwingEvent],
    metrics: List[EventMetrics],
    out_path: str,
):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    frame_idx = 0
    event_lookup = {m.event.strike_idx: (m, e) for m, e in zip(metrics, events)}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(frames):
            pf = frames[frame_idx]
            _draw_skeleton(frame, pf, (0, 200, 0))
            if frame_idx in event_lookup:
                m, e = event_lookup[frame_idx]
                wrist_name = "right_wrist" if e.hand == "right" else "left_wrist"
                wrist_kp = get_keypoint(pf, wrist_name)
                cv2.putText(
                    frame,
                    f"Strike score {m.score:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                if wrist_kp:
                    cv2.circle(frame, (int(wrist_kp.x), int(wrist_kp.y)), 10, (0, 0, 255), 3)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()


def render_summary_plot(frames: List[PoseFrame], events: List[SwingEvent], out_path: str):
    times = [f.time for f in frames]
    right_wrist_y = [get_keypoint(f, "right_wrist").y if get_keypoint(f, "right_wrist") else np.nan for f in frames]
    left_wrist_y = [get_keypoint(f, "left_wrist").y if get_keypoint(f, "left_wrist") else np.nan for f in frames]
    plt.figure(figsize=(10, 4))
    plt.plot(times, right_wrist_y, label="Right wrist y")
    plt.plot(times, left_wrist_y, label="Left wrist y", linestyle="--")
    for e in events:
        plt.axvline(frames[e.strike_idx].time, color="red", linestyle=":", alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel("Time (s)")
    plt.ylabel("Wrist height (pixels, inverted)")
    plt.title("Wrist trajectory & strike moments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
