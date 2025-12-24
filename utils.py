import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


LANDMARK_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class Keypoint:
    x: float
    y: float
    visibility: float


@dataclass
class PoseFrame:
    frame_idx: int
    time: float
    keypoints: Dict[str, Optional[Keypoint]]
    width: int
    height: int


def exponential_smooth(values: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    smoothed = np.copy(values)
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            smoothed[i] = smoothed[i - 1]
            continue
        if np.isnan(smoothed[i - 1]):
            smoothed[i] = values[i]
        else:
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def smooth_pose_sequence(frames: List[PoseFrame], alpha: float = 0.2) -> List[PoseFrame]:
    if not frames:
        return frames
    joint_names = LANDMARK_NAMES
    coords = {joint: {"x": [], "y": []} for joint in joint_names}
    for frame in frames:
        for joint in joint_names:
            kp = frame.keypoints.get(joint)
            coords[joint]["x"].append(np.nan if kp is None else kp.x)
            coords[joint]["y"].append(np.nan if kp is None else kp.y)

    for joint in joint_names:
        coords[joint]["x"] = exponential_smooth(np.array(coords[joint]["x"]), alpha)
        coords[joint]["y"] = exponential_smooth(np.array(coords[joint]["y"]), alpha)

    smoothed_frames: List[PoseFrame] = []
    for idx, frame in enumerate(frames):
        new_kps: Dict[str, Optional[Keypoint]] = {}
        for joint in joint_names:
            x = coords[joint]["x"][idx]
            y = coords[joint]["y"][idx]
            original = frame.keypoints.get(joint)
            vis = 0.0 if original is None else original.visibility
            if math.isnan(x) or math.isnan(y):
                new_kps[joint] = None
            else:
                new_kps[joint] = Keypoint(float(x), float(y), vis)
        smoothed_frames.append(
            PoseFrame(
                frame_idx=frame.frame_idx,
                time=frame.time,
                keypoints=new_kps,
                width=frame.width,
                height=frame.height,
            )
        )
    return smoothed_frames


def get_keypoint(frame: PoseFrame, name: str) -> Optional[Keypoint]:
    return frame.keypoints.get(name)


def midpoint(p1: Keypoint, p2: Keypoint) -> Tuple[float, float]:
    return (p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0


def vector(p1: Keypoint, p2: Keypoint) -> Tuple[float, float]:
    return p2.x - p1.x, p2.y - p1.y


def angle_between(p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
    """Returns angle at p2 formed by p1-p2-p3 in degrees."""
    a = np.array([p1.x - p2.x, p1.y - p2.y])
    b = np.array([p3.x - p2.x, p3.y - p2.y])
    dot = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    cos_val = np.clip(dot / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator
