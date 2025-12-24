from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

from utils import PoseFrame, get_keypoint

Hand = Literal["left", "right"]


@dataclass
class SwingEvent:
    start_idx: int
    strike_idx: int
    end_idx: int
    hand: Hand


def _dominant_hand_from_sequence(frames: List[PoseFrame]) -> Hand:
    left_heights = []
    right_heights = []
    for frame in frames:
        lw = get_keypoint(frame, "left_wrist")
        rw = get_keypoint(frame, "right_wrist")
        if lw:
            left_heights.append(-lw.y)
        if rw:
            right_heights.append(-rw.y)
    left_peak = np.percentile(left_heights, 90) if left_heights else 0
    right_peak = np.percentile(right_heights, 90) if right_heights else 0
    return "left" if left_peak > right_peak else "right"


def choose_hand(frames: List[PoseFrame], prefer: Literal["auto", "left", "right"]) -> Hand:
    if prefer in ("left", "right"):
        return prefer
    return _dominant_hand_from_sequence(frames)


def _get_wrist_series(frames: List[PoseFrame], hand: Hand) -> np.ndarray:
    series = []
    name = "left_wrist" if hand == "left" else "right_wrist"
    for frame in frames:
        kp = get_keypoint(frame, name)
        series.append(np.nan if kp is None else kp.y)
    return np.array(series, dtype=float)


def detect_swings(frames: List[PoseFrame], fps: float, hand: Hand) -> List[SwingEvent]:
    wrist_y = _get_wrist_series(frames, hand)
    if len(wrist_y) < 3:
        return []
    # Lower y means higher hand in image coordinates.
    speed = np.gradient(wrist_y)
    speed_mag = np.abs(speed)
    if np.nanmax(speed_mag) == 0:
        return []
    speed_thresh = np.nanmax(speed_mag) * 0.45
    min_gap_frames = int(0.5 * fps) if fps else 10

    candidates: List[int] = []
    for idx in range(1, len(wrist_y) - 1):
        if np.isnan(wrist_y[idx]) or np.isnan(speed_mag[idx]):
            continue
        if speed_mag[idx] < speed_thresh:
            continue
        local_min = wrist_y[idx] <= np.nanmin(wrist_y[max(0, idx - 3) : idx + 4])
        if local_min:
            if candidates and idx - candidates[-1] < min_gap_frames:
                continue
            candidates.append(idx)

    events: List[SwingEvent] = []
    for peak_idx in candidates:
        start_idx = max(0, peak_idx - int(0.35 * fps))
        end_idx = min(len(frames) - 1, peak_idx + int(0.45 * fps))
        # strike frame: highest speed near peak
        window_start = max(0, peak_idx - int(0.1 * fps))
        window_end = min(len(frames) - 1, peak_idx + int(0.1 * fps))
        local_speed = speed_mag[window_start : window_end + 1]
        strike_local = int(np.nanargmax(local_speed))
        strike_idx = window_start + strike_local
        events.append(SwingEvent(start_idx=start_idx, strike_idx=strike_idx, end_idx=end_idx, hand=hand))
    return events
