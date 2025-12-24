from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from clear_events import SwingEvent
from utils import Keypoint, PoseFrame, angle_between, get_keypoint, midpoint, safe_ratio, vector


Hand = Literal["left", "right"]


@dataclass
class EventMetrics:
    event: SwingEvent
    wrist_height_shoulder_ratio: float
    wrist_height_nose_ratio: float
    above_head: bool
    contact_front_class: str
    trunk_rotation_deg: float
    non_racket_abduction_deg: float
    non_racket_abduction_time: float
    elbow_extension_timing: float
    elbow_angle_at_contact: float
    hip_velocity_pre: float
    hip_velocity_post: float
    score: float
    suggestions: List[str]


def _get_torso_axis(frame: PoseFrame) -> Optional[Tuple[float, float]]:
    ls = get_keypoint(frame, "left_shoulder")
    rs = get_keypoint(frame, "right_shoulder")
    lh = get_keypoint(frame, "left_hip")
    rh = get_keypoint(frame, "right_hip")
    if not (ls and rs and lh and rh):
        return None
    center_shoulder = midpoint(ls, rs)
    center_hip = midpoint(lh, rh)
    # return hip center for front/back judgement
    return center_hip


def _orientation(frame: PoseFrame, a_name: str, b_name: str) -> Optional[float]:
    a = get_keypoint(frame, a_name)
    b = get_keypoint(frame, b_name)
    if not a or not b:
        return None
    dx, dy = vector(a, b)
    return float(np.degrees(np.arctan2(dy, dx)))


def _hip_velocity(frames: List[PoseFrame], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    hip_points = []
    times = []
    for idx in range(start, end):
        lh = get_keypoint(frames[idx], "left_hip")
        rh = get_keypoint(frames[idx], "right_hip")
        if lh and rh:
            cx, cy = midpoint(lh, rh)
            hip_points.append((cx, cy))
            times.append(frames[idx].time)
    if len(hip_points) < 2:
        return 0.0
    distances = [np.linalg.norm(np.array(hip_points[i + 1]) - np.array(hip_points[i])) for i in range(len(hip_points) - 1)]
    dt = times[-1] - times[0]
    return float(sum(distances) / dt) if dt > 0 else 0.0


def _elbow_angle_series(frames: List[PoseFrame], hand: Hand) -> List[float]:
    angles = []
    names = ("left_shoulder", "left_elbow", "left_wrist") if hand == "left" else ("right_shoulder", "right_elbow", "right_wrist")
    for f in frames:
        a, b, c = (get_keypoint(f, name) for name in names)
        if a and b and c:
            angles.append(angle_between(a, b, c))
        else:
            angles.append(np.nan)
    return angles


def classify_contact_front(frame: PoseFrame, hand: Hand) -> Optional[str]:
    wrist = get_keypoint(frame, "left_wrist" if hand == "left" else "right_wrist")
    torso_axis = _get_torso_axis(frame)
    if not wrist or torso_axis is None:
        return None
    hip_mid_x, _ = torso_axis
    offset = wrist.x - hip_mid_x
    if offset < -30:
        return "偏后"
    if offset > 30:
        return "偏前"
    return "合适"


def compute_event_metrics(frames: List[PoseFrame], event: SwingEvent) -> EventMetrics:
    fps_est = 30.0
    if len(frames) > 1:
        dt = frames[1].time - frames[0].time
        if dt > 0:
            fps_est = 1.0 / dt
    start_f = frames[event.start_idx]
    strike_f = frames[event.strike_idx]
    hand = event.hand
    wrist = get_keypoint(strike_f, "left_wrist" if hand == "left" else "right_wrist")
    shoulder = get_keypoint(strike_f, "left_shoulder" if hand == "left" else "right_shoulder")
    nose = get_keypoint(strike_f, "nose")
    wrist_height_shoulder_ratio = 0.0
    wrist_height_nose_ratio = 0.0
    above_head = False
    if wrist and shoulder:
        wrist_height_shoulder_ratio = safe_ratio((shoulder.y - wrist.y), strike_f.height, 0.0)
        above_head = wrist.y < shoulder.y
    if wrist and nose:
        wrist_height_nose_ratio = safe_ratio((nose.y - wrist.y), strike_f.height, 0.0)

    contact_front_class = classify_contact_front(strike_f, hand) or "未知"

    prep_orientation = _orientation(start_f, "left_shoulder", "right_shoulder")
    strike_orientation = _orientation(strike_f, "left_shoulder", "right_shoulder")
    prep_hip_orientation = _orientation(start_f, "left_hip", "right_hip")
    strike_hip_orientation = _orientation(strike_f, "left_hip", "right_hip")
    rotation_vals = []
    for pair in [(prep_orientation, strike_orientation), (prep_hip_orientation, strike_hip_orientation)]:
        if pair[0] is not None and pair[1] is not None:
            rotation_vals.append(abs(pair[1] - pair[0]))
    trunk_rotation_deg = float(np.nanmean(rotation_vals)) if rotation_vals else 0.0

    non_hand = "right" if hand == "left" else "left"
    non_shoulder = get_keypoint(strike_f, f"{non_hand}_shoulder")
    non_elbow = get_keypoint(strike_f, f"{non_hand}_elbow")
    non_wrist = get_keypoint(strike_f, f"{non_hand}_wrist")
    non_abduction_deg = 0.0
    if non_shoulder and non_elbow and non_wrist:
        non_abduction_deg = angle_between(non_elbow, non_shoulder, non_wrist)
    window_frames = frames[event.start_idx : event.strike_idx + 1]
    abduction_series = []
    for f in window_frames:
        ns, ne, nw = (get_keypoint(f, f"{non_hand}_{pt}") for pt in ["shoulder", "elbow", "wrist"])
        if ns and ne and nw:
            abduction_series.append(angle_between(ne, ns, nw))
    non_racket_abduction_deg = float(np.nanmax(abduction_series)) if abduction_series else 0.0
    non_racket_abduction_time = frames[event.start_idx + int(np.nanargmax(abduction_series))].time if abduction_series else strike_f.time

    elbow_angles = np.array(_elbow_angle_series(frames[event.start_idx : event.end_idx + 1], hand))
    elbow_angle_at_contact = 0.0
    elbow_extension_timing = 0.0
    if elbow_angles.size and not np.all(np.isnan(elbow_angles)):
        contact_idx = min(event.strike_idx - event.start_idx, len(elbow_angles) - 1)
        elbow_angle_at_contact = float(np.nan_to_num(elbow_angles[contact_idx]))
        elbow_min_idx = int(np.nanargmin(elbow_angles))
        elbow_extension_timing = frames[event.start_idx + elbow_min_idx].time - strike_f.time

    pre_start = max(event.start_idx, event.strike_idx - int(0.5 * fps_est))
    post_start = min(len(frames) - 1, event.strike_idx + int(0.1 * fps_est))
    hip_velocity_pre = _hip_velocity(frames, pre_start, event.strike_idx)
    hip_velocity_post = _hip_velocity(frames, post_start, min(event.end_idx, post_start + int(0.5 * fps_est)))

    score_components = []
    score_components.append(min(1.0, max(0.0, wrist_height_shoulder_ratio * 2.5)))
    score_components.append(1.0 if contact_front_class == "合适" else 0.6 if contact_front_class == "偏前" else 0.4)
    score_components.append(min(1.0, trunk_rotation_deg / 70))
    score_components.append(min(1.0, non_racket_abduction_deg / 120))
    score_components.append(min(1.0, max(0.0, (180 - abs(160 - elbow_angle_at_contact)) / 180)))
    score = float(np.mean(score_components) * 100)

    suggestions = []
    if not above_head:
        suggestions.append("击球点偏低：尝试更早侧身引拍，击球时手腕高度至少高于头顶。")
    if contact_front_class == "偏后":
        suggestions.append("击球点偏后：击球前最后一步再往后/往侧调整，让击球发生在身体前上方。")
    if trunk_rotation_deg < 25:
        suggestions.append("侧身不足：准备期让非持拍肩更朝网，髋部先转再带动上肢。")
    if non_racket_abduction_deg < 60:
        suggestions.append("非持拍手没打开：准备期把非持拍手抬起指向来球，帮助身体打开与稳定。")
    if elbow_angle_at_contact < 150:
        suggestions.append("肘部过早伸直：保持肘角到加速期再快速伸展，避免推球感。")
    if hip_velocity_post < hip_velocity_pre * 0.5:
        suggestions.append("击球后回位不足：击球后快速小步回位，保持重心活跃。")

    return EventMetrics(
        event=event,
        wrist_height_shoulder_ratio=wrist_height_shoulder_ratio,
        wrist_height_nose_ratio=wrist_height_nose_ratio,
        above_head=above_head,
        contact_front_class=contact_front_class,
        trunk_rotation_deg=trunk_rotation_deg,
        non_racket_abduction_deg=non_racket_abduction_deg,
        non_racket_abduction_time=non_racket_abduction_time,
        elbow_extension_timing=elbow_extension_timing,
        elbow_angle_at_contact=elbow_angle_at_contact,
        hip_velocity_pre=hip_velocity_pre,
        hip_velocity_post=hip_velocity_post,
        score=score,
        suggestions=suggestions,
    )
