import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import Keypoint, PoseFrame, LANDMARK_NAMES, smooth_pose_sequence


mp_pose = mp.solutions.pose


@dataclass
class PoseSequence:
    frames: List[PoseFrame]
    fps: float


class PoseExtractor:
    def __init__(self, smooth_alpha: float = 0.2, min_visibility: float = 0.35):
        self.smooth_alpha = smooth_alpha
        self.min_visibility = min_visibility

    def _landmark_to_keypoint(
        self, lm: mp.framework.formats.landmark_pb2.NormalizedLandmark, width: int, height: int
    ) -> Keypoint:
        return Keypoint(x=lm.x * width, y=lm.y * height, visibility=lm.visibility)

    def extract(self, video_path: str) -> PoseSequence:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames: List[PoseFrame] = []

        with mp_pose.Pose(model_complexity=1, enable_segmentation=False) as pose:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                keypoints = {name: None for name in LANDMARK_NAMES}
                if results.pose_landmarks:
                    for name, landmark in zip(LANDMARK_NAMES, results.pose_landmarks.landmark):
                        kp = self._landmark_to_keypoint(landmark, width, height)
                        if kp.visibility >= self.min_visibility:
                            keypoints[name] = kp
                        else:
                            keypoints[name] = None
                time = idx / fps if fps else 0.0
                frames.append(PoseFrame(frame_idx=idx, time=time, keypoints=keypoints, width=width, height=height))
                idx += 1
        cap.release()
        frames = smooth_pose_sequence(frames, alpha=self.smooth_alpha)
        return PoseSequence(frames=frames, fps=fps)


def save_pose_csv(sequence: PoseSequence, out_path: str) -> None:
    header = ["frame", "time"]
    for name in LANDMARK_NAMES:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_vis"])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for frame in sequence.frames:
            row = [frame.frame_idx, f"{frame.time:.4f}"]
            for name in LANDMARK_NAMES:
                kp = frame.keypoints.get(name)
                if kp is None:
                    row.extend(["", "", "0.0"])
                else:
                    row.extend([f"{kp.x:.2f}", f"{kp.y:.2f}", f"{kp.visibility:.3f}"])
            f.write(",".join(map(str, row)) + "\n")
