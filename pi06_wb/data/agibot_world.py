"""AgiBot World 2026 data loader for Pi0.6-WB training.

Dataset format: LeRobot v2.1 (Parquet + MP4 videos)
Robot: AgiBot G2 — dual 7-DOF arms, 2D grippers, 2D head, 2D waist, 4-wheel base

Field mappings (verified from dataset card):
    State:
        observation.state          — flattened state vector (joint positions + base state)
        Subfields defined in meta/info.json field_descriptions with indices

    Actions:
        action                     — flattened manipulation action vector (joint/position)
        Subfields: action/joint/position (20D), action/robot/velocity (3D)

    Images:
        observation.images.top_head
        observation.images.hand_left
        observation.images.hand_right
        (+ others available but not used by default)

    Instructions:
        From meta/tasks.jsonl (task_index → instruction string)
        Or from meta/info.json instruction_segments (subtask-level)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io as tv_io
import torchvision.transforms.functional as TF


# G2 action space dimensions (from dataset card)
MANIP_ACTION_DIM = 20   # action/joint/position: 14D arms + 2D grippers + 2D head + 2D waist
LOCO_ACTION_DIM  = 3    # action/robot/velocity: [vx, vy, omega_z]

# G2 state dimensions
STATE_DIM        = 25   # observation.state: ~14D arm joints + 2D gripper + 3D base pos + 6D base vel

# Default cameras (order matters — matches Pi0WBConfig.image_keys)
DEFAULT_IMAGE_KEYS = (
    "observation.images.top_head",      # → "base_0_rgb"
    "observation.images.hand_left",     # → "left_wrist_0_rgb"
    "observation.images.hand_right",    # → "right_wrist_0_rgb"
)

IMAGE_RESOLUTION = (224, 224)  # matches openpi preprocessing


class AgiBotWorldDataset(Dataset):
    """Dataset for AgiBot World 2026 in LeRobot v2.1 format.

    Each item returns:
        observation: dict with
            state:                  [state_dim] float32
            images: dict of
                base_0_rgb:         [3, H, W] float32 in [-1, 1]
                left_wrist_0_rgb:   [3, H, W] float32 in [-1, 1]
                right_wrist_0_rgb:  [3, H, W] float32 in [-1, 1]
            tokenized_prompt:       [max_token_len] int32
            tokenized_prompt_mask:  [max_token_len] bool
            image_masks: dict of
                base_0_rgb: bool scalar
                ...
        manip_actions:  [action_horizon, manip_action_dim] float32
        loco_actions:   [loco_action_dim] float32  (single command, not chunked)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        tokenizer,
        action_horizon: int = 50,
        max_token_len: int = 48,
        state_dim: int = STATE_DIM,
        manip_action_dim: int = MANIP_ACTION_DIM,
        loco_action_dim: int = LOCO_ACTION_DIM,
        image_keys: tuple = DEFAULT_IMAGE_KEYS,
        use_subtask_split: bool = True,
        norm_stats: dict | None = None,
        train: bool = True,
    ):
        self.root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.state_dim = state_dim
        self.manip_action_dim = manip_action_dim
        self.loco_action_dim = loco_action_dim
        self.image_keys = image_keys
        self.norm_stats = norm_stats
        self.train = train

        # Load dataset metadata
        self.info = self._load_info()
        self.episodes = self._load_episodes()
        self.tasks = self._load_tasks()

        # Build frame index: list of (episode_index, frame_index) for valid training frames
        # We need at least action_horizon frames ahead for action chunking
        self.frame_index = self._build_frame_index(use_subtask_split)

    def _load_info(self) -> dict:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"meta/info.json not found at {self.root}")
        with open(info_path) as f:
            return json.load(f)

    def _load_episodes(self) -> list[dict]:
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"meta/episodes.jsonl not found at {self.root}")
        episodes = []
        with open(episodes_path) as f:
            for line in f:
                episodes.append(json.loads(line.strip()))
        return episodes

    def _load_tasks(self) -> dict[int, str]:
        """Returns task_index → instruction string."""
        tasks_path = self.root / "meta" / "tasks.jsonl"
        tasks = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    t = json.loads(line.strip())
                    tasks[t.get("task_index", len(tasks))] = t.get("task", "")
        return tasks

    def _build_frame_index(self, use_subtask_split: bool) -> list[tuple[int, int]]:
        """Build list of valid (episode_idx, start_frame) pairs.

        A frame is valid as a training start if there are >= action_horizon frames
        remaining in the episode from that frame.
        """
        index = []

        # Check for instruction_segments (subtask-level splits) in info.json
        instruction_segments = self.info.get("instruction_segments", {})

        for ep in self.episodes:
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]

            if use_subtask_split and str(ep_idx) in instruction_segments:
                # Use subtask boundaries — each segment has its own instruction
                for seg in instruction_segments[str(ep_idx)]:
                    start = seg.get("start_frame_index", 0)
                    end   = seg.get("end_frame_index", ep_len)
                    # Add frames where we can fit a full action chunk
                    for f in range(start, end - self.action_horizon + 1):
                        index.append((ep_idx, f, seg.get("instruction", "")))
            else:
                # Use full episode with episode-level instruction
                instruction = self.tasks.get(ep.get("task_index", -1), "")
                for f in range(ep_len - self.action_horizon + 1):
                    index.append((ep_idx, f, instruction))

        return index

    def _get_chunk_path(self, episode_idx: int) -> Path:
        """Get path to parquet file for this episode."""
        chunk_idx = episode_idx // 1000  # LeRobot chunks by 1000 episodes
        return self.root / "data" / f"chunk-{chunk_idx:06d}" / f"episode_{episode_idx:06d}.parquet"

    def _get_video_path(self, episode_idx: int, camera_key: str) -> Path:
        chunk_idx = episode_idx // 1000
        return self.root / "videos" / f"chunk-{chunk_idx:06d}" / camera_key / f"episode_{episode_idx:06d}.mp4"

    def _load_episode_data(self, episode_idx: int) -> pd.DataFrame:
        parquet_path = self._get_chunk_path(episode_idx)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
        return pd.read_parquet(parquet_path)

    def _load_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> torch.Tensor:
        """Load a single frame from video. Returns [3, H, W] float32 in [-1, 1]."""
        video_path = self._get_video_path(episode_idx, camera_key)
        if not video_path.exists():
            # Return black frame if camera not available
            return torch.zeros(3, *IMAGE_RESOLUTION, dtype=torch.float32)

        # torchvision.io.read_video is slow for single frames; use pts-based seek
        pts = frame_idx / self.info.get("fps", 30.0)
        try:
            frames, _, _ = tv_io.read_video(
                str(video_path),
                start_pts=pts,
                end_pts=pts + 1.0 / self.info.get("fps", 30.0),
                pts_unit="sec",
            )
            if frames.shape[0] == 0:
                return torch.zeros(3, *IMAGE_RESOLUTION, dtype=torch.float32)
            img = frames[0]  # [H, W, 3] uint8
        except Exception:
            return torch.zeros(3, *IMAGE_RESOLUTION, dtype=torch.float32)

        img = img.permute(2, 0, 1).float() / 127.5 - 1.0  # [3, H, W], [-1, 1]
        img = TF.resize(img, list(IMAGE_RESOLUTION))
        return img

    def _parse_action_vector(self, action_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split flattened action vector into manipulation + locomotion.

        AgiBot World 2026 action field_descriptions (from dataset card):
            action/joint/position: indices [0, 20)  — 20D manipulation
            action/robot/velocity: indices [20, 23) — 3D base velocity

        NOTE: Verify exact indices from meta/info.json field_descriptions before training.
        These are best-estimate based on dataset card documentation.
        """
        # Try to get indices from info.json field_descriptions
        field_descs = self.info.get("features", {}).get("action", {}).get("field_descriptions", [])
        manip_indices = None
        loco_indices = None

        for fd in field_descs:
            desc = fd.get("description", "").lower()
            if "joint" in desc and "position" in desc:
                manip_indices = fd.get("indices", [0, self.manip_action_dim])
            elif "robot" in desc and "velocity" in desc:
                loco_indices = fd.get("indices", [self.manip_action_dim, self.manip_action_dim + self.loco_action_dim])

        # Fallback to defaults if not found in info.json
        if manip_indices is None:
            manip_indices = [0, self.manip_action_dim]
        if loco_indices is None:
            loco_indices = [self.manip_action_dim, self.manip_action_dim + self.loco_action_dim]

        manip = action_vec[manip_indices[0]:manip_indices[1]]
        loco  = action_vec[loco_indices[0]:loco_indices[1]]
        return manip, loco

    def _normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        if self.norm_stats is None or key not in self.norm_stats:
            return x
        stats = self.norm_stats[key]
        mean = stats.get("mean", 0.0)
        std  = stats.get("std", 1.0)
        return (x - mean) / (std + 1e-8)

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx, instruction = self.frame_index[idx]

        # Load parquet data for this episode
        df = self._load_episode_data(ep_idx)

        # State: single frame
        state_row = df.iloc[frame_idx]["observation.state"]
        if isinstance(state_row, (list, np.ndarray)):
            state = np.array(state_row, dtype=np.float32)
        else:
            state = np.zeros(self.state_dim, dtype=np.float32)

        # Pad/truncate state to state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
        state = self._normalize(state, "state")

        # Actions: chunk of action_horizon frames
        manip_actions = []
        loco_actions_list = []
        for i in range(self.action_horizon):
            action_frame = min(frame_idx + i, len(df) - 1)
            action_row = df.iloc[action_frame]["action"]
            if isinstance(action_row, (list, np.ndarray)):
                action_vec = np.array(action_row, dtype=np.float32)
            else:
                action_vec = np.zeros(self.manip_action_dim + self.loco_action_dim, dtype=np.float32)

            manip, loco = self._parse_action_vector(action_vec)

            # Pad/truncate to expected dims
            if len(manip) < self.manip_action_dim:
                manip = np.pad(manip, (0, self.manip_action_dim - len(manip)))
            manip_actions.append(manip[:self.manip_action_dim])
            loco_actions_list.append(loco[:self.loco_action_dim] if len(loco) >= self.loco_action_dim
                                     else np.pad(loco, (0, self.loco_action_dim - len(loco))))

        manip_actions = np.stack(manip_actions)   # [action_horizon, manip_action_dim]
        # Locomotion: use the command at the current frame (single command, not chunked)
        loco_action = loco_actions_list[0]          # [loco_action_dim]

        manip_actions = self._normalize(manip_actions, "manip_actions")
        loco_action   = self._normalize(loco_action, "loco_action")

        # Images: load from videos
        images = {}
        image_masks = {}
        cam_to_key = {
            "observation.images.top_head":   "base_0_rgb",
            "observation.images.hand_left":  "left_wrist_0_rgb",
            "observation.images.hand_right": "right_wrist_0_rgb",
        }
        for cam_key, model_key in cam_to_key.items():
            if cam_key in self.image_keys:
                img = self._load_frame_image(ep_idx, frame_idx, cam_key)
                images[model_key] = img
                image_masks[model_key] = torch.tensor(True)
            else:
                images[model_key] = torch.zeros(3, *IMAGE_RESOLUTION, dtype=torch.float32)
                image_masks[model_key] = torch.tensor(False)

        # Tokenize instruction
        tokens, token_mask = self._tokenize(instruction)

        return {
            "state": torch.from_numpy(state),
            "images": images,
            "image_masks": image_masks,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "manip_actions": torch.from_numpy(manip_actions),
            "loco_action": torch.from_numpy(loco_action),
            "instruction": instruction,
        }

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            # Return zeros if no tokenizer (for unit testing)
            tokens = torch.zeros(self.max_token_len, dtype=torch.int32)
            mask   = torch.zeros(self.max_token_len, dtype=torch.bool)
            return tokens, mask

        enc = self.tokenizer(
            text,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = enc["input_ids"].squeeze(0).to(torch.int32)
        mask   = enc["attention_mask"].squeeze(0).to(torch.bool)
        return tokens, mask


def compute_norm_stats(dataset_root: str | Path, num_samples: int = 10000) -> dict:
    """Compute mean/std normalization statistics from the dataset.

    Call this once before training and save to a JSON file.
    """
    import random
    from pathlib import Path

    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"

    with open(info_path) as f:
        info = json.load(f)
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))

    all_states, all_manip, all_loco = [], [], []
    sampled = 0

    for ep in random.sample(episodes, min(len(episodes), 100)):
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // 1000
        parquet_path = root / "data" / f"chunk-{chunk_idx:06d}" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            state = np.array(row["observation.state"], dtype=np.float32)
            action = np.array(row["action"], dtype=np.float32)
            manip = action[:MANIP_ACTION_DIM]
            loco  = action[MANIP_ACTION_DIM:MANIP_ACTION_DIM + LOCO_ACTION_DIM]
            all_states.append(state[:STATE_DIM])
            all_manip.append(manip)
            all_loco.append(loco)
            sampled += 1
            if sampled >= num_samples:
                break
        if sampled >= num_samples:
            break

    def stats(arr):
        a = np.stack(arr)
        return {"mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist()}

    return {
        "state":        stats(all_states),
        "manip_actions": stats(all_manip),
        "loco_action":   stats(all_loco),
    }
