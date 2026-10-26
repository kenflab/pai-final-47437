import os as _pai_os
__file__ = _pai_os.path.join(_pai_os.environ['PAI_POLICY_RUNTIME'], 'policy_server.py')
"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。渡されるのは以下の6キーのみ（画像2 + 固有受容4）。
                物体の絶対座標などは渡されない（画像から推定すること）:
                - "agentview_image": (128, 128, 3) uint8  — 正面カメラ
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8  — 手首カメラ
                - "robot0_joint_pos": (7,) float  — アーム関節角
                - "robot0_eef_pos": (3,) float  — エンドエフェクタ位置
                - "robot0_eef_quat": (4,) float  — エンドエフェクタ姿勢(quat)
                - "robot0_gripper_qpos": (2,) float  — グリッパ開度

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
                絶対座標ではなく **デルタ制御**（LIBERO の OSC_POSE コントローラ）。
                前 6 次元は EEF の相対移動・回転で通常 [-1, 1]、
                7 次元目は gripper 開閉指令。
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    """LeRobot pi0.5-LIBERO v0.4.4, offline and episode-deterministic."""

    FLOW_SEED = 2026
    EXECUTION_HORIZON = 5
    COMPILE_MODEL = False
    COMPILE_MODE = 'reduce-overhead'

    def __init__(self):
        import os
        import sys
        import time
        from pathlib import Path

        self.root = Path(__file__).resolve().parent
        vendor = self.root / "vendor"
        checkpoint = self.root / "model_weights" / "pi05_libero_finetuned_v044"
        tokenizer = self.root / "model_weights" / "paligemma_tokenizer"
        for path, label in (
            (vendor, "vendored Python runtime"),
            (checkpoint, "pi0.5 checkpoint"),
            (tokenizer, "PaliGemma tokenizer"),
        ):
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {label}: {path}")
        sys.path.insert(0, str(vendor))

        os.environ.update(
            {
                "HF_HOME": str(self.root / ".hf_home"),
                "HF_HUB_CACHE": str(self.root / ".hf_home" / "hub"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "DO_NOT_TRACK": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONNOUSERSITE": "1",
            }
        )

        import torch
        import torch.nn.functional as functional
        from lerobot.policies.pi05 import PI05Policy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is required for pi0.5; torch={torch.__version__}, "
                f"cuda_build={torch.version.cuda}"
            )
        self.torch = torch
        self.functional = functional
        self.time = time
        self.device = torch.device("cuda")
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("high")

        # The copied config is patched by the builder before loading.  Keep the
        # assertions here so an accidentally mixed checkpoint fails at startup.
        config_path = checkpoint / "config.json"
        import json

        config_json = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "type": "pi05",
            "chunk_size": 50,
            "n_action_steps": 50,
            "num_inference_steps": 10,
            "dtype": "bfloat16",
            "compile_model": self.COMPILE_MODEL,
        }
        mismatches = {
            key: (value, config_json.get(key))
            for key, value in expected.items()
            if config_json.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Unexpected pi0.5 config values: {mismatches}")

        load_started = time.monotonic()
        self.policy = PI05Policy.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            strict=False,
        ).to(self.device)
        if not 1 <= int(self.EXECUTION_HORIZON) <= 50:
            raise RuntimeError(f"Invalid execution horizon: {self.EXECUTION_HORIZON}")
        # Keep the trained 50-step prediction chunk, but execute only a shorter
        # prefix before replanning from the latest observation.
        self.policy.config.n_action_steps = int(self.EXECUTION_HORIZON)
        self.policy.reset()
        self.policy.eval()

        # Load the processor files shipped with the checkpoint.  The direct
        # pi05 processor factory only creates new pipelines and its v0.4.4
        # signature does not accept a checkpoint path or overrides.
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint),
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides={
                "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
                "device_processor": {"device": "cuda"},
            },
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint),
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            overrides={
                "device_processor": {"device": "cpu"},
            },
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.instruction = ""

        # Warm up before /health is exposed.  This catches offline tokenizer,
        # dtype, CUDA-memory, and optional torch.compile failures before the
        # evaluator's first 10-second /act deadline.
        self._seed_episode()
        dummy = {
            "observation.images.image": torch.zeros(
                (3, 256, 256), dtype=torch.float32
            ),
            "observation.images.image2": torch.zeros(
                (3, 256, 256), dtype=torch.float32
            ),
            "observation.state": torch.zeros(8, dtype=torch.float32),
            "task": "do something",
        }
        with torch.inference_mode():
            processed = self.preprocessor(dummy)
            warmup = self.postprocessor(self.policy.select_action(processed))
        warmup_action = self._as_action(warmup)
        if warmup_action.shape != (7,):
            raise RuntimeError(f"Unexpected warmup action: {warmup_action.shape}")
        self.policy.reset()
        torch.cuda.synchronize()
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"pi05 load+warmup seconds={time.monotonic() - load_started:.3f} "
            f"peak_allocated_gib={peak_gib:.3f} "
            f"gpu={torch.cuda.get_device_name(0)!r} "
            f"compile={self.COMPILE_MODEL} execution_horizon={self.EXECUTION_HORIZON}",
            flush=True,
        )
        self.reset("")

    def _seed_episode(self) -> None:
        self.torch.manual_seed(self.FLOW_SEED)
        self.torch.cuda.manual_seed_all(self.FLOW_SEED)

    @staticmethod
    def _quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
        quat = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("Invalid end-effector quaternion")
        quat /= norm
        if quat[3] < 0.0:
            quat *= -1.0
        vector_norm = float(np.linalg.norm(quat[:3]))
        if vector_norm < 1e-8:
            vector = 2.0 * quat[:3]
        else:
            angle = 2.0 * np.arctan2(vector_norm, quat[3])
            vector = quat[:3] * (angle / vector_norm)
        return vector.astype(np.float32)

    def _image_tensor(self, image: np.ndarray):
        array = np.asarray(image)
        if array.shape != (128, 128, 3) or array.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 image (128,128,3), got {array.dtype} {array.shape}"
            )
        # Match the official OpenPI LIBERO transform: flip both image axes.
        array = np.ascontiguousarray(array[::-1, ::-1, :])
        tensor = self.torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        # The v0.4.4 processor metadata expects (3,256,256); the policy then
        # performs its own PaliGemma-side 224x224 preparation.
        tensor = self.functional.interpolate(
            tensor.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return tensor.contiguous()

    def _frame(self, obs: dict[str, np.ndarray]) -> dict:
        required = (
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Missing observations: {missing}")
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
                self._quat_xyzw_to_axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
            )
        ).astype(np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Robot state contains NaN or Inf")
        return {
            "observation.images.image": self._image_tensor(obs["agentview_image"]),
            "observation.images.image2": self._image_tensor(
                obs["robot0_eye_in_hand_image"]
            ),
            "observation.state": self.torch.from_numpy(state),
            "task": self.instruction,
        }

    @staticmethod
    def _as_action(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        action = np.asarray(value, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise RuntimeError(f"Expected pi0.5 action (7,), got {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError("pi0.5 action contains NaN or Inf")
        return action

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if self._g09h_overlap.needs_chunk():
            frame = self._frame(obs)
            with self.torch.inference_mode():
                processed = self.preprocessor(frame)
                decoded_samples = []
                for _ in range(3):
                    raw_chunk = self.policy.predict_action_chunk(processed)
                    if tuple(raw_chunk.shape) != (1, 50, 7):
                        raise RuntimeError('Expected pi0.5 chunk (1,50,7)')
                    decoded_samples.append(np.stack([
                        self._as_action(self.postprocessor(raw_chunk[:, j].clone()))
                        for j in range(20)
                    ]).astype(np.float32, copy=False))
                from g09h_overlap import consensus
                self._g09h_overlap.push(consensus(np.stack(decoded_samples)))
        return self._g09h_overlap.pop()

    def reset(self, instruction: str = "") -> None:
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        self.instruction = instruction.strip()
        self.policy.reset()
        self._seed_episode()

        from g09h_overlap import OverlapActions
        self._g09h_overlap = OverlapActions(old_weight=0.4, horizon=5, smooth=True, ema_alpha=1.0, transition_replan=False)
