"""Evaluation script using the original diffusion_policy PushT environment.

This uses the exact same environment as the original diffusion_policy paper
with pymunk 6.2.1 for accurate dataset replay.

Server API:
- POST /predict: Run policy inference
  - Input: {
      "image": "<base64_png>",        # 96x96 RGB image encoded as base64 PNG
      "prompt": "<task_description>", # e.g., "PushT task"
      "domain_name": "<domain_name>", # e.g., "pusht"
      "image_size": <int>             # e.g., 96
    }
  - Output: {
      "action": [[a0_0, a0_1, ...], ..., [aN_0, aN_1, ...]],  # (N, D) normalized in [-1, 1]
      "video": ["<base64_png>", ...]  # List of T base64-encoded PNG frames (optional)
    }

- POST /reset: Reset episode state (for dataset replay)
  - Input: {"episode_idx": <int>}
  - Output: {}

- GET /info: Get server information
  - Output: {
      "run_name": "<str>",      # Used for output directory naming
      "checkpoint": "<str>"     # Model checkpoint path or identifier
    }

Run with:
    conda activate diffusion_pusht
    python -m eval_pusht.eval --server-url http://localhost:8000/predict --replay-dataset
    python -m eval_pusht.eval --server-url http://localhost:8000/predict --num-episodes 1 --debug
    python -m eval_pusht.eval --server-url http://localhost:8000/predict --num-episodes 10
    python -m eval_pusht.eval --server-url http://localhost:8000/predict --num-episodes 10 --num-rollouts 10

    # For UVA models that predict absolute states (not deltas):
    python -m eval_pusht.eval --server-url http://localhost:8000/predict --action-space absolute --direct-pos --num-episodes 10
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import requests


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

# Add diffusion_policy to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from diffusion_policy.env.pusht.pusht_env import PushTEnv


def compute_pusht_coverage(env: PushTEnv) -> float:
    """Compute current PushT block coverage against the goal."""
    from shapely.geometry import MultiPolygon, Polygon

    def get_block_geom(body: object, shapes: object) -> MultiPolygon:
        geoms = []
        for shape in shapes:
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(Polygon(verts))
        return MultiPolygon(geoms)

    goal_body = env._get_goal_pose_body(env.goal_pose)
    goal_geom = get_block_geom(goal_body, env.block.shapes)
    block_geom = get_block_geom(env.block, env.block.shapes)
    return float(goal_geom.intersection(block_geom).area / goal_geom.area)


class SimpleInferenceClient:
    """Simple inference client for the mock server."""

    def __init__(self, server_url: str, action_norm_range: float = 512.0, timeout: float = 300.0,
                 action_space: str = "relative"):
        self.server_url = server_url
        self.base_url = server_url.rsplit("/", 1)[0]
        self.action_norm_range = action_norm_range
        self.timeout = timeout
        self.session = requests.Session()
        self.image_size = 96  # Default image size, can be changed to 256 for res256 models
        self.action_space = action_space  # "relative" or "absolute"

    def get_server_info(self) -> dict:
        try:
            response = self.session.get(f"{self.base_url}/info", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}


    def reset_episode(self, episode_idx: int = 0) -> dict:
        try:
            response = self.session.post(
                f"{self.base_url}/reset",
                json={"episode_idx": episode_idx},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}

    def get_action_chunk(
        self,
        image: np.ndarray,
        prompt: str = "PushT task",
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Get action chunk from server.
        
        Returns:
            Tuple of (raw_actions, video_frames) where raw_actions are normalized in [-1, 1]
            and video_frames is a list of decoded images.
        """
        import base64
        import io
        from PIL import Image

        # Encode image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        # Upsample to target image_size if needed (e.g., 96 -> 256 for res256 models)
        if self.image_size != pil_image.size[0]:
            pil_image = pil_image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        payload = {
            "image": encoded,
            "prompt": prompt,
            "domain_name": "pusht",
            "image_size": self.image_size,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        # Send request
        response = self.session.post(
            self.server_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        action = np.array(result["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        # Slice to first 2 dimensions (x, y) - server may return more dimensions
        action = action[:, :2]

        # Decode video frames if present
        video_frames = []
        if "video" in result and result["video"]:
            for frame_b64 in result["video"]:
                try:
                    frame_data = base64.b64decode(frame_b64)
                    frame_img = Image.open(io.BytesIO(frame_data))
                    video_frames.append(np.array(frame_img))
                except Exception:
                    pass

        return action, video_frames

    def wait_until_ready(self, timeout: float = 600.0, interval: float = 5.0) -> None:
        """Wait for the model server health endpoint before starting rollouts."""
        import time

        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            try:
                response = self.session.get(f"{self.base_url}/", timeout=min(interval, self.timeout))
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(interval)
        raise TimeoutError(f"Server did not become ready within {timeout:.1f}s: {last_error}")

    def to_env_action(self, raw_action: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
        """Convert raw server output to absolute position for env.step().

        For 'relative' action_space (default):
            Server outputs normalized deltas in [-1, 1].
            absolute = delta * norm_range + agent_pos

        For 'absolute' action_space:
            Server outputs normalized absolute positions in [-1, 1].
            absolute = (raw + 1) / 2 * norm_range
            (agent_pos is NOT used — the prediction IS the target position.)
        """
        if self.action_space == "absolute":
            # PushT absolute coordinates are normalized as action / 512 * 2 - 1
            # during training, so invert that affine transform for the env.
            absolute = (raw_action + 1.0) * 0.5 * self.action_norm_range
        else:
            # Server predicts relative delta from agent_pos.
            absolute = raw_action * self.action_norm_range + agent_pos
        return np.clip(absolute, 0, self.action_norm_range)

    # Keep old name for backward compatibility
    def convert_to_absolute(self, delta: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
        """Legacy wrapper — use to_env_action() instead."""
        return self.to_env_action(delta, agent_pos)

    def close(self):
        self.session.close()


class DatasetReplayHelper:
    """Helper to load LeRobot dataset for replay."""

    def __init__(self, repo_id: str = "lerobot/pusht_image"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        print(f"[DatasetReplayHelper] Loading dataset: {repo_id}")
        self.dataset = LeRobotDataset(repo_id=repo_id)
        self._build_episode_index()
        print(f"[DatasetReplayHelper] Loaded {self.num_episodes} episodes")

    def _build_episode_index(self):
        self.episode_indices = {}
        # New lerobot API: iterate through dataset to build index
        print(f"[DatasetReplayHelper] Building episode index from {len(self.dataset)} samples...")
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            ep_idx = sample.get("episode_index", 0)
            if hasattr(ep_idx, "item"):
                ep_idx = ep_idx.item()
            if ep_idx not in self.episode_indices:
                self.episode_indices[ep_idx] = []
            self.episode_indices[ep_idx].append(idx)
        self.num_episodes = len(self.episode_indices)

    def get_initial_state(self, episode_idx: int) -> list:
        """Get initial state [agent_x, agent_y, block_x, block_y, block_angle]."""
        if episode_idx not in self.episode_indices:
            raise ValueError(f"Episode {episode_idx} not found")
        first_idx = self.episode_indices[episode_idx][0]
        sample = self.dataset[first_idx]
        state = sample.get("observation.state", None)
        if state is not None:
            if hasattr(state, "numpy"):
                state = state.numpy()
            return list(state)
        return None

    def get_frame(self, episode_idx: int, frame_idx: int) -> dict:
        """Get a frame from the dataset."""
        if episode_idx not in self.episode_indices:
            raise ValueError(f"Episode {episode_idx} not found")
        episode_indices = self.episode_indices[episode_idx]
        if frame_idx >= len(episode_indices):
            raise ValueError(f"Frame {frame_idx} out of range")
        dataset_idx = episode_indices[frame_idx]
        sample = self.dataset[dataset_idx]

        result = {}
        if "observation.image" in sample:
            img = sample["observation.image"]
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            result["image"] = img
        return result


def save_video(frames: list, output_path: str, fps: int = 10):
    """Save frames as MP4 video."""
    if not frames:
        return
    imageio.mimwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )


def save_gif(frames: list, output_path: str, fps: int = 10):
    """Save frames as GIF video."""
    if not frames:
        return
    imageio.mimsave(output_path, frames, fps=fps)


def concatenate_videos_horizontal(frames1: list, frames2: list) -> list:
    """Concatenate two frame lists horizontally.
    
    Args:
        frames1: First list of frames (left side)
        frames2: Second list of frames (right side)
    
    Returns:
        List of horizontally concatenated frames
    """
    if not frames1 or not frames2:
        return frames1 or frames2 or []
    
    # Use shorter length to ensure alignment
    min_len = min(len(frames1), len(frames2))
    
    combined = []
    for i in range(min_len):
        f1 = frames1[i]
        f2 = frames2[i]
        
        # Resize if heights don't match
        if f1.shape[0] != f2.shape[0]:
            from PIL import Image
            target_height = f1.shape[0]
            f2_pil = Image.fromarray(f2)
            new_width = int(f2.shape[1] * target_height / f2.shape[0])
            f2_pil = f2_pil.resize((new_width, target_height))
            f2 = np.array(f2_pil)
        
        # Concatenate horizontally
        combined.append(np.concatenate([f1, f2], axis=1))
    
    return combined


def run_episode(
    env: PushTEnv,
    client: SimpleInferenceClient,
    max_steps: int = 300,
    dataset_episode_idx: Optional[int] = None,
    dataset_helper: Optional[DatasetReplayHelper] = None,
    debug_mode: bool = False,
    augment_prompt: bool = False,
    direct_pos: bool = False,
    policy_seed: int | None = None,
) -> dict:
    """Run a single episode.

    Args:
        direct_pos: If True, use env.step_direct() to bypass PID controller.
                    Use when the model predicts target states (positions).
    """

    # Always reset server state at start of episode
    client.reset_episode(dataset_episode_idx if dataset_episode_idx is not None else 0)
    
    # Get initial state from dataset
    initial_state = None
    if dataset_episode_idx is not None and dataset_helper:
        initial_state = dataset_helper.get_initial_state(dataset_episode_idx)
        print(f"[REPLAY] Episode {dataset_episode_idx} initial state (agent_pos): {initial_state}")

    # Reset environment
    # First do a random reset to get block state
    env.reset_to_state = None
    obs = env.reset()
    
    # If we have agent position from dataset, reset with that agent pos + random block
    if initial_state and len(initial_state) >= 2:
        # Build full state: [agent_x, agent_y, block_x, block_y, block_angle]
        # Use agent pos from dataset, keep random block state
        full_state = [
            initial_state[0],  # agent_x from dataset
            initial_state[1],  # agent_y from dataset  
            obs[2],  # block_x from random reset
            obs[3],  # block_y from random reset
            obs[4],  # block_angle from random reset
        ]
        env.reset_to_state = full_state
        obs = env.reset()

    print(f"[ENV] After reset - agent: {obs[:2]}, block: {obs[2:4]}, angle: {obs[4]:.4f}")
    step_fn_name = "step_direct" if direct_pos else "step"
    print(f"[ENV] Action space: {client.action_space} | Step mode: {step_fn_name}")

    frames = []
    obs_frames = []
    dataset_frames = []
    server_frames = []  # Video frames from server response
    step = 0
    total_reward = 0.0
    success = False
    request_count = 0
    frame_idx = 0
    initial_coverage = compute_pusht_coverage(env)
    max_coverage = initial_coverage
    max_coverage_step = 0
    coverage_history = [initial_coverage]

    # Select the step function
    step_fn = env.step_direct if direct_pos else env.step

    # Determine prompt
    if augment_prompt:
        prompt = "You are given a task to push the green T into the yellow T region. Current prediction mode is policy. The video is 1.6 seconds long and is of 10 FPS."
    else:
        prompt = "PushT task"

    while step < max_steps:
        agent_pos = obs[:2]

        # Render observation (96x96)
        obs_img = env.render("rgb_array")
        obs_frames.append(obs_img.copy())

        request_count += 1

        # Get action chunk from server (raw normalized actions in [-1, 1])
        request_seed = None if policy_seed is None else policy_seed + request_count - 1
        raw_actions, video_frames = client.get_action_chunk(obs_img, prompt=prompt, seed=request_seed)
        # Skip the first frame (input image) as it duplicates the observation
        chunk_server_frames = video_frames[1:] if video_frames else []
        print(f"[Step {step}] Received {len(raw_actions)} actions from server (request #{request_count})")
        
        # Store agent position at chunk start — used as reference for relative deltas
        chunk_start_agent_pos = agent_pos.copy()

        # Execute all actions in chunk
        for i, raw in enumerate(raw_actions):
            
            # Convert raw server output → absolute env position
            action = client.to_env_action(raw, chunk_start_agent_pos)

            # Step environment (PID or direct depending on flag)
            obs, reward, done, info = step_fn(action)
            agent_pos = obs[:2]
            total_reward += reward
            current_coverage = compute_pusht_coverage(env)
            current_coverage_step = step + 1
            coverage_history.append(current_coverage)
            if current_coverage > max_coverage:
                max_coverage = current_coverage
                max_coverage_step = current_coverage_step

            # Record frame
            frame = env.render("rgb_array")
            frames.append(frame)
            
            # Add corresponding server frame (1:1 with rollout frames)
            if i < len(chunk_server_frames):
                server_frames.append(chunk_server_frames[i])
            
            # Get corresponding dataset frame for comparison
            if dataset_helper and dataset_episode_idx is not None:
                try:
                    ds_frame = dataset_helper.get_frame(dataset_episode_idx, step)
                    if "image" in ds_frame:
                        dataset_frames.append(ds_frame["image"].copy())
                except (ValueError, IndexError):
                    pass

            step += 1
            frame_idx += 1

            if done:
                success = True
                break

            if step >= max_steps:
                break
        
        # Debug mode: exit after first action chunk
        if debug_mode:
            print(f"[DEBUG] Executed {len(raw_actions)} actions from 1 request, exiting.")
            break

        if done or step >= max_steps:
            break

    final_coverage = compute_pusht_coverage(env)

    return {
        "frames": frames,
        "obs_frames": obs_frames,
        "dataset_frames": dataset_frames,
        "server_frames": server_frames,
        "total_reward": total_reward,
        "success": success,
        "steps": step,
        "final_coverage": final_coverage,
        "max_coverage": max_coverage,
        "max_coverage_step": max_coverage_step,
        "coverage_history": coverage_history,
        "num_requests": request_count,
    }


def evaluate(
    server_url: str,
    num_episodes: int = 1,
    max_steps: int = 300,
    output_dir: str = "eval_results",
    episode_start: int = 0,
    replay_dataset: bool = False,
    dataset_start_episode: int = 0,
    dataset_repo_id: str = "lerobot/pusht_image",
    debug_mode: bool = False,
    seed: Optional[int] = None,
    augment_prompt: bool = False,
    num_rollouts: int = 10,
    action_space: str = "relative",
    direct_pos: bool = False,
    wait_timeout: float = 600.0,
    save_gifs: bool = False,
):
    """Run evaluation with multiple rollouts per episode for robustness."""
    # Create original diffusion_policy environment
    env = PushTEnv(legacy=True, render_size=96)

    # Verify step_direct is available when requested
    if direct_pos and not hasattr(env, "step_direct"):
        raise RuntimeError(
            "env.step_direct() not found — make sure you have the patched pusht_env.py "
            "with step_direct() support."
        )

    # Create client
    client = SimpleInferenceClient(server_url=server_url, action_space=action_space)
    client.wait_until_ready(timeout=wait_timeout)

    # Load dataset
    dataset_helper = None
    if replay_dataset:
        dataset_helper = DatasetReplayHelper(repo_id=dataset_repo_id)

    # Get run name and checkpoint info from /info
    server_info = client.get_server_info()
    checkpoint_info = server_info.get("checkpoint", "")
    run_name = server_info.get("run_name", "diffusion_policy_eval")
    
    # Check if model uses 256 resolution based on run_name or checkpoint
    model_identifier = f"{run_name}_{checkpoint_info}".lower()
    if "res256" in model_identifier:
        client.image_size = 256
        print(f"[INFO] Detected res256 model, upsampling images to 256x256")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{run_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting evaluation: {num_episodes} episodes x {num_rollouts} rollouts")
    print(f"Episode start: {episode_start}")
    print(f"Image size: {client.image_size}x{client.image_size}")
    print(f"Action space: {action_space}")
    print(f"Direct position control: {direct_pos}")
    print(f"Server URL: {server_url}")
    print(f"Output directory: {output_path}")
    print(f"Using original diffusion_policy environment")
    print(f"pymunk version: {__import__('pymunk').version}")
    print("-" * 50)

    results = []
    total_successes = 0
    total_rollouts = 0

    try:
        for episode_idx in range(num_episodes):
            global_episode_idx = episode_start + episode_idx
            dataset_episode_idx = None
            if replay_dataset:
                dataset_episode_idx = dataset_start_episode + episode_idx

            episode_rollout_results = []
            episode_successes = 0

            for rollout_idx in range(num_rollouts):
                # Set different seed for each rollout while maintaining reproducibility
                env_seed = None
                policy_seed = None
                if seed is not None:
                    env_seed = seed + global_episode_idx
                    policy_seed = seed + global_episode_idx * num_rollouts + rollout_idx
                    set_seed(policy_seed)
                    env.seed(env_seed)  # Seed environment's internal RNG

                try:
                    result = run_episode(
                        env=env,
                        client=client,
                        max_steps=max_steps,
                        dataset_episode_idx=dataset_episode_idx,
                        dataset_helper=dataset_helper,
                        debug_mode=debug_mode,
                        augment_prompt=augment_prompt,
                        direct_pos=direct_pos,
                        policy_seed=policy_seed,
                    )

                    if result["success"]:
                        episode_successes += 1
                        total_successes += 1
                    total_rollouts += 1

                    video_paths = {}
                    if result.get("frames"):
                        env_mp4_path = output_path / f"episode_{episode_idx:03d}_rollout_{rollout_idx:02d}_env.mp4"
                        save_video(result["frames"], str(env_mp4_path))
                        video_paths["env_mp4"] = str(env_mp4_path)
                    if result.get("server_frames"):
                        model_mp4_path = output_path / f"episode_{episode_idx:03d}_rollout_{rollout_idx:02d}_model.mp4"
                        save_video(result["server_frames"], str(model_mp4_path))
                        video_paths["model_mp4"] = str(model_mp4_path)
                    if save_gifs and result.get("frames"):
                        env_gif_path = output_path / f"episode_{episode_idx:03d}_rollout_{rollout_idx:02d}_env.gif"
                        save_gif(result["frames"], str(env_gif_path))
                        video_paths["env_gif"] = str(env_gif_path)
                    if save_gifs and result.get("server_frames"):
                        model_gif_path = output_path / f"episode_{episode_idx:03d}_rollout_{rollout_idx:02d}_model.gif"
                        save_gif(result["server_frames"], str(model_gif_path))
                        video_paths["model_gif"] = str(model_gif_path)

                    # Save combined video (env | model) side by side
                    if result.get("frames") and result.get("server_frames"):
                        combined_frames = concatenate_videos_horizontal(
                            result["frames"], result["server_frames"]
                        )
                        video_path = output_path / f"episode_{episode_idx:03d}_rollout_{rollout_idx:02d}_combined.mp4"
                        save_video(combined_frames, str(video_path))
                        video_paths["combined_mp4"] = str(video_path)
                        if save_gifs:
                            combined_gif_path = output_path / f"episode_{episode_idx:03d}_rollout_{rollout_idx:02d}_combined.gif"
                            save_gif(combined_frames, str(combined_gif_path))
                            video_paths["combined_gif"] = str(combined_gif_path)

                    episode_rollout_results.append({
                        "rollout": rollout_idx,
                        "episode_seed": env_seed,
                        "policy_seed": policy_seed,
                        "success": result["success"],
                        "steps": result["steps"],
                        "max_coverage": result["max_coverage"],
                        "max_coverage_step": result["max_coverage_step"],
                        "final_coverage": result["final_coverage"],
                        "coverage_history": result["coverage_history"],
                        "videos": video_paths,
                    })

                    status = "SUCCESS" if result["success"] else "FAIL"
                    print(
                        f"Episode {global_episode_idx} ({episode_idx + 1}/{num_episodes}), "
                        f"Rollout {rollout_idx + 1}/{num_rollouts}: {status} | "
                        f"Steps: {result['steps']} | "
                        f"Peak Coverage: {result['max_coverage']:.2%} @ step {result['max_coverage_step']} | "
                        f"Final Coverage: {result['final_coverage']:.2%}"
                    )

                except Exception as e:
                    print(f"Episode {episode_idx + 1}, Rollout {rollout_idx + 1}: ERROR - {e}")
                    import traceback
                    traceback.print_exc()

            # Aggregate rollout results for this episode
            if episode_rollout_results:
                coverages = [r["max_coverage"] for r in episode_rollout_results]
                final_coverages = [r["final_coverage"] for r in episode_rollout_results]
                steps_list = [r["steps"] for r in episode_rollout_results]
                episode_success_rate = episode_successes / len(episode_rollout_results)
                
                results.append({
                    "episode": global_episode_idx,
                    "local_episode": episode_idx,
                    "episode_seed": seed + global_episode_idx if seed is not None else None,
                    "num_rollouts": len(episode_rollout_results),
                    "coverage_metric": "max_coverage",
                    "success_rate": episode_success_rate,
                    "successes": episode_successes,
                    "mean_coverage": float(np.mean(coverages)),
                    "std_coverage": float(np.std(coverages)),
                    "min_coverage": float(np.min(coverages)),
                    "max_coverage": float(np.max(coverages)),
                    "mean_final_coverage": float(np.mean(final_coverages)),
                    "std_final_coverage": float(np.std(final_coverages)),
                    "min_final_coverage": float(np.min(final_coverages)),
                    "max_final_coverage": float(np.max(final_coverages)),
                    "mean_steps": float(np.mean(steps_list)),
                    "std_steps": float(np.std(steps_list)),
                    "rollouts": episode_rollout_results,
                })

                print(
                    f"  >> Episode {episode_idx + 1} Summary: "
                    f"Success Rate: {episode_success_rate:.1%} | "
                    f"Peak Coverage: {np.mean(coverages):.2%} ± {np.std(coverages):.2%} | "
                    f"Final Coverage: {np.mean(final_coverages):.2%} ± {np.std(final_coverages):.2%}"
                )

    finally:
        client.close()

    # Summary
    overall_success_rate = total_successes / total_rollouts if total_rollouts > 0 else 0
    all_coverages = [r["mean_coverage"] for r in results] if results else []
    all_final_coverages = [r["mean_final_coverage"] for r in results] if results else []
    avg_coverage = np.mean(all_coverages) if all_coverages else 0
    std_coverage = np.std(all_coverages) if all_coverages else 0
    avg_final_coverage = np.mean(all_final_coverages) if all_final_coverages else 0
    std_final_coverage = np.std(all_final_coverages) if all_final_coverages else 0

    # Save results to JSON
    summary = {
        "domain_name": "pusht",
        "image_size": 96,
        "action_space": action_space,
        "direct_pos": direct_pos,
        "save_gifs": save_gifs,
        "coverage_metric": "max_coverage",
        "episode_start": episode_start,
        "num_episodes": num_episodes,
        "num_rollouts_per_episode": num_rollouts,
        "total_rollouts": total_rollouts,
        "overall_success_rate": overall_success_rate,
        "total_successes": total_successes,
        "avg_coverage": float(avg_coverage),
        "std_coverage": float(std_coverage),
        "avg_final_coverage": float(avg_final_coverage),
        "std_final_coverage": float(std_final_coverage),
        "seed": seed,
        "server_url": server_url,
        "max_steps": max_steps,
        "checkpoint": checkpoint_info,
        "episodes": results,
    }
    json_path = output_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 50)
    print(f"Overall Success Rate: {overall_success_rate:.2%} ({total_successes}/{total_rollouts})")
    print(f"Average Peak Coverage: {avg_coverage:.2%} ± {std_coverage:.2%}")
    print(f"Average Final Coverage: {avg_final_coverage:.2%} ± {std_final_coverage:.2%}")
    print(f"Results saved to: {output_path}")
    print(f"JSON results: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Eval with original diffusion_policy env")
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-rollouts", type=int, default=10, help="Number of rollouts per episode for robustness")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--episode-start", type=int, default=0,
                        help="Global episode index for sharded evaluations")
    parser.add_argument("--replay-dataset", action="store_true")
    parser.add_argument("--dataset-start-episode", type=int, default=0)
    parser.add_argument("--dataset-repo-id", type=str, default="lerobot/pusht_image")
    parser.add_argument("--debug", action="store_true", help="Debug mode: run 1 action chunk only")
    parser.add_argument("--seed", type=int, default=9999, help="Random seed for reproducibility")
    parser.add_argument("--augment-prompt", action="store_true", help="Use augmented prompt with detailed task description")
    parser.add_argument("--action-space", type=str, default="relative", choices=["relative", "absolute"],
                        help="Action space: 'relative' (delta from agent pos) or 'absolute' (target position)")
    parser.add_argument("--direct-pos", action="store_true",
                        help="Bypass PID controller: set agent position directly. Use with --action-space absolute")
    parser.add_argument("--wait-timeout", type=float, default=600.0,
                        help="Seconds to wait for the model server health endpoint before rollouts")
    parser.add_argument("--save-gifs", action="store_true", help="Also save GIF copies of rollout videos")

    args = parser.parse_args()

    evaluate(
        server_url=args.server_url,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        episode_start=args.episode_start,
        replay_dataset=args.replay_dataset,
        dataset_start_episode=args.dataset_start_episode,
        dataset_repo_id=args.dataset_repo_id,
        debug_mode=args.debug,
        seed=args.seed,
        augment_prompt=args.augment_prompt,
        num_rollouts=args.num_rollouts,
        action_space=args.action_space,
        direct_pos=args.direct_pos,
        wait_timeout=args.wait_timeout,
        save_gifs=args.save_gifs,
    )


if __name__ == "__main__":
    main()
