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
      "action": [[a0_0, a0_1, ...], ..., [aN_0, aN_1, ...]],  # (N, D) normalized deltas in [-1, 1]
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


class SimpleInferenceClient:
    """Simple inference client for the mock server."""

    def __init__(self, server_url: str, action_norm_range: float = 512.0, timeout: float = 300.0):
        self.server_url = server_url
        self.base_url = server_url.rsplit("/", 1)[0]
        self.action_norm_range = action_norm_range
        self.timeout = timeout
        self.session = requests.Session()

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

    def get_action_chunk(self, image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Get action chunk from server.
        
        Returns:
            Tuple of (actions, video_frames) where video_frames is a list of decoded images
        """
        import base64
        import io
        from PIL import Image

        # Encode image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        # Send request
        response = self.session.post(
            self.server_url,
            json={
                "image": encoded,
                "prompt": "PushT task",
                "domain_name": "pusht",
                "image_size": 96,
            },
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

    def convert_to_absolute(self, delta: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
        """Convert delta to absolute position."""
        absolute = delta * self.action_norm_range + agent_pos
        return np.clip(absolute, 0, self.action_norm_range)

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
) -> dict:
    """Run a single episode."""
    
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

    frames = []
    obs_frames = []
    dataset_frames = []
    server_frames = []  # Video frames from server response
    step = 0
    total_reward = 0.0
    success = False
    request_count = 0
    frame_idx = 0

    while step < max_steps:
        agent_pos = obs[:2]

        # Render observation (96x96)
        obs_img = env.render("rgb_array")
        obs_frames.append(obs_img.copy())

        request_count += 1

        # Get action chunk from server
        delta_actions, video_frames = client.get_action_chunk(obs_img)
        # Skip the first frame (input image) as it duplicates the observation
        chunk_server_frames = video_frames[1:] if video_frames else []
        print(f"[Step {step}] Received {len(delta_actions)} actions from server (request #{request_count})")
        
        # Store agent position at chunk start - this is the reference for all deltas
        chunk_start_agent_pos = agent_pos.copy()
        

        # Execute all actions in chunk
        for i, delta in enumerate(delta_actions):
            
            # Convert delta to absolute using chunk_start_agent_pos (the reference position)
            action = client.convert_to_absolute(delta, chunk_start_agent_pos)
            

            # Step environment
            obs, reward, done, info = env.step(action)
            agent_pos = obs[:2]
            total_reward += reward


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
            print(f"[DEBUG] Executed {len(delta_actions)} actions from 1 request, exiting.")
            break

        if done or step >= max_steps:
            break

    # Calculate final coverage
    from shapely.geometry import MultiPolygon, Polygon
    import pymunk

    def get_block_geom(body, shapes):
        geoms = []
        for shape in shapes:
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(Polygon(verts))
        return MultiPolygon(geoms)

    goal_body = env._get_goal_pose_body(env.goal_pose)
    goal_geom = get_block_geom(goal_body, env.block.shapes)
    block_geom = get_block_geom(env.block, env.block.shapes)
    coverage = goal_geom.intersection(block_geom).area / goal_geom.area

    return {
        "frames": frames,
        "obs_frames": obs_frames,
        "dataset_frames": dataset_frames,
        "server_frames": server_frames,
        "total_reward": total_reward,
        "success": success,
        "steps": step,
        "final_coverage": coverage,
        "num_requests": request_count,
    }


def evaluate(
    server_url: str,
    num_episodes: int = 1,
    max_steps: int = 300,
    output_dir: str = "eval_results",
    replay_dataset: bool = False,
    dataset_start_episode: int = 0,
    dataset_repo_id: str = "lerobot/pusht_image",
    debug_mode: bool = False,
    seed: Optional[int] = None,
):
    """Run evaluation."""
    # Create original diffusion_policy environment
    env = PushTEnv(legacy=True, render_size=96)

    # Create client
    client = SimpleInferenceClient(server_url=server_url)

    # Load dataset
    dataset_helper = None
    if replay_dataset:
        dataset_helper = DatasetReplayHelper(repo_id=dataset_repo_id)

    # Get run name and checkpoint info from /info
    server_info = client.get_server_info()
    checkpoint_info = server_info.get("checkpoint", "")
    run_name = server_info.get("run_name", "diffusion_policy_eval")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{run_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting evaluation: {num_episodes} episodes")
    print(f"Server URL: {server_url}")
    print(f"Output directory: {output_path}")
    print(f"Using original diffusion_policy environment")
    print(f"pymunk version: {__import__('pymunk').version}")
    print("-" * 50)

    results = []
    successes = 0

    try:
        for episode_idx in range(num_episodes):
            # Set different seed for each episode for variety while maintaining reproducibility
            if seed is not None:
                episode_seed = seed + episode_idx
                set_seed(episode_seed)
                env.seed(episode_seed)  # Seed environment's internal RNG

            dataset_episode_idx = None
            if replay_dataset:
                dataset_episode_idx = dataset_start_episode + episode_idx

            try:
                result = run_episode(
                    env=env,
                    client=client,
                    max_steps=max_steps,
                    dataset_episode_idx=dataset_episode_idx,
                    dataset_helper=dataset_helper,
                    debug_mode=debug_mode,
                )

                if result["success"]:
                    successes += 1

                # Save combined video (env | model) side by side
                if result.get("frames") and result.get("server_frames"):
                    combined_frames = concatenate_videos_horizontal(
                        result["frames"], result["server_frames"]
                    )
                    video_path = output_path / f"episode_{episode_idx:03d}_combined.mp4"
                    save_video(combined_frames, str(video_path))

                results.append({
                    "episode": episode_idx,
                    "success": result["success"],
                    "steps": result["steps"],
                    "final_coverage": result["final_coverage"],
                })

                status = "SUCCESS" if result["success"] else "FAIL"
                print(
                    f"Episode {episode_idx + 1}/{num_episodes}: {status} | "
                    f"Steps: {result['steps']} | Coverage: {result['final_coverage']:.2%}"
                )

            except Exception as e:
                print(f"Episode {episode_idx + 1}: ERROR - {e}")
                import traceback
                traceback.print_exc()

    finally:
        client.close()

    # Summary
    success_rate = successes / num_episodes if num_episodes > 0 else 0
    avg_coverage = sum(r["final_coverage"] for r in results) / len(results) if results else 0

    # Save results to JSON
    summary = {
        "domain_name": "pusht",
        "image_size": 96,
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_coverage": avg_coverage,
        "successes": successes,
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
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average coverage: {avg_coverage:.2%}")
    print(f"Results saved to: {output_path}")
    print(f"JSON results: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Eval with original diffusion_policy env")
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--replay-dataset", action="store_true")
    parser.add_argument("--dataset-start-episode", type=int, default=0)
    parser.add_argument("--dataset-repo-id", type=str, default="lerobot/pusht_image")
    parser.add_argument("--debug", action="store_true", help="Debug mode: run 1 action chunk only")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()

    evaluate(
        server_url=args.server_url,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        replay_dataset=args.replay_dataset,
        dataset_start_episode=args.dataset_start_episode,
        dataset_repo_id=args.dataset_repo_id,
        debug_mode=args.debug,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
