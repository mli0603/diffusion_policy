"""Mock inference server for testing the PushT evaluation client.

Supports two modes:
1. Pattern mode: Returns fixed pattern actions (right, left, circle, etc.)
2. Dataset mode: Replays actions from lerobot/pusht_image dataset

Run with:
    Pattern mode: python -m eval_pusht.mock_server --pattern right
    Dataset mode: python -m eval_pusht.mock_server --dataset lerobot/pusht_image
"""

import argparse
import base64
import io
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
from typing import Optional

from PIL import Image


class DatasetLoader:
    """Loads and serves actions from a LeRobot dataset."""

    def __init__(self, repo_id: str, chunk_size: int = 16):
        """Initialize dataset loader.

        Args:
            repo_id: HuggingFace dataset repo ID (e.g., 'lerobot/pusht_image')
            chunk_size: Number of actions to return per request
        """
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise ImportError(
                "lerobot package required for dataset mode. "
                "Install with: pip install lerobot"
            )

        print(f"[DatasetLoader] Loading dataset: {repo_id}")
        self.dataset = LeRobotDataset(repo_id=repo_id)
        self.chunk_size = chunk_size

        # Build episode index mapping
        self._build_episode_index()
        print(f"[DatasetLoader] Loaded {self.num_episodes} episodes")

    def _build_episode_index(self):
        """Build mapping from episode_idx to dataset indices."""
        self.episode_indices = {}

        # LeRobotDataset stores episode info in episode_data_index
        if hasattr(self.dataset, "episode_data_index"):
            # Use episode_data_index which contains 'from' and 'to' for each episode
            ep_data_idx = self.dataset.episode_data_index
            for ep_idx in range(len(ep_data_idx["from"])):
                start = ep_data_idx["from"][ep_idx].item()
                end = ep_data_idx["to"][ep_idx].item()
                self.episode_indices[ep_idx] = list(range(start, end))
            print(f"[DatasetLoader] Built index from episode_data_index")
        else:
            # Fallback: iterate through dataset
            print(f"[DatasetLoader] Building index by iterating dataset...")
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                ep_idx = sample.get("episode_index", 0)
                if hasattr(ep_idx, "item"):
                    ep_idx = ep_idx.item()
                if ep_idx not in self.episode_indices:
                    self.episode_indices[ep_idx] = []
                self.episode_indices[ep_idx].append(idx)

        self.num_episodes = len(self.episode_indices)
        print(f"[DatasetLoader] Episode indices: {list(self.episode_indices.keys())[:10]}...")

    def get_episode_length(self, episode_idx: int) -> int:
        """Get number of frames in an episode."""
        if episode_idx not in self.episode_indices:
            raise ValueError(f"Episode {episode_idx} not found")
        return len(self.episode_indices[episode_idx])

    def get_action_chunk(
        self,
        episode_idx: int,
        frame_idx: int,
        as_delta: bool = True,
        action_norm_range: float = 512.0,
    ) -> list:
        """Get action chunk starting from a frame.

        Args:
            episode_idx: Episode index in dataset
            frame_idx: Starting frame index within episode
            as_delta: If True, convert absolute actions to normalized deltas
            action_norm_range: Normalization range for delta conversion

        Returns:
            List of actions [[x, y], ...] with length up to chunk_size
        """
        if episode_idx not in self.episode_indices:
            raise ValueError(f"Episode {episode_idx} not found")

        episode_dataset_indices = self.episode_indices[episode_idx]
        episode_length = len(episode_dataset_indices)

        # Get the first frame's agent position for delta computation
        # All deltas in the chunk are relative to this starting position
        first_agent_pos = None
        if as_delta and frame_idx < episode_length:
            first_dataset_idx = episode_dataset_indices[frame_idx]
            first_sample = self.dataset[first_dataset_idx]
            first_state = first_sample.get("observation.state", None)
            if first_state is not None:
                if hasattr(first_state, "tolist"):
                    first_state = first_state.tolist()
                elif hasattr(first_state, "numpy"):
                    first_state = first_state.numpy().tolist()
                first_agent_pos = first_state[:2]

        actions = []
        for i in range(self.chunk_size):
            current_frame = frame_idx + i
            if current_frame >= episode_length:
                # Pad with last action if we run out
                if actions:
                    actions.append(actions[-1])
                else:
                    actions.append([0.0, 0.0])
            else:
                dataset_idx = episode_dataset_indices[current_frame]
                sample = self.dataset[dataset_idx]
                action = sample["action"]

                # Convert to list
                if hasattr(action, "tolist"):
                    action = action.tolist()
                elif hasattr(action, "numpy"):
                    action = action.numpy().tolist()

                # Take only first 2 dimensions (x, y)
                action = action[:2]

                if as_delta and first_agent_pos is not None:
                    # Compute delta relative to the FIRST frame's agent position
                    # This ensures the client can reconstruct absolute actions correctly
                    delta = [
                        (action[0] - first_agent_pos[0]) / action_norm_range,
                        (action[1] - first_agent_pos[1]) / action_norm_range,
                    ]
                    action = delta

                actions.append(action)

        return actions

    def get_initial_state(self, episode_idx: int) -> dict:
        """Get initial state for an episode.

        Returns:
            Dictionary with initial agent_pos and block_pos
        """
        if episode_idx not in self.episode_indices:
            raise ValueError(f"Episode {episode_idx} not found")

        first_idx = self.episode_indices[episode_idx][0]
        sample = self.dataset[first_idx]

        state = sample.get("observation.state", None)
        if state is not None:
            if hasattr(state, "tolist"):
                state = state.tolist()
            elif hasattr(state, "numpy"):
                state = state.numpy().tolist()

            return {
                "agent_pos": state[:2],
                "block_pos": state[2:4] if len(state) >= 4 else None,
                "block_angle": state[4] if len(state) >= 5 else None,
            }
        return {}

    def get_image_chunk(self, episode_idx: int, frame_idx: int) -> list:
        """Get image chunk starting from a frame.

        Args:
            episode_idx: Episode index in dataset
            frame_idx: Starting frame index within episode

        Returns:
            List of base64-encoded PNG images
        """
        import base64
        import io
        import numpy as np

        if episode_idx not in self.episode_indices:
            raise ValueError(f"Episode {episode_idx} not found")

        episode_dataset_indices = self.episode_indices[episode_idx]
        episode_length = len(episode_dataset_indices)

        images = []
        for i in range(self.chunk_size):
            current_frame = frame_idx + i
            if current_frame >= episode_length:
                # Pad with last image if we run out
                if images:
                    images.append(images[-1])
                continue

            dataset_idx = episode_dataset_indices[current_frame]
            sample = self.dataset[dataset_idx]

            img = sample.get("observation.image", None)
            if img is not None:
                # Convert to numpy array
                if hasattr(img, "numpy"):
                    img = img.numpy()

                # Handle channel-first format (C, H, W) -> (H, W, C)
                if img.ndim == 3 and img.shape[0] in [1, 3]:
                    img = np.transpose(img, (1, 2, 0))

                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)

                # Encode as base64 PNG
                pil_img = Image.fromarray(img)
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                buffer.seek(0)
                encoded = base64.b64encode(buffer.read()).decode("utf-8")
                images.append(encoded)

        return images


class MockInferenceHandler(BaseHTTPRequestHandler):
    """HTTP handler that returns actions from pattern or dataset."""

    # Class-level config
    action_pattern = "right"
    chunk_size = 16
    save_images = False
    image_dir = "mock_server_images"
    request_count = 0
    run_name = "mock_model"

    # Dataset mode
    dataset_loader: Optional[DatasetLoader] = None
    current_episode = 0
    current_frame = 0
    action_format = "delta"  # "delta" or "absolute"

    def do_GET(self):
        if self.path == "/info":
            self._handle_info()
        elif self.path.startswith("/episode/"):
            self._handle_episode_info()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/predict":
            self._handle_predict()
        elif self.path == "/reset":
            self._handle_reset()
        else:
            self.send_error(404, "Not Found")

    def _handle_info(self):
        """Return server metadata."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        info = {
            "run_name": self.run_name,
            "checkpoint": "mock_server",
            "chunk_size": self.chunk_size,
            "mode": "dataset" if self.dataset_loader else "pattern",
        }

        if self.dataset_loader:
            info["num_episodes"] = self.dataset_loader.num_episodes
            info["current_episode"] = self.current_episode
            info["current_frame"] = self.current_frame
        else:
            info["action_pattern"] = self.action_pattern

        self.wfile.write(json.dumps(info).encode())

    def _handle_episode_info(self):
        """Return episode-specific info including initial state."""
        try:
            episode_idx = int(self.path.split("/")[-1])
        except ValueError:
            self.send_error(400, "Invalid episode index")
            return

        if not self.dataset_loader:
            self.send_error(400, "Dataset mode not enabled")
            return

        try:
            initial_state = self.dataset_loader.get_initial_state(episode_idx)
            episode_length = self.dataset_loader.get_episode_length(episode_idx)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            response = {
                "episode_idx": episode_idx,
                "episode_length": episode_length,
                "initial_state": initial_state,
            }
            self.wfile.write(json.dumps(response).encode())
        except ValueError as e:
            self.send_error(404, str(e))

    def _handle_reset(self):
        """Reset to a specific episode."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            data = {}

        episode_idx = data.get("episode_idx", 0)

        MockInferenceHandler.current_episode = episode_idx
        MockInferenceHandler.current_frame = 0
        MockInferenceHandler.request_count = 0

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        response = {
            "status": "ok",
            "episode_idx": episode_idx,
            "frame_idx": 0,
        }

        if self.dataset_loader:
            try:
                response["initial_state"] = self.dataset_loader.get_initial_state(episode_idx)
                response["episode_length"] = self.dataset_loader.get_episode_length(episode_idx)
            except ValueError as e:
                response["error"] = str(e)

        self.wfile.write(json.dumps(response).encode())

    def _handle_predict(self):
        """Handle prediction request."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
            if self.save_images and "image" in data:
                self._save_image(data["image"])
        except Exception as e:
            print(f"[MockServer] Warning: Could not parse request: {e}")

        # Generate actions based on mode
        if self.dataset_loader:
            actions, video = self._get_dataset_actions()
        else:
            actions = self._generate_pattern_actions()
            video = []

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {"action": actions, "video": video}
        self.wfile.write(json.dumps(response).encode())

    def _get_dataset_actions(self) -> tuple:
        """Get actions and video frames from dataset for current episode/frame.
        
        Returns:
            Tuple of (actions, video) where video is a list of base64-encoded PNGs
        """
        try:
            actions = self.dataset_loader.get_action_chunk(
                episode_idx=self.current_episode,
                frame_idx=self.current_frame,
                as_delta=(self.action_format == "delta"),
            )
            
            video = self.dataset_loader.get_image_chunk(
                episode_idx=self.current_episode,
                frame_idx=self.current_frame,
            )

            # Advance frame pointer
            MockInferenceHandler.current_frame += self.chunk_size
            MockInferenceHandler.request_count += 1

            return actions, video
        except Exception as e:
            print(f"[MockServer] Error getting dataset actions: {e}")
            return [[0.0, 0.0]] * self.chunk_size, []

    def _save_image(self, base64_image: str):
        """Decode and save the received image."""
        count = MockInferenceHandler.request_count

        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))

            os.makedirs(self.image_dir, exist_ok=True)
            image_path = Path(self.image_dir) / f"request_{count:04d}.png"
            image.save(image_path)
            print(f"[MockServer] Saved image: {image_path} (size: {image.size})")
        except Exception as e:
            print(f"[MockServer] Failed to save image: {e}")

    def _generate_pattern_actions(self):
        """Generate fixed pattern normalized delta actions (T, 2)."""
        T = self.chunk_size
        pattern = self.action_pattern

        if pattern == "right":
            actions = [[0.1, 0.0]] * T
        elif pattern == "left":
            actions = [[-0.1, 0.0]] * T
        elif pattern == "up":
            actions = [[0.0, -0.1]] * T
        elif pattern == "down":
            actions = [[0.0, 0.1]] * T
        elif pattern == "circle":
            import math
            actions = []
            for i in range(T):
                angle = 2 * math.pi * i / T
                actions.append([0.1 * math.cos(angle), 0.1 * math.sin(angle)])
        elif pattern == "toward_goal":
            actions = [[0.05, -0.05]] * T
        else:
            actions = [[0.0, 0.0]] * T

        MockInferenceHandler.request_count += 1
        return actions

    def log_message(self, format, *args):
        print(f"[MockServer] {args[0]}")


def run_server(
    host: str = "localhost",
    port: int = 8000,
    pattern: str = "right",
    chunk_size: int = 16,
    save_images: bool = False,
    image_dir: str = "mock_server_images",
    run_name: str = "mock_model",
    dataset: Optional[str] = None,
    action_format: str = "delta",
):
    """Run the mock inference server.

    Args:
        host: Server host
        port: Server port
        pattern: Action pattern (right, left, up, down, circle, toward_goal)
        chunk_size: Number of actions per chunk (T)
        save_images: Whether to save received images to disk
        image_dir: Directory to save images
        run_name: Model/run identifier returned by /info endpoint
        dataset: HuggingFace dataset repo ID for dataset replay mode
        action_format: "delta" for normalized deltas, "absolute" for raw positions
    """
    MockInferenceHandler.action_pattern = pattern
    MockInferenceHandler.chunk_size = chunk_size
    MockInferenceHandler.save_images = save_images
    MockInferenceHandler.image_dir = image_dir
    MockInferenceHandler.request_count = 0
    MockInferenceHandler.run_name = run_name
    MockInferenceHandler.current_episode = 0
    MockInferenceHandler.current_frame = 0
    MockInferenceHandler.action_format = action_format

    # Load dataset if specified
    if dataset:
        MockInferenceHandler.dataset_loader = DatasetLoader(dataset, chunk_size)
        MockInferenceHandler.run_name = f"dataset_replay_{dataset.replace('/', '_')}"
    else:
        MockInferenceHandler.dataset_loader = None

    server_address = (host, port)
    httpd = HTTPServer(server_address, MockInferenceHandler)

    print(f"Mock server running at http://{host}:{port}")
    print(f"Endpoints:")
    print(f"  POST /predict - get actions")
    print(f"  POST /reset   - reset to episode (body: {{'episode_idx': N}})")
    print(f"  GET  /info    - get server info")
    print(f"  GET  /episode/N - get episode info and initial state")
    print(f"Run name: {MockInferenceHandler.run_name}")

    if dataset:
        print(f"Mode: Dataset replay")
        print(f"Dataset: {dataset}")
        print(f"Episodes: {MockInferenceHandler.dataset_loader.num_episodes}")
        print(f"Action format: {action_format}")
    else:
        print(f"Mode: Pattern")
        print(f"Action pattern: {pattern}")

    print(f"Chunk size: {chunk_size}")
    if save_images:
        print(f"Saving images to: {image_dir}/")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Mock inference server for PushT evaluation")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--pattern",
        type=str,
        default="right",
        choices=["right", "left", "up", "down", "circle", "toward_goal", "none"],
        help="Action pattern to return (ignored in dataset mode)",
    )
    parser.add_argument("--chunk-size", type=int, default=16, help="Actions per chunk (T)")
    parser.add_argument("--save-images", action="store_true", help="Save received images to disk")
    parser.add_argument("--image-dir", type=str, default="mock_server_images", help="Directory to save images")
    parser.add_argument("--run-name", type=str, default="mock_model", help="Model/run identifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID for replay mode (e.g., lerobot/pusht_image)",
    )
    parser.add_argument(
        "--action-format",
        type=str,
        default="delta",
        choices=["delta", "absolute"],
        help="Action format: 'delta' for normalized deltas, 'absolute' for raw positions",
    )

    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        pattern=args.pattern,
        chunk_size=args.chunk_size,
        save_images=args.save_images,
        image_dir=args.image_dir,
        run_name=args.run_name,
        dataset=args.dataset,
        action_format=args.action_format,
    )


if __name__ == "__main__":
    main()
