# training/data_collection.py
"""
Roll out random KAZ episodes and save (image, zombie_boxes) pairs for
detector training.

Why parallel_env: it advances one full game step per env.step() call, so each
iteration of the loop produces exactly one distinct frame — no duplicate frames
from the two-archer AEC turn cycle.

Output layout:
  <save_dir>/
    images/0000001.jpg   ...
    annotations.json     [{file, boxes:[[l,t,w,h],...], episode, step}, ...]

Boxes are in the pixel coordinate system of the 720×1280 render image,
in [left, top, width, height] format — the same format required by
submission.py CustomZombieDetectorFunction.

Usage (on Kaggle):
  python training/data_collection.py --episodes 200 --save-dir data/raw
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from pettingzoo.butterfly import knights_archers_zombies_v10

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _zombie_boxes(env):
    """Return [[left, top, width, height], ...] in render-pixel coordinates."""
    boxes = []
    for z in env.unwrapped.zombie_list:
        r = z.rect
        boxes.append([int(r.x), int(r.y), int(r.width), int(r.height)])
    return boxes


# ---------------------------------------------------------------------------
# main collection loop
# ---------------------------------------------------------------------------

def collect_dataset(
    n_episodes: int,
    max_cycles: int,
    save_dir: str,
    seed: int,
    skip_empty: bool,
    spawn_rate: int,
    max_zombies: int,
) -> None:
    out = Path(save_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # render_mode="rgb_array" gives env.render() → (H, W, 3) uint8
    # vector_state=False keeps the pixel-based obs space; we read the render
    # directly for images and zombie_list for labels, so vector_state doesn't
    # affect what we save.
    env = knights_archers_zombies_v10.parallel_env(
        num_archers=2,
        num_knights=0,
        max_zombies=max_zombies,
        vector_state=False,
        render_mode="rgb_array",
        max_cycles=max_cycles,
        spawn_rate=spawn_rate,
    )

    annotations = []
    frame_id = 0

    for ep in tqdm(range(n_episodes), desc="Collecting episodes"):
        env.reset(seed=seed + ep)
        step = 0

        while env.agents:
            img = env.render()          # (720, 1280, 3) uint8, HWC
            boxes = _zombie_boxes(env)

            if (not skip_empty) or boxes:
                fname = f"{frame_id:07d}.jpg"
                Image.fromarray(img).save(img_dir / fname, quality=90)
                annotations.append({
                    "file": fname,
                    "boxes": boxes,
                    "episode": ep,
                    "step": step,
                })
                frame_id += 1

            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
            step += 1

    env.close()

    ann_path = out / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(annotations, f)

    n_zombie = sum(1 for a in annotations if a["boxes"])
    total_boxes = sum(len(a["boxes"]) for a in annotations)
    print(f"\nSaved {frame_id} frames → {out}")
    print(f"  Frames with ≥1 zombie : {n_zombie} / {frame_id}")
    print(f"  Total zombie instances : {total_boxes}")
    print(f"  Annotations            : {ann_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Collect KAZ zombie detection dataset")
    p.add_argument("--episodes",   type=int,   default=200,
                   help="number of episodes to roll out (default 200 ≈ 30k frames)")
    p.add_argument("--max-cycles", type=int,   default=500,
                   help="max game steps per episode")
    p.add_argument("--save-dir",   default="../data/raw",
                   help="output directory (created if absent)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--skip-empty", action="store_true",
                   help="skip frames with no zombies (saves ~30%% disk)")
    p.add_argument("--spawn-rate", type=int,   default=20,
                   help="steps between zombie spawns (lower = more zombies)")
    p.add_argument("--max-zombies",type=int,   default=4)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    collect_dataset(
        n_episodes=args.episodes,
        max_cycles=args.max_cycles,
        save_dir=args.save_dir,
        seed=args.seed,
        skip_empty=args.skip_empty,
        spawn_rate=args.spawn_rate,
        max_zombies=args.max_zombies,
    )
