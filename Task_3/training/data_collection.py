# training/data_collection.py
"""
Roll out random episodes; at every frame save (image, list[bbox]).
Ground-truth boxes come from the env's internal sprite rectangles.
"""
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10


def collect_dataset(n_episodes=200, distortion_level=0, save_dir="data/"):
    env = knights_archers_zombies_v10.parallel_env(
        vector_state=False, render_mode=None, max_cycles=500,
    )
    samples = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = {a: False for a in env.agents}
        while not all(done.values()):
            img = env.unwrapped.render()         # H×W×3 uint8
            zombie_boxes = extract_zombie_boxes(env)  # parse env.unwrapped.zombie_list
            samples.append({"image": img, "boxes": zombie_boxes})
            actions = {a: env.action_space(a).sample() for a in env.agents
                       if not done[a]}
            obs, rew, term, trunc, _ = env.step(actions)
            done = {a: term[a] or trunc[a] for a in env.agents}
    save(samples, save_dir)


def extract_zombie_boxes(env):
    """Read pygame.sprite rects of zombies from the unwrapped env."""
    boxes = []
    for z in env.unwrapped.zombie_list:
        r = z.rect
        boxes.append([r.x, r.y, r.x + r.width, r.y + r.height])
    return np.array(boxes, dtype=np.float32)