import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(PROJECT_DIR, "logs", "rsl_rl", "franka_pick_place")

STAGES = ["reach", "grasp", "lift", "transport_coarse", "transport_fine", "place", "success_bonus"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    args = parser.parse_args()
    run = args.run or max(glob.glob(os.path.join(ROOT, "*")), key=os.path.getmtime)

    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]

    def series(tag):
        ev = ea.Scalars(tag)
        return [e.step for e in ev], [e.value for e in ev]

    def find(suffix):
        hits = [t for t in tags if t.endswith(suffix)]
        return hits[0] if hits else None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    for tag, label in [(find("mean_reward"), "mean reward")]:
        if tag:
            ax.plot(*series(tag), label=label)
    ax.set_title("Mean episode reward")

    ax = axes[0, 1]
    for name in STAGES:
        tag = find(f"Episode_Reward/{name}")
        if tag:
            ax.plot(*series(tag), label=name)
    ax.set_title("Reward terms (per episode)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name in ["success", "time_out", "object_dropped"]:
        tag = find(f"Episode_Termination/{name}")
        if tag:
            ax.plot(*series(tag), label=name)
    ax.set_title("Termination fraction (success = task done)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    tag = find("mean_episode_length")
    if tag:
        ax.plot(*series(tag))
    ax.set_title("Mean episode length (steps; timeout = 400 old task, 750 for 360 task)")

    for a in axes.flat:
        a.set_xlabel("iteration")
        a.grid(alpha=0.3)
    fig.suptitle(f"Franka pick & place PPO - {os.path.basename(run)}")
    fig.tight_layout()
    out = os.path.join(run, "training_curves.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    for name in ["success", "time_out", "object_dropped"]:
        tag = find(f"Episode_Termination/{name}")
        if tag:
            _, v = series(tag)
            print(f"final {name:15s}: {v[-1]:.3f}   (best {max(v):.3f})")


if __name__ == "__main__":
    main()
