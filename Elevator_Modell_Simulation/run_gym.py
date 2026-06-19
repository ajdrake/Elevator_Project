"""
Run the trained PPO model using the Gymnasium environment.

Usage:
    uv run python run_gym.py                    # headless, prints stats
    uv run python run_gym.py --render human     # with Pygame window
    uv run python run_gym.py --model path.zip   # custom model path
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from elevator_env import ElevatorEnv
from Visualization import (
    plot_travel_times_per_hour,
    plot_wait_times_per_hour,
    plot_total_travel_times_per_hour,
    plot_guest_counts_per_hour,
    plot_average_total_time_per_hour,
    append_episode_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Run trained RL elevator model")
    parser.add_argument(
        "--render", choices=["human", "rgb_array"], default=None,
        help="Render mode (omit for headless)",
    )
    parser.add_argument(
        "--model", default="modell/ppo_elevator_episode_1318.zip",
        help="Path to trained MaskablePPO model",
    )
    parser.add_argument(
        "--max-guests", type=int, default=200,
        help="Number of guests to simulate",
    )
    args = parser.parse_args()

    model = MaskablePPO.load(args.model)

    env = ElevatorEnv(
        render_mode=args.render,
        num_elevators=3,
        num_floors=10,
        max_passengers=5,
        max_guests=args.max_guests,
        spawn_intervall=7200,
    )

    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated:
        action_masks = info["action_mask"]

        # Model was trained single-elevator; run it per-elevator
        actions = []
        for i in range(env.num_elevators):
            elev_obs = _get_single_elevator_obs(obs, i, env.num_floors)
            action, _ = model.predict(
                elev_obs, action_masks=action_masks[i], deterministic=True,
            )
            actions.append(int(action[0]))

        obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1

        if env.render_mode == "human":
            env.render()
            # Handle pygame quit
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

    env.close()

    # Print results
    print(f"\nSimulation complete in {step_count} steps.")
    print(f"Guests processed: {env.guests_left_building}/{env.max_guests}")
    print(f"Total reward: {env.total_reward:.1f}")
    print(f"Log entries: {len(env.logs)}")

    if env.logs:
        import pandas as pd
        df = pd.DataFrame(env.logs)
        wait_df = df[df["mode"] == "elevator_waiting"]
        drive_df = df[df["mode"] == "elevator_drive"]
        if not wait_df.empty:
            print(f"Avg wait time: {wait_df['wait_time'].mean():.1f}s")
        if not drive_df.empty:
            print(f"Avg ride time: {drive_df['travel_time'].mean():.1f}s")
            avg_total = (drive_df["wait_time"] + drive_df["travel_time"]).mean()
            print(f"Avg total time: {avg_total:.1f}s")

        ep = 1
        plot_wait_times_per_hour(env.logs, ep)
        plot_travel_times_per_hour(env.logs, ep)
        plot_total_travel_times_per_hour(env.logs, ep)
        plot_guest_counts_per_hour(env.logs, ep)
        plot_average_total_time_per_hour(env.logs, ep)
        append_episode_stats(env.logs, ep)


def _get_single_elevator_obs(full_obs, elevator_idx, num_floors):
    """
    Extract a single-elevator observation from the multi-elevator obs.

    The trained model expects: [floor, n_passengers, dest_hist(10),
                                other_elev_1(12 zeros), other_elev_2(12 zeros),
                                waiting_per_floor(10)]
    """
    elev_size = 2 + num_floors
    start = elevator_idx * elev_size
    elev_obs = list(full_obs[start:start + elev_size])

    # Pad with zeros for "other elevators" (model trained single-elevator)
    for _ in range(2):
        elev_obs.extend([0] * elev_size)

    # Waiting per floor is at the end of the full obs
    waiting_start = len(full_obs) - num_floors
    elev_obs.extend(full_obs[waiting_start:])

    return np.array(elev_obs, dtype=np.int32)


if __name__ == "__main__":
    main()
