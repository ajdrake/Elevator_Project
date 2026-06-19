import sys
import argparse
import simpy
from Building import Building
from Visualization import (
    plot_travel_times_per_hour,
    plot_wait_times_per_hour,
    plot_total_travel_times_per_hour,
    plot_guest_counts_per_hour,
    plot_average_total_time_per_hour,
    append_episode_stats,
)
from sb3_contrib import MaskablePPO


def get_simulation_params(param_defs):
    """
    Opens a popup with input fields for all parameters in param_defs
    and returns a dictionary with the converted values.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox

    results = {}

    def on_submit():
        try:
            for name, (typ, _) in param_defs.items():
                raw = entries[name].get()
                results[name] = typ(raw)
            root.destroy()
        except Exception as e:
            messagebox.showerror("Invalid input", f"Error converting a value:\n{e}")

    root = tk.Tk()
    root.title("Simulation Settings")
    entries = {}

    for row, (name, (typ, default)) in enumerate(param_defs.items()):
        ttk.Label(root, text=f"{name} ({typ.__name__}):").grid(
            row=row, column=0, padx=5, pady=5, sticky="e"
        )
        entry = ttk.Entry(root)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        entries[name] = entry

    btn = ttk.Button(root, text="Start Simulation", command=on_submit)
    btn.grid(row=len(param_defs), column=0, columnspan=2, pady=10)

    root.mainloop()
    return results


DEFAULT_PARAMS = {
    "num_floors": (int, 10),
    "num_elevators": (int, 3),
    "elevator_capacity": (int, 5),
    "door_time": (float, 4),
    "building_width": (int, 600),
    "building_height": (int, 800),
    "shaft_width": (int, 60),
    "shaft_spacing": (int, 10),
    "waiting_area_width": (int, 140),
    "visualize_every": (float, 1.0),
    "max_guests": (int, 200),
    "working_time": (int, 480),
    "screen_width": (int, 800),
    "screen_height": (int, 800),
    "no_floor_zero": (str, "False"),
    "spawn_intervall": (int, 7200),
}


def main():
    parser = argparse.ArgumentParser(description="Elevator RL Model Simulation")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run simulation without GUI (no Pygame/tkinter required)",
    )
    parser.add_argument(
        "--model", default="modell/ppo_elevator_episode_1318.zip",
        help="Path to the trained MaskablePPO model zip file",
    )
    args = parser.parse_args()

    if args.headless:
        params = {name: typ(default) for name, (typ, default) in DEFAULT_PARAMS.items()}
    else:
        params = get_simulation_params(DEFAULT_PARAMS)

    env = simpy.Environment()
    model = MaskablePPO.load(args.model)

    if args.headless:
        building = Building(
            screen=None,
            env=env,
            num_floors=params["num_floors"],
            num_elevators=params["num_elevators"],
            elevator_capacity=params["elevator_capacity"],
            door_time=params["door_time"],
            building_width=params["building_width"],
            building_height=params["building_height"],
            shaft_width=params["shaft_width"],
            shaft_spacing=params["shaft_spacing"],
            waiting_area_width=params["waiting_area_width"],
            visualize_every=params["visualize_every"],
            max_guests=params["max_guests"],
            working_time=params["working_time"],
            no_floor_zero=params["no_floor_zero"],
            spawn_intervall=params["spawn_intervall"],
            modell=model,
            headless=True,
        )
    else:
        import pygame
        pygame.init()
        screen_width, screen_height = params["screen_width"], params["screen_height"]
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Elevator Simulation (RL Model)")

        building = Building(
            screen=screen,
            env=env,
            num_floors=params["num_floors"],
            num_elevators=params["num_elevators"],
            elevator_capacity=params["elevator_capacity"],
            door_time=params["door_time"],
            building_width=params["building_width"],
            building_height=params["building_height"],
            shaft_width=params["shaft_width"],
            shaft_spacing=params["shaft_spacing"],
            waiting_area_width=params["waiting_area_width"],
            visualize_every=params["visualize_every"],
            max_guests=params["max_guests"],
            working_time=params["working_time"],
            no_floor_zero=params["no_floor_zero"],
            spawn_intervall=params["spawn_intervall"],
            modell=model,
        )

    env.process(building.run())
    env.run(until=building.stop_event)

    ep = 11
    plot_wait_times_per_hour(building.logs, ep)
    plot_travel_times_per_hour(building.logs, ep)
    plot_total_travel_times_per_hour(building.logs, ep)
    plot_guest_counts_per_hour(building.logs, ep)
    plot_average_total_time_per_hour(building.logs, ep)
    append_episode_stats(building.logs, ep)

    if not args.headless:
        import pygame
        pygame.quit()

    print(f"\nSimulation complete. {len(building.logs)} log entries recorded.")
    print(f"Guests processed: {building.guests_left_building}/{building.max_guests}")
    print(f"Total reward: {building.total_reward:.1f}")


if __name__ == "__main__":
    main()
