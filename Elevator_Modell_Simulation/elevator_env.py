import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Elevator import Elevator
from Guest import Guest


class ElevatorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        num_elevators=3,
        num_floors=10,
        max_passengers=5,
        max_guests=200,
        spawn_intervall=7200,
        sim_step_size=1,
        ride_time=4,
        door_time=4,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.max_passengers = max_passengers
        self.max_guests = max_guests
        self.spawn_intervall = spawn_intervall
        self.sim_step_size = sim_step_size
        self.ride_time = ride_time
        self.door_time = door_time

        self.action_space = spaces.MultiDiscrete([3] * self.num_elevators)
        self.observation_space = self._build_obs_space()

        self.logs = []
        self.allguests = []
        self.left_guests = []
        self.guests_in_elevator = []
        self.total_reward = 0

        # Pygame state (lazy init)
        self._screen = None
        self._clock = None
        self._font = None
        self._images = None

        self.reset()

    def _build_obs_space(self):
        low_elev = [0, 0] + [0] * self.num_floors
        high_elev = [self.num_floors - 1, self.max_passengers] + [self.max_passengers] * self.num_floors

        low = low_elev * self.num_elevators
        high = high_elev * self.num_elevators

        low += [0] * self.num_floors
        high += [self.max_guests] * self.num_floors

        return spaces.Box(
            low=np.array(low, dtype=np.int32),
            high=np.array(high, dtype=np.int32),
            dtype=np.int32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.elevators = []
        for i in range(self.num_elevators):
            self.elevators.append(Elevator(
                self,
                id=i,
                min_floor=0,
                max_floor=self.num_floors - 1,
                capacity=self.max_passengers,
                door_time=self.door_time,
                ride_time=self.ride_time,
                sim_step_size=self.sim_step_size,
            ))

        self.guests_in_building = 0
        self.guests_left_building = 0
        self.episode_steps = 0
        self.logs = []
        self.waiting_guests = []
        self.guests_on_floors = []
        self._next_guest_id = 0
        self.allguests = []
        self.left_guests = []
        self.guests_in_elevator = []
        self.total_reward = 0

        self.lam = self.max_guests / self.spawn_intervall
        self.mean_inter = 1 / self.lam
        self._time_since_last_spawn = 0.0
        self.time_until_next_arrival = self.np_random.exponential(self.mean_inter)

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def _spawn_guest(self):
        if self._next_guest_id >= self.max_guests:
            return
        possible_targets = list(range(self.num_floors))
        target_floor = self.np_random.choice(possible_targets)

        guest = Guest(
            self,
            guest_id=self._next_guest_id,
            start_floor=0,
            target_floor=target_floor,
            current_floor=0,
            waiting_since=self.episode_steps,
            entered_elevator_step=None,
        )
        self._next_guest_id += 1
        self.guests_in_building += 1
        self.allguests.append(guest)

        if target_floor == 0:
            self.guests_on_floors.append(guest)
            guest.state = "on_floor"
            guest.current_floor = 0
        else:
            self.waiting_guests.append(guest)

    def step(self, actions):
        reward = 0.0
        self.episode_steps += 1

        for guest in list(self.guests_on_floors):
            guest.step(self.sim_step_size, False)

        # Poisson guest spawn
        self._time_since_last_spawn += self.sim_step_size
        while (
            (self.guests_in_building + self.guests_left_building) < self.max_guests
            and self._time_since_last_spawn >= self.time_until_next_arrival
        ):
            self._spawn_guest()
            self._time_since_last_spawn -= self.time_until_next_arrival
            self.time_until_next_arrival = self.np_random.exponential(self.mean_inter)

        # Apply actions
        for i, action in enumerate(actions):
            self.elevators[i].do_action(action)

        # Execute pending actions: dropoff
        for elev in self.elevators:
            if elev.busy_time <= 0 and elev.pending_action is not None:
                elev.execute_pending_action()
                if elev.pending_action[0] == "open":
                    leaving = elev.dropoff_guests()
                    if leaving:
                        for g in leaving:
                            waited_steps = int(
                                (self.episode_steps - g.entered_elevator_step) / 60
                            )
                            self.guests_in_elevator.remove(g)
                            reward += max(1, 20 - waited_steps)

        # Execute pending actions: boarding
        for elev in self.elevators:
            if elev.busy_time <= 0 and elev.pending_action is not None:
                if elev.pending_action[0] == "open":
                    boarded = elev.board_guests(self.waiting_guests)
                    if boarded:
                        for g in boarded:
                            self.waiting_guests.remove(g)
                            self.guests_in_elevator.append(g)
                            waited_steps = int(
                                (self.episode_steps - g.waiting_since) / 60
                            )
                            reward += max(1, 10 - waited_steps)

        # Penalty for waiting
        reward -= 0.1 * len(self.waiting_guests)
        reward -= 0.05 * len(self.guests_in_elevator)
        self.total_reward += reward

        terminated = self.guests_left_building >= self.max_guests
        truncated = self.episode_steps > 45000

        obs = self._get_obs()
        info = {"action_mask": self.get_action_mask()}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        obs = []
        for elev in self.elevators:
            obs.append(elev.current_floor)
            obs.append(len(elev.passengers))
            dest_hist = [0] * self.num_floors
            for p in elev.passengers:
                dest_hist[p.target_floor] += 1
            obs.extend(dest_hist)

        waiting_per_floor = [0] * self.num_floors
        for g in self.waiting_guests:
            waiting_per_floor[g.current_floor] += 1
        obs.extend(waiting_per_floor)

        return np.array(obs, dtype=np.int32)

    def get_action_mask(self):
        masks = []
        for elev in self.elevators:
            mask = np.zeros(3, dtype=bool)
            mask[0] = True

            if elev.busy_time > 0:
                masks.append(np.ones(3, dtype=bool))
                continue

            if elev.busy_time <= 0 and elev.pending_action is not None:
                if elev.pending_action[0] == "close":
                    mask[0] = False
                elev.pending_action = None

            if not elev._guests_waiting_or_leaving():
                mask[0] = False

            if not elev.door_open and elev.current_floor < elev.max_floor:
                mask[1] = True
            if not elev.door_open and elev.current_floor > elev.min_floor:
                mask[2] = True

            masks.append(mask)

        return np.array(masks, dtype=bool)

    def log(self, time, guest_id, mode, wait_time, travel_time):
        self.logs.append({
            "time": time,
            "guest_id": guest_id,
            "mode": mode,
            "wait_time": wait_time,
            "travel_time": travel_time,
        })

    # --- Rendering ---

    def render(self):
        if self.render_mode is None:
            return None
        return self._render_frame()

    def _init_pygame(self):
        import pygame
        pygame.init()
        self._screen_width = 800
        self._screen_height = 800
        self._building_width = 600
        self._building_height = 700
        self._waiting_area_width = 140
        self._shaft_width = 60
        self._shaft_spacing = 10

        if self.render_mode == "human":
            self._screen = pygame.display.set_mode(
                (self._screen_width, self._screen_height)
            )
            pygame.display.set_caption("Elevator Simulation (Gymnasium)")
        else:
            self._screen = pygame.Surface(
                (self._screen_width, self._screen_height)
            )

        self._clock = pygame.time.Clock()
        pygame.font.init()
        self._font = pygame.font.SysFont(None, 24)

        self._floor_height = self._building_height / self.num_floors
        self._building_x = self._waiting_area_width + 50
        self._shaft_start_x = self._building_x + 20
        self._waiting_area_x = 10

    def _render_frame(self):
        import pygame

        if self._screen is None:
            self._init_pygame()

        self._screen.fill((255, 255, 255))

        # Waiting area
        pygame.draw.rect(
            self._screen, (210, 180, 140),
            (self._waiting_area_x, 0, self._waiting_area_width, self._building_height),
        )

        # Building
        pygame.draw.rect(
            self._screen, (200, 200, 200),
            (self._building_x, 0, self._building_width, self._building_height),
        )

        # Floor lines
        for i in range(self.num_floors + 1):
            y = i * self._floor_height
            pygame.draw.line(
                self._screen, (0, 0, 0),
                (self._building_x, y),
                (self._building_x + self._building_width, y), 2,
            )

        # Elevator shafts
        for idx in range(self.num_elevators):
            x = self._shaft_start_x + idx * (self._shaft_width + self._shaft_spacing)
            pygame.draw.rect(
                self._screen, (50, 50, 50),
                (x, 0, self._shaft_width, self._building_height),
            )

        # Elevators
        for idx, elev in enumerate(self.elevators):
            x = self._shaft_start_x + idx * (self._shaft_width + self._shaft_spacing)
            y = (self.num_floors - 1 - elev.current_floor) * self._floor_height

            color = (100, 200, 100) if elev.door_open else (100, 100, 200)
            pygame.draw.rect(
                self._screen, color,
                (x + 2, y + 2, self._shaft_width - 4, self._floor_height - 4),
            )

            txt = self._font.render(str(len(elev.passengers)), True, (255, 255, 255))
            self._screen.blit(txt, (x + 20, y + 15))

        # Stats
        sim_seconds = self.episode_steps
        hours = 8 + sim_seconds // 3600
        minutes = (sim_seconds % 3600) // 60
        time_str = f"{hours:02}:{minutes:02}"

        stats = [
            f"Time: {time_str}",
            f"Waiting: {len(self.waiting_guests)}",
            f"In elevator: {len(self.guests_in_elevator)}",
            f"On floor: {len(self.guests_on_floors)}",
            f"Left: {self.guests_left_building}/{self.max_guests}",
            f"Reward: {self.total_reward:.0f}",
        ]
        for i, line in enumerate(stats):
            txt = self._font.render(line, True, (0, 0, 0))
            self._screen.blit(txt, (self._waiting_area_x + 5, 10 + i * 25))

        # Waiting per floor indicators
        for floor in range(self.num_floors):
            count = sum(1 for g in self.waiting_guests if g.current_floor == floor)
            if count > 0:
                x = self._building_x + self._building_width + 10
                y = (self.num_floors - 1 - floor) * self._floor_height + 10
                txt = self._font.render(f"W:{count}", True, (200, 0, 0))
                self._screen.blit(txt, (x, y))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
