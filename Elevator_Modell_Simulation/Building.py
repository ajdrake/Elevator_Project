import sys
import simpy
import numpy as np
from Elevator import Elevator
from Guest import Guest


class Building:

    def __init__(
        self,
        screen=None,
        env=None,
        num_floors=10,
        num_elevators=3,
        elevator_capacity=5,
        door_time=30,
        building_width=600,
        building_height=800,
        shaft_width=60,
        shaft_spacing=10,
        waiting_area_width=140,
        visualize_every=1,
        max_guests=50,
        working_time=20,
        colors=None,
        no_floor_zero=False,
        spawn_intervall=120,
        modell=None,
        headless=False,
    ):
        # 1) Normal SimPy environment, not real-time
        self.env = env or simpy.Environment()
        self.no_floor_zero = no_floor_zero
        self.headless = headless
        # 2) Pygame base config (skip in headless mode)
        self.screen = screen
        if not headless:
            import pygame
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)
        self.model = modell
        # 3) Parameters
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.elevator_capacity = elevator_capacity
        self.spawn_intervall = spawn_intervall
        # Door open/close time
        self.door_time = door_time
        self.building_width = building_width
        self.building_height = building_height
        self.shaft_width = shaft_width
        self.shaft_spacing = shaft_spacing
        self.waiting_area_width = waiting_area_width
        self.max_guests = max_guests
        self.working_time = working_time
        self.guests_left_building = 0
        # 4) Helper structures
        self.floor_counts = [0] * num_floors
        self.logs = []
        # 5) Geometry
        self.floor_height = building_height / num_floors
        self.building_x = waiting_area_width + 50
        self.building_y = 0
        self.shaft_start_x = self.building_x + 20
        self.waiting_area_x = 10
        self.waiting_guests = []
        self.guests_on_floors = []
        self.allguests = []
        self.left_guests = []
        self.guests_in_elevator = []
        self.episode_steps = 0
        self._next_guest_id = 0
        self.guests_in_building = 0
        # For Poisson spawn:
        self.lam = self.max_guests / self.spawn_intervall
        self.mean_inter = 1 / self.lam
        self._time_since_last_spawn = 0.0
        self.time_until_next_arrival = np.random.exponential(self.mean_inter)
        self.sim_step_size = 1
        # 6) Colors
        default_colors = {
            "white": (255, 255, 255),
            "gray": (200, 200, 200),
            "dark_gray": (50, 50, 50),
            "black": (0, 0, 0),
            "waiting_area": (210, 180, 140),
        }
        self.colors = colors or default_colors
        self.pickup_queue = []
        self.total_reward = 0
        # 8) Elevators & dispatcher
        self.elevators = []
        self._next_elevator_id = 0
        for _ in range(self.num_elevators):
            elevator = Elevator(
                self,
                id=self._next_elevator_id,
                min_floor=0,
                max_floor=self.num_floors - 1,
                capacity=5,
                door_time=4,
                ride_time=4,
                sim_step_size=1,
            )
            self._next_elevator_id += 1
            self.elevators.append(elevator)

        self.stop_event = self.env.event()
        # 9) Start processes
        self.env.process(self.step())

        # 10) Load graphics (skip in headless mode)
        if not headless:
            import pygame
            self.elevator_img = pygame.transform.scale(
                pygame.image.load("elevator.bmp").convert(),
                (shaft_width, self.floor_height),
            )
            self.elevator_img_open = pygame.transform.scale(
                pygame.image.load("elevator_open.bmp").convert(),
                (shaft_width, self.floor_height),
            )
            self.GUEST_SIZE = 40
            self.guest_img = pygame.transform.scale(
                pygame.image.load("stick_1.bmp").convert_alpha(),
                (self.GUEST_SIZE, self.GUEST_SIZE),
            )

        # 11) Visualization interval & buttons
        self.visualize_every = visualize_every
        if not headless:
            self._setup_buttons()

    def _setup_buttons(self):
        import pygame
        btn_w, btn_h = 30, 30
        x0 = self.waiting_area_x + 5
        y0 = 5
        self.btn_minus = pygame.Rect(x0, y0, btn_w, btn_h)
        self.btn_plus = pygame.Rect(x0 + 40, y0, btn_w, btn_h)
        self.btn_times2 = pygame.Rect(x0 + 80, y0, btn_w, btn_h)

    def log(self, time, guest_id, mode, wait_time, travel_time):
        """Writes a log entry."""
        self.logs.append(
            {
                "time": time,
                "guest_id": guest_id,
                "mode": mode,  # 'elevator_waiting', 'elevator_drive' or 'stairs'
                "wait_time": wait_time,  # None if not relevant
                "travel_time": travel_time,  # None if not relevant
            }
        )

    def _spawn_guest(self, direction="up", floor=None):
        if (self._next_guest_id) >= self.max_guests:
            return
        if floor is None:
            start_floor = 0  # In the morning, guests always start on the ground floor
        else:
            start_floor = floor

        if direction == "up":
            possible_targets = [f for f in range(0, self.num_floors)]

        target_floor = np.random.choice(possible_targets)
        guest = Guest(
            self,
            guest_id=self._next_guest_id,
            start_floor=start_floor,
            target_floor=target_floor,
            current_floor=start_floor,
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
            return

        self.waiting_guests.append(guest)

    def draw(self):
        import pygame
        # 1) Background
        self.screen.fill(self.colors["white"])
        pygame.draw.rect(
            self.screen,
            self.colors["waiting_area"],
            (self.waiting_area_x, 0, self.waiting_area_width, self.building_height),
        )
        pygame.draw.rect(
            self.screen,
            self.colors["gray"],
            (
                self.building_x,
                self.building_y,
                self.building_width,
                self.building_height,
            ),
        )
        for i in range(self.num_floors + 1):
            y = self.building_y + i * self.floor_height
            pygame.draw.line(
                self.screen,
                self.colors["black"],
                (self.building_x, y),
                (self.building_x + self.building_width, y),
                2,
            )
        for idx in range(self.num_elevators):
            x = self.shaft_start_x + idx * (self.shaft_width + self.shaft_spacing)
            pygame.draw.rect(
                self.screen,
                self.colors["dark_gray"],
                (x, self.building_y, self.shaft_width, self.building_height),
            )

            # Buttons
            pygame.draw.rect(self.screen, (180, 180, 180), self.btn_minus)
            pygame.draw.rect(self.screen, (180, 180, 180), self.btn_plus)
            pygame.draw.rect(self.screen, (180, 180, 180), self.btn_times2)
            self.screen.blit(
                self.font.render("–", True, self.colors["black"]),
                (self.btn_minus.x + 8, self.btn_minus.y + 3),
            )
            self.screen.blit(
                self.font.render("+", True, self.colors["black"]),
                (self.btn_plus.x + 9, self.btn_plus.y + 3),
            )
            self.screen.blit(
                self.font.render("×2", True, self.colors["black"]),
                (self.btn_times2.x + 2, self.btn_times2.y + 3),
            )

            self.screen.blit(
                self.font.render(
                    f"Show every {self.visualize_every}", True, self.colors["black"]
                ),
                (self.waiting_area_x + 5, 40),
            )

            # Calculate time
            sim_seconds = self.episode_steps
            start_hour = 8
            hours = start_hour + (sim_seconds // 3600)
            minutes = (sim_seconds % 3600) // 60
            seconds = sim_seconds % 60
            if hours >= 24:
                hours = hours % 24
            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            self.screen.blit(
                self.font.render(f"Time: {time_str}", True, self.colors["black"]),
                (self.waiting_area_x + 5, 65),  # Etwas unterhalb von "Show every ..."
            )

        # 2) Counters for waiting, on floor, waiting on floor, etc.
        waiting_area_right = self.waiting_area_x + self.waiting_area_width
        building_left = self.building_x
        gap = building_left - waiting_area_right
        waiting = 0
        waiting_on_floor = [0] * self.num_floors
        on_floor = [0] * self.num_floors

        for g in self.allguests:
            if g.state == "waiting":
                waiting += 1
            if g.state == "on_floor":
                on_floor[g.current_floor] += 1
            if g.state == "waiting_on_floor":
                waiting_on_floor[g.current_floor] += 1

        # Waiting area counter
        x = self.waiting_area_x
        y = (
            self.building_y
            + (self.num_floors - 1) * self.floor_height
            + (self.floor_height - self.GUEST_SIZE)
        )
        self.screen.blit(self.guest_img, (x, y))
        txt = self.font.render(str(waiting), True, self.colors["black"])
        x = self.waiting_area_x + self.GUEST_SIZE
        y = self.building_y + (self.num_floors - 1) * self.floor_height
        self.screen.blit(txt, (x, y))
        # Floor waiting counter
        x = self.shaft_start_x + self.num_elevators * (
            self.shaft_width + self.shaft_spacing
        )
        xx = x + self.GUEST_SIZE
        for i in range(self.num_floors):
            y = (
                self.building_y
                + (self.num_floors - 1 - i) * self.floor_height
                + (self.floor_height - self.GUEST_SIZE)
            )
            self.screen.blit(self.guest_img, (x, y))
            txt = self.font.render(str(waiting_on_floor[i]), True, self.colors["black"])
            self.screen.blit(txt, (xx, y))
        # Floor counter
        x = self.building_x + self.building_width - self.GUEST_SIZE * 2
        xx = x + self.GUEST_SIZE
        for i in range(self.num_floors):
            y = (
                self.building_y
                + (self.num_floors - 1 - i) * self.floor_height
                + (self.floor_height - self.GUEST_SIZE)
            )
            self.screen.blit(self.guest_img, (x, y))
            txt = self.font.render(str(on_floor[i]), True, self.colors["black"])
            self.screen.blit(txt, (xx, y))

        # 3) Elevators
        for idx, e in enumerate(self.elevators):
            if not e.door_open:
                x = self.shaft_start_x + idx * (self.shaft_width + self.shaft_spacing)
                y = (
                    self.building_y
                    + (self.num_floors - 1 - e.current_floor) * self.floor_height
                )
                self.screen.blit(self.elevator_img, (x, y))
                anz = len(e.passengers)
                x = (
                    self.shaft_start_x
                    + idx * (self.shaft_width + self.shaft_spacing)
                    + (self.shaft_width - self.GUEST_SIZE) // 2
                )
                y = (
                    self.building_y
                    + (self.num_floors - 1 - e.current_floor) * self.floor_height
                    + (self.floor_height - self.GUEST_SIZE)
                )
                self.screen.blit(self.guest_img, (x, y))
                txt = self.font.render(str(anz), True, self.colors["black"])
                ty = y - (self.floor_height - self.GUEST_SIZE)
                self.screen.blit(txt, (x, ty))
            else:
                x = self.shaft_start_x + idx * (self.shaft_width + self.shaft_spacing)
                y = (
                    self.building_y
                    + (self.num_floors - 1 - e.current_floor) * self.floor_height
                )
                self.screen.blit(self.elevator_img_open, (x, y))
                anz = len(e.passengers)
                x = (
                    self.shaft_start_x
                    + idx * (self.shaft_width + self.shaft_spacing)
                    + (self.shaft_width - self.GUEST_SIZE) // 2
                )
                y = (
                    self.building_y
                    + (self.num_floors - 1 - e.current_floor) * self.floor_height
                    + (self.floor_height - self.GUEST_SIZE)
                )
                self.screen.blit(self.guest_img, (x, y))
                txt = self.font.render(str(anz), True, self.colors["black"])
                ty = y - (self.floor_height - self.GUEST_SIZE)
                self.screen.blit(txt, (x, ty))

    def get_action_mask_for_elevator(self, elevator, min_floor=0, max_floor=9):
        mask = np.zeros(3, dtype=bool)
        # 0 = wait: always allowed
        mask[0] = True
        if elevator.busy_time > 0:
            return [1, 1, 1]
        if elevator.busy_time <= 0 and elevator.pending_action is not None:
            if elevator.pending_action[0] == "close":
                mask[0] = False
            elevator.pending_action = None

        if not elevator._guests_waiting_or_leaving():
            mask[0] = False

        # 1 = up: only if door closed and not on top floor
        if not elevator.door_open and elevator.current_floor < max_floor:
            mask[1] = True

        # 2 = down: only if door closed and not on bottom floor
        if not elevator.door_open and elevator.current_floor > min_floor:
            mask[2] = True

        return mask

    def _get_obs(self, elevator):
        obs = []
        obs.append(elevator.current_floor)
        obs.append(len(elevator.passengers))
        passenger_dest_hist = [0] * 10
        for p in elevator.passengers:
            passenger_dest_hist[p.target_floor] += 1
        obs.extend(passenger_dest_hist)

        # Fill missing elevators with zeros
        for _ in range(2):
            obs.append(0)  # current_floor
            obs.append(0)  # len(passengers)
            obs.extend([0] * 10)  # target histogram

        # Waiting guests per floor (all elevators)
        waiting_per_floor = [0] * 10
        for g in self.waiting_guests:
            waiting_per_floor[g.current_floor] += 1
        obs.extend(waiting_per_floor)

        return np.array(obs, dtype=np.int32)

    def get_action_mask(self):
        # self.elevators: list of Elevator objects
        masks = []
        for elev in self.elevators:
            mask = self.get_action_mask_for_elevator(
                elev, min_floor=0, max_floor=self.num_floors - 1
            )
            masks.append(mask)
        # Result is shape (num_elevators, 3)
        return np.array(masks, dtype=bool)

    def step(self):
        while True:
            if len(self.allguests) == self.max_guests == self.guests_left_building:
                print("Step: ", self.episode_steps)
                print("Guests_left_building: ", self.guests_left_building)
                print("Guests_in_building: ", self.guests_in_building)
                print("Total_reward: ", self.total_reward)
                self.stop_event.succeed()
                break
            reward = 0
            self.episode_steps += 1
            print(self.episode_steps)
            for guest in self.guests_on_floors:
                guest.step(1, False)

            # === POISSON GUEST SPAWN ===
            self._time_since_last_spawn += self.sim_step_size
            while (
                (self.guests_in_building + self.guests_left_building) < self.max_guests
                and self._time_since_last_spawn >= self.time_until_next_arrival
            ):
                self._spawn_guest(direction="up")
                self._time_since_last_spawn -= self.time_until_next_arrival
                self.time_until_next_arrival = np.random.exponential(self.mean_inter)
            action_maskss = self.get_action_mask()
            actions = []
            for i in range(3):
                obs = self._get_obs(self.elevators[i])
                action, _ = self.model.predict(
                    obs, action_masks=action_maskss[i], deterministic=True
                )
                actions.append(action)

            for i, action in enumerate(actions):
                self.elevators[i].do_action(action)

            # Finish actions + reward for dropoff
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
                                h = 20 - waited_steps
                                reward += max(1, h)

            # Finish actions + reward for boarding + reward for dropoff
            for elev in self.elevators:
                if elev.busy_time <= 0 and elev.pending_action is not None:
                    if elev.pending_action[0] == "open":
                        boarded_guests = elev.board_guests(
                            self.waiting_guests,
                        )
                        if boarded_guests:
                            for g in boarded_guests:
                                self.waiting_guests.remove(g)
                                self.guests_in_elevator.append(g)
                                waited_steps = int(
                                    (self.episode_steps - g.waiting_since) / 60
                                )
                                h = 10 - waited_steps
                                reward += max(1, h)
            self.total_reward += reward
            yield self.env.timeout(1)

    def run(self):
        if self.headless:
            yield from self._run_headless()
        else:
            yield from self._run_visual()

    def _run_headless(self):
        while True:
            yield self.env.timeout(1)

    def _run_visual(self):
        import pygame
        running = True
        skip_to_end = False

        while running:
            for _ in range(int(self.visualize_every)):
                if skip_to_end:
                    break
                step_start_time = pygame.time.get_ticks()
                while (
                    not skip_to_end and pygame.time.get_ticks() - step_start_time < 1000
                ):
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif e.type == pygame.MOUSEBUTTONDOWN:
                            if self.btn_minus.collidepoint(e.pos):
                                self.visualize_every = max(1, self.visualize_every - 1)
                            elif self.btn_plus.collidepoint(e.pos):
                                self.visualize_every += 1
                            elif self.btn_times2.collidepoint(e.pos):
                                self.visualize_every = min(
                                    3600, self.visualize_every * 2
                                )
                        elif e.type == pygame.KEYDOWN:
                            if e.key == pygame.K_SPACE:
                                skip_to_end = True
                                break

                    self.draw()
                    pygame.display.flip()
                    self.clock.tick(60)

                yield self.env.timeout(1 * self.visualize_every)

            if skip_to_end:
                while True:
                    yield self.env.timeout(1)
