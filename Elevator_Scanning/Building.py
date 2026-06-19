import sys
import simpy
import numpy as np
import Dispatcher
import Elevator
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
        stair_speed=3,
        headless=False,
    ):
        # 1) Regular SimPy environment, not real-time
        self.env = env or simpy.Environment()
        self.no_floor_zero = no_floor_zero
        self.headless = headless

        # 2) Pygame base configuration (skip in headless mode)
        self.screen = screen
        if not headless:
            import pygame
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)

        # 3) Parameters
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.elevator_capacity = elevator_capacity
        self.spawn_intervall = spawn_intervall
        self.door_time = door_time
        self.building_width = building_width
        self.building_height = building_height
        self.shaft_width = shaft_width
        self.shaft_spacing = shaft_spacing
        self.waiting_area_width = waiting_area_width
        self.max_guests = max_guests
        self.working_time = working_time
        self.people_left_building = 0
        # 4) Helper structures
        self.floor_counts = [0] * num_floors
        self.riders = []
        self.logs = []
        # 5) Geometry
        self.floor_height = building_height / num_floors
        self.building_x = waiting_area_width + 50
        self.building_y = 0
        self.shaft_start_x = self.building_x + 20
        self.waiting_area_x = 10
        self.episode_steps = 0

        # 6) Colors
        default_colors = {
            "white": (255, 255, 255),
            "gray": (200, 200, 200),
            "dark_gray": (50, 50, 50),
            "black": (0, 0, 0),
            "waiting_area": (210, 180, 140),
        }
        self.colors = colors or default_colors

        # 7) Queues
        self.dispatcher_queue = simpy.Store(self.env)
        self.elevator_queues = [simpy.Store(self.env) for _ in range(num_elevators)]
        self.pickup_queue = []
        self.mutex = simpy.Resource(env, capacity=1)
        # 8) Elevators & Dispatcher
        self.elevators = []
        for idx, q in enumerate(self.elevator_queues):
            elev = Elevator.Elevator(
                env=self.env,
                queue=q,
                capacity=elevator_capacity,
                door_time=door_time,
                id=idx,
                pickup_queue=self.pickup_queue,
                mutex=self.mutex,
                max_floor=self.num_floors - 1,
            )
            self.elevators.append(elev)
        self.dispatcher = Dispatcher.Dispatcher(
            self.env, self.dispatcher_queue, self.elevator_queues
        )
        self.stop_event = self.env.event()
        # 9) Start processes
        self.env.process(self.guest_spawner())
        self.env.process(self.dispatcher.run())

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

    def guest_spawner(self):
        gid = 0
        while True:
            # Guest limit
            if len(self.riders) == self.max_guests == self.people_left_building:
                self.stop_event.succeed()
                break
            if self.max_guests and len(self.riders) >= self.max_guests:
                yield self.env.timeout(1)
                continue

            # Exponential interval
            lam = self.max_guests / self.spawn_intervall
            mean_inter = 1 / lam
            inter = np.random.exponential(mean_inter)
            yield self.env.timeout(inter)

            # Create guest
            g = Guest(
                self.env, gid, self, self.no_floor_zero, working_time=self.working_time
            )
            self.riders.append(g)
            gid += 1

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

        # Calculate time and show it
        sim_seconds = int(self.env.now)
        start_hour = 8
        hours = int(start_hour + (sim_seconds // 3600))
        minutes = int((sim_seconds % 3600) // 60)
        seconds = int(sim_seconds % 60)

        if hours >= 24:
            hours = hours % 24
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        self.screen.blit(
            self.font.render(f"Time: {time_str}", True, self.colors["black"]),
            (self.waiting_area_x + 5, 65),
        )
        # 2) Counters for waiting and on-floor guests
        waiting_area_right = self.waiting_area_x + self.waiting_area_width
        building_left = self.building_x
        gap = building_left - waiting_area_right
        waiting = 0
        waiting_on_floor = [0] * self.num_floors
        on_floor = [0] * self.num_floors

        for g in self.riders:
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
            if not e.door:
                x = self.shaft_start_x + idx * (self.shaft_width + self.shaft_spacing)
                y = (
                    self.building_y
                    + (self.num_floors - 1 - e.current_floor) * self.floor_height
                )
                self.screen.blit(self.elevator_img, (x, y))
                anz = len(e.riders)
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
                anz = len(e.riders)
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
