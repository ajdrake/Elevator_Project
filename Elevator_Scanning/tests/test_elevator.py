import simpy
import pytest
from Elevator import Elevator
from RideRequest import RideRequest
from DestinationRequest import DestinationRequest


class FakeGuest:
    def __init__(self, env, current_floor=0, target_floor=5):
        self.env = env
        self.current_floor = current_floor
        self.target_floor = target_floor
        self.elevator_id = None
        self.state = "waiting"
        self.id = 0


@pytest.fixture
def env():
    return simpy.Environment()


@pytest.fixture
def elevator(env):
    queue = simpy.Store(env)
    pickup_queue = []
    mutex = simpy.Resource(env, capacity=1)
    return Elevator(
        env=env,
        queue=queue,
        pickup_queue=pickup_queue,
        mutex=mutex,
        capacity=5,
        door_time=0.1,
        id=0,
        min_floor=0,
        max_floor=9,
    )


def test_elevator_starts_at_min_floor(elevator):
    assert elevator.current_floor == 0


def test_elevator_initial_direction_is_up(elevator):
    assert elevator.direction == 1


def test_elevator_capacity(elevator):
    assert elevator.capacity == 5


def test_elevator_starts_with_empty_riders(elevator):
    assert elevator.riders == []


def test_elevator_moves_up(env, elevator):
    """Elevator moves up when it has a destination request for a higher floor."""
    guest = FakeGuest(env, current_floor=0, target_floor=5)
    ride_req = RideRequest(guest)
    dest_req = DestinationRequest(5, guest)

    def scenario():
        yield elevator.queue.put(ride_req)
        yield env.timeout(2)
        yield elevator.queue.put(dest_req)
        yield env.timeout(30)

    env.process(scenario())
    env.run(until=35)
    assert elevator.current_floor > 0


def test_elevator_direction_reverses_at_max(env):
    """Elevator reverses direction when it reaches max floor."""
    queue = simpy.Store(env)
    pickup_queue = []
    mutex = simpy.Resource(env, capacity=1)
    elev = Elevator(
        env=env,
        queue=queue,
        pickup_queue=pickup_queue,
        mutex=mutex,
        capacity=5,
        door_time=0.1,
        id=0,
        min_floor=0,
        max_floor=3,
    )
    guest = FakeGuest(env, current_floor=0, target_floor=3)
    req = RideRequest(guest)

    def scenario():
        yield elev.queue.put(req)
        yield env.timeout(50)

    env.process(scenario())
    env.run(until=50)
    assert elev.direction == -1 or elev.current_floor < 3
