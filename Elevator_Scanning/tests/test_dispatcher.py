import simpy
import pytest
from Dispatcher import Dispatcher
from RideRequest import RideRequest
from DestinationRequest import DestinationRequest


class FakeGuest:
    def __init__(self, env, target_floor=5):
        self.env = env
        self.current_floor = 0
        self.target_floor = target_floor
        self.elevator_id = None
        self.state = "waiting"
        self.id = 0


@pytest.fixture
def env():
    return simpy.Environment()


def test_dispatcher_routes_ride_request(env):
    """Dispatcher puts a RideRequest into one of the elevator queues."""
    dispatcher_queue = simpy.Store(env)
    elevator_queues = [simpy.Store(env) for _ in range(3)]
    dispatcher = Dispatcher(env, dispatcher_queue, elevator_queues)

    guest = FakeGuest(env, target_floor=5)
    req = RideRequest(guest)

    def scenario():
        yield dispatcher_queue.put(req)
        yield env.timeout(1)

    env.process(dispatcher.run())
    env.process(scenario())
    env.run(until=2)

    assert guest.elevator_id is not None
    assert 0 <= guest.elevator_id < 3
    assigned_queue = elevator_queues[guest.elevator_id]
    assert len(assigned_queue.items) == 1


def test_dispatcher_routes_destination_request(env):
    """Dispatcher routes a DestinationRequest to the guest's assigned elevator."""
    dispatcher_queue = simpy.Store(env)
    elevator_queues = [simpy.Store(env) for _ in range(3)]
    dispatcher = Dispatcher(env, dispatcher_queue, elevator_queues)

    guest = FakeGuest(env, target_floor=7)
    guest.elevator_id = 1
    dest_req = DestinationRequest(7, guest)

    def scenario():
        yield dispatcher_queue.put(dest_req)
        yield env.timeout(1)

    env.process(dispatcher.run())
    env.process(scenario())
    env.run(until=2)

    assert len(elevator_queues[1].items) == 1
    assert elevator_queues[1].items[0] == dest_req


def test_dispatcher_handles_multiple_requests(env):
    """Dispatcher handles a sequence of requests correctly."""
    dispatcher_queue = simpy.Store(env)
    elevator_queues = [simpy.Store(env) for _ in range(2)]
    dispatcher = Dispatcher(env, dispatcher_queue, elevator_queues)

    guests = [FakeGuest(env, target_floor=i + 1) for i in range(5)]
    requests = [RideRequest(g) for g in guests]

    def scenario():
        for req in requests:
            yield dispatcher_queue.put(req)
        yield env.timeout(5)

    env.process(dispatcher.run())
    env.process(scenario())
    env.run(until=6)

    total_routed = sum(len(q.items) for q in elevator_queues)
    assert total_routed == 5
