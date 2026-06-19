import simpy
import pytest
from RideRequest import RideRequest
from DestinationRequest import DestinationRequest
from ElevatorException import ElevatorFull
from Wait import Wait


class FakeGuest:
    def __init__(self, env):
        self.env = env
        self.current_floor = 0
        self.target_floor = 5
        self.elevator_id = None
        self.state = "waiting"
        self.id = 0


@pytest.fixture
def env():
    return simpy.Environment()


def test_ride_request_creates_events(env):
    guest = FakeGuest(env)
    req = RideRequest(guest)
    assert req.guest is guest
    assert req.boarded_event is not None
    assert req.arrived_event is not None


def test_destination_request_stores_floor(env):
    guest = FakeGuest(env)
    req = DestinationRequest(7, guest)
    assert req.target_floor == 7
    assert req.guest is guest
    assert req.arrived_event is not None


def test_elevator_full_exception():
    with pytest.raises(ElevatorFull):
        raise ElevatorFull("Elevator is full")


def test_wait_returns_positive_value():
    env = simpy.Environment()
    result = Wait(env, mean=10.0, std=2.0)
    assert result > 0
