# pl15_j20_sim/missile/flight_dynamics.py
class FlightDynamics:
    def __init__(self, mass: float, thrust_profile: Callable[[float], float]):
        self.mass, self.thrust_profile = mass, thrust_profile
    def propagate(self, state: Any, dt: float):
        acc = self.thrust_profile(state.time) / self.mass - 9.81
        state.vel += acc * dt
        state.pos += state.vel * dt
        state.time += dt
        return state

_fdyn_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.flight_dynamics")
if not hasattr(_fdyn_mod, "FlightDynamics"):
    setattr(_fdyn_mod, "FlightDynamics", FlightDynamics)
