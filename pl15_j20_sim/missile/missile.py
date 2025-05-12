

# pl15_j20_sim/missile/missile.py
class PL15Missile:
    def __init__(self, st: Any, cfg: Dict[str, Dict[str, Any]]):
        self.state  = st
        self.seeker = RadarSeeker(**cfg["seeker"])
        self.guidance = GuidanceLaw(**cfg["guidance"])
        self.dynamics = FlightDynamics(**cfg["flight_dynamics"])
        self.datalink = DataLink(**cfg["datalink"])
        self.eccm     = ECCM(**cfg["eccm"])

    def _extract_rel(self, tgt: Any) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        if hasattr(tgt, "position"):
            pos = tgt.position
            vel = getattr(tgt, "velocity", cp.zeros_like(pos))
        else:
            pos = cp.asarray(tgt, dtype=cp.float32)
            vel = cp.zeros_like(pos)
        rel_p = pos - self.state.position
        rel_v = vel - self.state.velocity
        return pos, rel_p, rel_v

    def update(self, env: Any, dt: float):
        tgt, _ = self.seeker.scan(env, self.state)
        if tgt.size:
            tgt_pos, rel_p, rel_v = self._extract_rel(tgt[0])
            acc = self.guidance.compute_steering(rel_p, rel_v)
        else:
            acc = cp.zeros(3, dtype=cp.float32)
        self.state = self.dynamics.propagate(self.state, dt)
        self.state.velocity += acc * dt
        if self.datalink and tgt.size:
            self.state.velocity += self.datalink.send_correction(self.state, tgt_pos)

_missile_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.missile")
if not hasattr(_missile_mod, "PL15Missile"):
    setattr(_missile_mod, "PL15Missile", PL15Missile)