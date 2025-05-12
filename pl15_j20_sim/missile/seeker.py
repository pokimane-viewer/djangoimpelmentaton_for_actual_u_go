
# pl15_j20_sim/missile/seeker.py
class RadarSeeker:
    def __init__(self, update_rate: float = 100., sensitivity: float = 1e-6, active: bool = True):
        self.update_rate, self.sensitivity, self.active = update_rate, sensitivity, active
    def scan(self, environment: Any, own_state: Any) -> Tuple[cp.ndarray, cp.ndarray]:
        tgt = environment.get_targets()
        rel = tgt[:, :3] - own_state.position
        rng = cp.linalg.norm(rel, axis=1)
        m   = rng < self.range_max()
        return tgt[m], rng[m]
    def range_max(self) -> float:
        return cp.inf

_seeker_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.seeker")
if not hasattr(_seeker_mod, "RadarSeeker"):
    setattr(_seeker_mod, "RadarSeeker", RadarSeeker)
