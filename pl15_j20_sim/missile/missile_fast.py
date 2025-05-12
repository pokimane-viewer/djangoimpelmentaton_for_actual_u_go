
# pl15_j20_sim/missile/missile_fast.py
class RadarSeekerFast(RadarSeeker):
    __slots__ = ()
    def scan(self, env, own) -> Tuple[cp.ndarray, cp.ndarray]:
        t   = env.get_targets()[:, :3]
        rel = t - own.position
        r   = cp.linalg.norm(rel, axis=1)
        m   = r < self.range_max()
        return t[m], r[m]

class PL15MissileFast(PL15Missile):
    __slots__ = ()
    def __init__(self, st: Any, cfg: Dict[str, Dict[str, Any]]):
        super().__init__(st, cfg)
        self.seeker = RadarSeekerFast(**cfg["seeker"])
        self._rel = cp.zeros(3, dtype=cp.float32)
        self._vel = cp.zeros(3, dtype=cp.float32)
        self._acc = cp.zeros(3, dtype=cp.float32)

    def update(self, env, dt):
        tgt, _ = self.seeker.scan(env, self.state)
        if tgt.size:
            self._rel[:] = tgt[0] - self.state.position
            self._vel[:] = -self.state.velocity
            pn_guidance_kernel(self._rel[0:1], self._rel[1:2], self._rel[2:3],
                               self._vel[0:1], self._vel[1:2], self._vel[2:3],
                               self.guidance.N,
                               self._acc[0:1], self._acc[1:2], self._acc[2:3])
        else:
            self._acc.fill(0)
        self.state.position, self.state.velocity = integrate_state(
            self.state.position, self.state.velocity, self._acc, dt
        )
        self.state.time += dt

_mfast_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.missile_fast")
for _n in ("PL15MissileFast", "RadarSeekerFast"):
    if not hasattr(_mfast_mod, _n):
        setattr(_mfast_mod, _n, globals()[_n])
