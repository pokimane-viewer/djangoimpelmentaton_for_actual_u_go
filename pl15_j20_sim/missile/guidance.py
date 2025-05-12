
# pl15_j20_sim/missile/guidance.py
class GuidanceLaw:
    def __init__(self, N: float = 3.):
        self.N = N
    def compute_steering(self, rel_p: cp.ndarray, rel_v: cp.ndarray) -> cp.ndarray:
        los   = rel_p / cp.linalg.norm(rel_p)
        los_r = cp.cross(rel_p, rel_v) / (cp.linalg.norm(rel_p)**2 + 1e-6)
        return self.N * los_r * (-cp.dot(rel_v, los))

_guidance_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.guidance")
if not hasattr(_guidance_mod, "GuidanceLaw"):
    setattr(_guidance_mod, "GuidanceLaw", GuidanceLaw)

