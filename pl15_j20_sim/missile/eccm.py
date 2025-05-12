
# pl15_j20_sim/missile/eccm.py
class ECCM:
    def __init__(self, adaptive_gain: float = 1.):
        self.adaptive_gain = adaptive_gain
    def mitigate_jamming(self, radar):
        return radar
    def deploy_decoys(self): ...

_eccm_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.eccm")
if not hasattr(_eccm_mod, "ECCM"):
    setattr(_eccm_mod, "ECCM", ECCM)
