
# pl15_j20_sim/missile/datalink.py
class DataLink:
    def __init__(self, delay: float = 0.1):
        self.delay = delay
    @staticmethod
    def _pos(tgt: Any) -> cp.ndarray:
        return tgt.position if hasattr(tgt, "position") else cp.asarray(tgt, dtype=cp.float32)
    def send_correction(self, ms, tgt):
        ms_pos = getattr(ms, "position", cp.zeros(3, dtype=cp.float32))
        return self._pos(tgt) - ms_pos

_datalink_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.datalink")
if not hasattr(_datalink_mod, "DataLink"):
    setattr(_datalink_mod, "DataLink", DataLink)
