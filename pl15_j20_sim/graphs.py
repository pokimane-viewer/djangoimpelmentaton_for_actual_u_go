
# pl15_j20_sim/graphs.py
from contextlib import contextmanager

class _GraphProxy:
    def __init__(self):
        self._graph: Optional[cp.cuda.graph.Graph] = None
    def _set_graph(self, g):
        self._graph = g
    def launch(self, stream: Optional[cp.cuda.Stream] = None):
        if self._graph is None:
            raise RuntimeError("CUDA graph not captured.")
        if hasattr(self._graph, "launch"):
            self._graph.launch(stream) if stream else self._graph.launch()
        elif hasattr(self._graph, "instantiate"):
            instance = self._graph.instantiate()
            instance.launch(stream) if stream else instance.launch()
        else:
            raise AttributeError("Unsupported CUDA Graph object: no launch method.")

@contextmanager
def capture_graph():
    s = cp.cuda.Stream(non_blocking=True)
    p = _GraphProxy()
    with s:
        if hasattr(s, "begin_capture"):
            s.begin_capture()
            yield s, p
            if hasattr(s, "end_capture"):
                p._set_graph(s.end_capture())
        else:
            yield s, p

_graphs_mod = _ensure_submodule(f"{_PKG_ROOT}.graphs")
for _n in ("capture_graph", "_GraphProxy"):
    if not hasattr(_graphs_mod, _n):
        setattr(_graphs_mod, _n, globals()[_n])