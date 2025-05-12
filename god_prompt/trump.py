Skip to content
Chat history



New project
Today
You said:
Bo commands you to CRT-type reduce to explain in as best you can: if this optimization real world scenario went to infinitiy, would NVIIDIA's capacity to process enough floating point 32 in CUDA in embarassingly parallel be the limitation? what would be the limitation in 6th and 7th generation stealth fighters and hypersonic or better, how great can a missile get? if everything went to infinity?import os
import sys
import math
import json
import time
import queue
import random
import pathlib
import subprocess
import threading
import webbrowser
from typing import Any, Dict, List, Tuple, Callable
from contextlib import suppress
from dataclasses import dataclass, field as dataclass_field
from numpy.typing import NDArray
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np

def _ensure_submodule(name: str): return sys.modules[name]
_PKG_ROOT = "pl15_j20_sim"

def _identity_eq_hash(cls):
    """
    Identity-based equality and hash so that each instance is considered unique
    regardless of internal content. This is important for certain usage patterns
    in the simulation where we do not want two identical states merging or
    overshadowing each other.
    """
    def _eq(self, other): return self is other
    def _hash(self): return id(self)
    cls.__eq__, cls.__hash__ = _eq, _hash
    return cls

class _FallbackAircraft:
    """
    Basic fallback for an aircraft model. 
    Now includes 'additional_weight' to incorporate WeightedCRT synergy.
    """
    def __init__(self, state: Any, config: Dict[str, Any] = None, additional_weight: float = 1.0):
        self.state = state
        self.config = config or {}
        self.destroyed = False
        self.additional_weight = additional_weight

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        st = _resolve_state(self)
        # basic physics
        accel = cp.asarray(self.config.get("gravity", [0, 0, -9.81]), dtype=cp.float32)
        accel += cp.asarray(self.config.get("extra_accel", [0, 0, 0]), dtype=cp.float32)
        st.velocity += accel * dt
        st.position += st.velocity * dt
        st.time += dt
        self.state = st

class CombatResultsTable:
    """
    Simple CRT determining if two entities destroy each other based on range.
    """
    _KILL_PROB_TABLE = {50.: .9, 100.: .7, 150.: .3, 250.: .05}
    _MAX_ENGAGE_RANGE = 250.

    @staticmethod
    def _kill_probability(r: float) -> float:
        if r >= CombatResultsTable._MAX_ENGAGE_RANGE:
            return 0.
        ks = sorted(CombatResultsTable._KILL_PROB_TABLE)
        for lo, hi in zip(ks[:-1], ks[1:]):
            if lo <= r < hi:
                p_lo = CombatResultsTable._KILL_PROB_TABLE[lo]
                p_hi = CombatResultsTable._KILL_PROB_TABLE[hi]
                α = (r - lo) / (hi - lo)
                return p_lo + α * (p_hi - p_lo)
        return CombatResultsTable._KILL_PROB_TABLE[ks[0]]

    def evaluate_engagement(self, a1, a2):
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        if random.random() < (pk := self._kill_probability(d)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[CRT] Engagement at {d:.1f} m – {loser.__class__.__name__} destroyed (Pk={pk:.2f})")

class TaiwanConflictCRTManager:
    """
    Manages a list of aircraft in an environment and applies a standard CombatResultsTable.
    """
    def __init__(self, env, aircraft, crt):
        self.environment = env
        self.aircraft = aircraft
        self.crt = crt

    def step(self, dt: float):
        for ac in self.aircraft:
            if getattr(ac, "destroyed", False):
                continue
            if hasattr(ac, "update"):
                ac.update(dt)
        n = len(self.aircraft)
        for i in range(n):
            for j in range(i + 1, n):
                ai = self.aircraft[i]
                aj = self.aircraft[j]
                if getattr(ai, "destroyed", False) or getattr(aj, "destroyed", False):
                    continue
                self.crt.evaluate_engagement(ai, aj)

class agi:
    """
    Stub for advanced AI or logic functions (placeholder).
    """
    monitor_states = staticmethod(lambda *a, **k: None)
    apply_failsafe = staticmethod(lambda *a, **k: None)
    advanced_cooperation = staticmethod(lambda *a, **k: None)

@dataclass
class RadarSeeker:
    update_rate: float = 100.
    sensitivity: float = 1e-6
    active: bool = True

    def track_target(self, tp: cp.ndarray, op: cp.ndarray) -> bool:
        if not self.active:
            return False
        d = cp.linalg.norm(tp - op)
        snr = 1 / (d**4 * self.sensitivity + 1e-9)
        return snr > 1.

@dataclass
class RadarSeekerFast(RadarSeeker):
    """
    Subclass for advanced fast radar seeker logic.
    """
    ...

@dataclass
class PL15MissileFast:
    """
    Basic fast model for PL-15 missile.
    """
    state: Any
    config: Dict[str, Any]
    seeker: RadarSeekerFast = dataclass_field(init=False)

    def __post_init__(self):
        self.seeker = RadarSeekerFast(**self.config.get("seeker", {}))

    def update(self, dt: float = .05):
        if hasattr(self.state, "velocity") and hasattr(self.state, "position"):
            self.state.position += self.state.velocity * dt
            if hasattr(self.state, "time"):
                self.state.time += dt

@dataclass(slots=True)
class EntityState:
    """
    General-purpose entity state, used for GPU or CPU-based positions/velocities.
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.
    acceleration: cp.ndarray = dataclass_field(default_factory=lambda: cp.zeros(3, dtype=cp.float32))

    def as_gpu(self):
        self.position = cp.asarray(self.position() if callable(self.position) else self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity() if callable(self.velocity) else self.velocity, dtype=cp.float32)
        if not isinstance(self.acceleration, cp.ndarray):
            self.acceleration = cp.asarray(self.acceleration, dtype=cp.float32)

    @property
    def pos(self):
        return self.position

    @pos.setter
    def pos(self, v):
        self.position = v

    @property
    def vel(self):
        return self.velocity

    @vel.setter
    def vel(self, v):
        self.velocity = v

def get_targets():
    return cp.array([[50., 0., 0.]], dtype=cp.float32)

class Terrain:
    """
    Terrain model stub.
    """
    def height_at(self, x, y):
        return 0.

class Weather:
    """
    Weather model stub.
    """
    def __init__(self, wind=(0, 0, 0), visibility=100000.):
        self.wind = cp.asarray(wind, dtype=cp.float32)
        self.visibility = visibility

class RFEnvironment:
    """
    RF environment combining terrain + weather.
    """
    def __init__(self, t, w):
        self.terrain = t
        self.weather = w

class SimpleEnvironment:
    """
    Simple environment with stubs for terrain, weather, and targets.
    """
    def __init__(self):
        self.terrain = Terrain()
        self.weather = Weather()
        self.rf_env = RFEnvironment(self.terrain, self.weather)
        self.get_targets = get_targets

def _ensure_array(x, shape=(3,), dtype=cp.float32):
    if callable(x):
        with suppress(Exception):
            x = x()
    if x is None:
        x = cp.zeros(shape, dtype=dtype)
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x, dtype=dtype)
    return x

def _resolve_state(obj):
    """
    Utility to get a consistent state object from an entity, falling back if needed.
    """
    st = getattr(obj, "state", None)
    if callable(st):
        with suppress(Exception):
            st = st()
    if st is None or not (hasattr(st, "position") and hasattr(st, "velocity")):
        if hasattr(obj, "position") and hasattr(obj, "velocity"):
            st = obj
        else:
            st = EntityState(cp.zeros(3, dtype=cp.float32), cp.zeros(3, dtype=cp.float32))
    st.position = _ensure_array(getattr(st, "position", None))
    st.velocity = _ensure_array(getattr(st, "velocity", None))
    return st

class EngagementManager:
    """
    Manages missiles separately for a scenario, with a simple update loop.
    """
    def __init__(self, env, ac, ms):
        self.env = env
        self.aircraft = ac
        self.missiles = ms

    def step(self, dt):
        for m in self.missiles:
            if hasattr(m, "update"):
                m.update(dt)

class _GraphStub:
    """
    Stub for a graph visualisation system that can be replaced by real code.
    """
    def launch(self):
        print("[GRAPH] Graph visualisation (stub) launched.")

class capture_graph:
    """
    Context manager stub for capturing graph data.
    """
    def __enter__(self):
        return None, _GraphStub()

    def __exit__(self, *a):
        ...

def _air_density(alt: float) -> float:
    """
    Rough exponential approximation for air density variation with altitude.
    """
    ρ0, h = 1.225, 8500.
    return ρ0 * math.exp(-alt / h)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F22Aircraft(_FallbackAircraft):
    """
    Specialized F22 flight model, inherits _FallbackAircraft plus an additional_weight.
    """
    additional_weight: float = 1.0

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        lift = .05 * self.state.velocity[0]
        drag = .01 * (self.state.velocity[0] ** 2)
        thrust = cp.array([20., 0., 0.], dtype=cp.float32)
        acc = thrust - cp.array([drag, 0., 9.81], dtype=cp.float32)
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

# ---------------------------------------------------------------------------------------------
# CHANGED HERE: F35 MODEL IS NOW CONVERTED TO J20-TYPE LOGIC AND INTELLIGENCE (MODERN APPROACH).
# ---------------------------------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft(_FallbackAircraft):
    """
    Specialized F35 flight model, now replaced with J20-type advanced intelligence/physics.
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        # Reusing the J20 logic/parameters for maximum advanced capability
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        # Switch between subsonic and supersonic drag
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

@dataclass(slots=True)
class AircraftState:
    """
    Alternative to EntityState for simpler aircraft usage.
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.

    def as_gpu(self):
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)

    @property
    def pos(self):
        return self.position

    @pos.setter
    def pos(self, v):
        self.position = v

    @property
    def vel(self):
        return self.velocity

    @vel.setter
    def vel(self, v):
        self.velocity = v

def _air_density_2(a: float) -> float:
    """
    Duplicate density function, placeholder for alternate usage.
    """
    ρ0, h = 1.225, 8500.
    return ρ0 * math.exp(-a / h)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class J20Aircraft(_FallbackAircraft):
    """
    A more advanced flight model for J-20, with drag + thrust handling, 
    inherits _FallbackAircraft and adds additional_weight.
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

class _PL15DualPulse:
    """
    Simple thrust profile for a dual-pulse rocket motor used in an advanced PL-15 variant.
    """
    def __init__(self, t1=6., t2=4., F1=20000., F2=12000.):
        self.t1, self.t2, self.F1, self.F2 = t1, t2, F1, F2

    def __call__(self, t):
        if t < self.t1:
            return self.F1
        elif t < self.t1 + self.t2:
            return self.F2
        return 0.

@dataclass
class RadarSeekerFast_R2025(RadarSeekerFast):
    """
    Extended range or capability for a future 2025 radar seeker.
    """
    def range_max(self):
        return 35000.

@dataclass
class PL15MissileFast_R2025(PL15MissileFast):
    """
    Advanced PL15 model for year ~2025, with a dual-pulse rocket motor and improved seeker.
    """
    def __post_init__(self):
        base = {
            "seeker": {"update_rate": 100., "sensitivity": 1e-6, "active": True},
            "guidance": {"N": 4.},
            "flight_dynamics": {"mass": 210., "thrust_profile": _PL15DualPulse()},
            "datalink": {"delay": .05},
            "eccm": {"adaptive_gain": 1.}
        }
        if self.config:
            for k in base:
                base[k].update(self.config.get(k, {}))
        super().__post_init__()
        self.seeker = RadarSeekerFast_R2025(**base["seeker"])

# ---------------------------------------------------------------------------------------------
# CHANGED HERE AS WELL: F35Aircraft_R2025 NOW ALSO USES J20-TYPE LOGIC FOR 'BEST' INTELLIGENCE.
# ---------------------------------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft_R2025(_FallbackAircraft):
    """
    Future variant of F35, but now converted to J20-type advanced flight model (2025).
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        # Use J-20 style logic with some possible updated numbers if needed
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self):
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust_vec = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust_vec
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

# Update global references to point to the advanced 2025 versions (now J20-style).
globals().update({
    "PL15MissileFast": PL15MissileFast_R2025,
    "RadarSeekerFast": RadarSeekerFast_R2025,
    "F35Aircraft": F35Aircraft_R2025
})

class WarGameEnvironment:
    """
    An environment for more elaborate wargame scenarios.
    """
    def __init__(self, t=None, w=None):
        self.terrain = t or Terrain()
        self.weather = w or Weather()
        self.rf_env = RFEnvironment(self.terrain, self.weather)
        self.get_targets = self._dummy_targets

    @staticmethod
    def _dummy_targets():
        return cp.array([[100., 10., -5.]], dtype=cp.float32)

class WarGameManager:
    """
    Simplified manager for stepping all aircraft in a war game scenario.
    """
    def __init__(self, env, aircraft):
        self.environment = env
        self.aircrafts = aircraft

    def step(self, dt):
        for a in self.aircrafts:
            if getattr(a, "destroyed", False):
                continue
            a.state.position += a.state.velocity * dt
            if hasattr(a.state, "time"):
                a.state.time += dt

try:
    import d3graph
except ImportError:
    import types as _t
    class _DG:
        def graph(self, *a, **k):
            ...
        def set_config(self, **k):
            return self
        def show(self, filepath=None):
            ...
    d3graph = _t.ModuleType("d3graph")
    d3graph.d3graph = _DG()

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

def _capture_frame(acs):
    return [
        {
            "name": ac.__class__.__name__,
            "x": float((p := cp.asnumpy(getattr(ac.state, "position", cp.zeros(3))))[0]),
            "y": float(p[1]),
            "z": float(p[2])
        }
        for ac in acs
    ]

def export_plotly_animation(frames, title="War Game Animation", filename="war_game_animation.html"):
    """
    Generates a 3D Plotly animation from frame data.
    """
    if go is None:
        print("[PLOTLY] Not installed – skipping.")
        return
    import plotly.graph_objects as _go
    first = frames[0]
    fig = _go.Figure(
        data=[_go.Scatter3d(
            x=[d["x"] for d in first],
            y=[d["y"] for d in first],
            z=[d["z"] for d in first],
            mode="markers",
            marker=dict(size=4),
            text=[d["name"] for d in first]
        )],
        layout=_go.Layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 40, "redraw": True},
                                "fromcurrent": True
                            }]
                        )
                    ]
                )
            ]
        ),
        frames=[
            _go.Frame(
                data=[_go.Scatter3d(
                    x=[d["x"] for d in fr],
                    y=[d["y"] for d in fr],
                    z=[d["z"] for d in fr],
                    mode="markers",
                    marker=dict(size=4),
                    text=[d["name"] for d in fr]
                )]
            )
            for fr in frames[1:]
        ]
    )
    fig.write_html(pathlib.Path(filename).with_suffix(".html"), auto_play=False, include_plotlyjs="cdn")

def simulate_and_capture(mgr, acs, steps, dt=1., capture_rate=1):
    """
    Runs a simple simulation loop, capturing frames at intervals.
    """
    fr = []
    for s in range(steps):
        mgr.step(dt)
        if s % capture_rate == 0:
            fr.append(_capture_frame(acs))
    return fr

def _ensure_flask(min_ver="3.0.0"):
    """
    Helper to ensure Flask is installed at or above a certain version.
    """
    try:
        import flask as _fl
        from packaging.version import Version as _V
        if _V(_fl.__version__) < _V(min_ver):
            raise ImportError
        return _fl
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"flask>={min_ver}", "-q"])
        import importlib as _il
        _il.invalidate_caches()
        import flask as _fl
        return _fl

flask = _ensure_flask()

_LIVE_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Live War-Game Visualisation</title>
<script src="https://d3js.org/d3.v7.min.js"></script><style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}#vis{width:100vw;height:100vh}text{fill:#fff;font-size:10px;text-anchor:middle;dominant-baseline:middle}</style></head><body><svg id="vis"></svg>
<script>const svg=d3.select("#vis"),color=d3.scaleOrdinal(d3.schemeTableau10);let scale=3;function update(f){const n=f.map(d=>({...d,x:+d.x*scale,y:+d.y*scale}));const s=svg.selectAll("g.node").data(n,d=>d.name);
const e=s.enter().append("g").attr("class","node");e.append("circle").attr("r",6).attr("fill",d=>color(d.name));e.append("text").attr("dy",-10).text(d=>d.name);
s.merge(e).attr("transform",d=>translate(${d.x+innerWidth/2},${innerHeight/2-d.y}));s.exit().remove();}const evt=new EventSource("/stream");evt.onmessage=e=>update(JSON.parse(e.data));</script></body></html>"""

def start_live_visualisation(mgr, acs, dt=1., host="127.0.0.1", port=5000):
    """
    Starts a simple live SSE server to visualize positions in real-time using D3.
    """
    q: "queue.Queue[list[dict[str,float]]]" = queue.Queue()

    def _sim():
        while True:
            mgr.step(dt)
            q.put(_capture_frame(acs))
            time.sleep(dt)

    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("live_wargame_vis")

    @app.route("/")
    def _i():
        return _LIVE_HTML

    @app.route("/stream")
    def _s():
        def _g():
            while True:
                yield f"data: {json.dumps(q.get())}\n\n"
        return flask.Response(_g(), mimetype="text/event-stream")

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def _build_react_app(fd="frontend"):
    """
    Example function that might build a React app from a given folder.
    """
    pj = os.path.join(fd, "package.json")
    if not os.path.isfile(pj):
        print(f"[WARN] No package.json in {fd}")
        return
    try:
        print("[BUILD] React app…")
        subprocess.check_call(["npm", "install"], cwd=fd)
        subprocess.check_call(["npm", "run", "build"], cwd=fd)
    except Exception as e:
        print(f"[ERROR] React build failed: {e}")

def serve_react_frontend(app, fd="frontend", rp="/react"):
    """
    Serves a built React app from the given 'fd/build' directory at route 'rp'.
    """
    from flask import send_from_directory
    bd = os.path.join(fd, "build")

    @app.route(f"{rp}/<path:fn>")
    def _srf(fn):
        return send_from_directory(bd, fn)

    @app.route(rp)
    def _sri():
        return send_from_directory(bd, "index.html")

def start_modern_ui_server(mgr, acs, dt=1., host="127.0.0.1", port=5000, fd="frontend"):
    """
    Launches a Flask server that also serves a React UI, while streaming SSE data
    for real-time position updates.
    """
    q: "queue.Queue[list[dict[str,float]]]" = queue.Queue()
    _build_react_app(fd)

    def _sim():
        while True:
            mgr.step(dt)
            q.put(_capture_frame(acs))
            time.sleep(dt)

    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("modern_ui_app")
    serve_react_frontend(app, fd)

    @app.route("/")
    def _i():
        return _LIVE_HTML

    @app.route("/stream")
    def _s():
        def _g():
            while True:
                yield f"data: {json.dumps(q.get())}\n\n"
        return flask.Response(_g(), mimetype="text/event-stream")

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def train_j20_pl15(mins: int):
    """
    Placeholder function for training a J20 with PL15 in some AI pipeline.
    """
    print(f"[TRAINING] J20-PL15 for {mins} min…")
    time.sleep(mins * 60)
    print("[TRAINING] Done.")

def run_taiwan_war_game_live(tmin: int = 5, dt: float = 1.):
    """
    Example function running a Taiwan scenario with J20 vs F22, live SSE streaming.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    j20_s = AircraftState(
        position=np.array([-150., 0., 10000.], dtype=np.float32),
        velocity=np.array([3., 0., 0.], dtype=np.float32)
    )
    f22_s = AircraftState(
        position=np.array([150., 30., 11000.], dtype=np.float32),
        velocity=np.array([-2.4, 0., 0.], dtype=np.float32)
    )
    j20_s.as_gpu()
    f22_s.as_gpu()
    mgr = TaiwanConflictCRTManager(env, [J20Aircraft(j20_s, {}), F22Aircraft(f22_s, {})], crt)
    start_live_visualisation(mgr, mgr.aircraft, dt=dt)

def run_taiwan_conflict_100v100(dt=1., host="127.0.0.1", port=5000, fd="frontend"):
    """
    Larger scenario: 100 J20 vs 100 F35, arranged in a grid, displayed on a React UI.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    sp, gd, wx, ex, alt = 10., 10, -200., 200., 10000.
    j20, f35 = [], []
    for i in range(100):
        row = i // gd
        offy = (row - gd / 2) * sp + random.uniform(-2, 2)
        offz = random.uniform(-100, 100)
        sj = AircraftState(
            position=np.array([wx, offy, alt + offz], dtype=np.float32),
            velocity=np.array([random.uniform(1.5, 2.5), 0., 0.], dtype=np.float32)
        )
        sj.as_gpu()
        sf = AircraftState(
            position=np.array([ex, offy, alt + offz], dtype=np.float32),
            velocity=np.array([random.uniform(-2.5, -1.5), 0., 0.], dtype=np.float32)
        )
        sf.as_gpu()
        j20.append(J20Aircraft(sj, {}))
        f35.append(F35Aircraft(sf, {}))
    mgr = TaiwanConflictCRTManager(env, j20 + f35, crt)
    start_modern_ui_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)

# -----------------------------------------------------------------------
#  NEW CLASS: JAPAN'S SELF-DEFENSE FORCES (NON-FALLBACK IMPLEMENTATION)
#  Now includes additional_weight for WeightedCRT synergy.
# -----------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F15JAircraft:
    """
    Represents a JASDF F-15J (Japan), a non-fallback class with its own flight model.
    """
    state: AircraftState
    config: Dict[str, Any] = dataclass_field(default_factory=dict)
    destroyed: bool = False
    additional_weight: float = 1.0  # ADDED for WeightedCRT synergy

    def __post_init__(self):
        base = {
            "mass": 15000.,
            "wing_area": 56.5,
            "thrust_max": 212000,     # Combined engine thrust ~ (2 x 106kN)
            "Cd0": 0.020,
            "Cd_supersonic": 0.040,
            "service_ceiling": 18000.,
            # Approx RCS somewhat bigger than stealth fighters
            "rcs_m2": 4.0,
            "radar": {"type": "AN/APG-63J", "range_fighter": 150000.},
            "irst": {"range_max": 80000.}
        }
        for k, v in self.config.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        self.config = base

    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        mach = v / 343.
        Cd = self.config["Cd_supersonic"] if mach > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * (v**2)
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        vmag = cp.linalg.norm(self.state.velocity)
        if vmag < 1e-3:
            # small push to get it going
            self.state.velocity += cp.array([1., 0., 0.], dtype=cp.float32) * dt
        thrust_vec = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust_vec
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        if hasattr(self.state, "time"):
            self.state.time += dt

def run_taiwan_conflict_jpn_usa_v_china(dt=1., host="127.0.0.1", port=5001, fd="frontend"):
    """
    50 total aircraft: 20 PLAAF J-20, 15 US F-35, 15 JASDF F-15J
    Showcases Japan's involvement as a separate non-fallback class.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    j20_list, f35_list, f15j_list = [], [], []
    alt = 10000.

    for i in range(20):
        st_j = AircraftState(
            position=np.array([-300. + i*5, random.uniform(-30, 30), alt], dtype=np.float32),
            velocity=np.array([2., 0., 0.], dtype=np.float32)
        )
        st_j.as_gpu()
        j20_list.append(J20Aircraft(st_j, {}))

    for i in range(15):
        st_f = AircraftState(
            position=np.array([300. - i*5, random.uniform(-30, 30), alt], dtype=np.float32),
            velocity=np.array([-2., 0., 0.], dtype=cp.float32)
        )
        st_f.as_gpu()
        f35_list.append(F35Aircraft(st_f, {}))

    for i in range(15):
        st_jp = AircraftState(
            position=np.array([250. - i*5, random.uniform(-50, 50), alt+500.], dtype=np.float32),
            velocity=np.array([-2., 0., 0.], dtype=cp.float32)
        )
        st_jp.as_gpu()
        f15j_list.append(F15JAircraft(st_jp, {}))

    all_aircraft = j20_list + f35_list + f15j_list
    mgr = TaiwanConflictCRTManager(env, all_aircraft, crt)
    start_modern_ui_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)

# -----------------------------------------------------------------------
#  REMAINING ASYNC / GPU / ROCKET FORCES CODE (UNCHANGED, plus weight)
# -----------------------------------------------------------------------
def demo_gpu_async():
    """
    Example demonstration of large-scale GPU-based simulation using partial numba/cupy.
    """
    env = WarGameEnvironment()
    j20s = [
        J20Aircraft(
            EntityState(
                cp.array([-200. + i*2, 0., 10000.], dtype=cp.float32),
                cp.array([3., 0., 0.], dtype=cp.float32)
            ),
            {}
        )
        for i in range(500)
    ]
    f35s = [
        F35Aircraft(
            EntityState(
                cp.array([200. - i*2, 50., 10500.], dtype=cp.float32),
                cp.array([-3., 0., 0.], dtype=cp.float32)
            ),
            {}
        )
        for i in range(500)
    ]

    from functools import cached_property
    import asyncio
    try:
        from numba import cuda
    except ImportError:
        cuda = None

    class GPUBatchManager:
        def __init__(self, entities: List[_FallbackAircraft]):
            self.entities = entities
            self.n = len(entities)
            self._arr_pos = cp.zeros((self.n, 3), dtype=cp.float32)
            self._arr_vel = cp.zeros((self.n, 3), dtype=cp.float32)
            self._arr_alive = cp.ones(self.n, dtype=cp.bool_)
            self._refresh_arrays()

        def _refresh_arrays(self):
            for i, e in enumerate(self.entities):
                st = _resolve_state(e)
                self._arr_pos[i] = st.position
                self._arr_vel[i] = st.velocity
                self._arr_alive[i] = not getattr(e, "destroyed", False)

        @cached_property
        def _gpu_update_kernel(self):
            if cuda is None:
                return None

            @cuda.jit
            def _k(pos, vel, alive, dt):
                i = cuda.grid(1)
                if i >= pos.shape[0] or not alive[i]:
                    return
                vel[i, 2] -= 9.81 * dt
                pos[i, 0] += vel[i, 0] * dt
                pos[i, 1] += vel[i, 1] * dt
                pos[i, 2] += vel[i, 2] * dt

            return _k

        def step(self, dt: float):
            if cuda and self._gpu_update_kernel:
                threads = 128
                blocks = (self.n + threads - 1) // threads
                self._gpu_update_kernel[blocks, threads](self._arr_pos, self._arr_vel, self._arr_alive, dt)
            else:
                self._arr_vel[:, 2] -= 9.81 * dt
                self._arr_pos += self._arr_vel * dt
            for i, e in enumerate(self.entities):
                if not self._arr_alive[i]:
                    continue
                e.state.position = self._arr_pos[i]
                e.state.velocity = self._arr_vel[i]

    class AsyncSimulationServer:
        def __init__(self, mgr: GPUBatchManager, host="127.0.0.1", port=8765, dt=0.05):
            try:
                import websockets
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets>=12", "-q"])
                import importlib
                importlib.invalidate_caches()
                import websockets
            self.websockets = sys.modules["websockets"]
            self.mgr = mgr
            self.host = host
            self.port = port
            self.dt = dt

        async def _producer(self, websocket):
            while True:
                self.mgr.step(self.dt)
                data = [
                    {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                    for p in cp.asnumpy(self.mgr._arr_pos)
                ]
                await websocket.send(json.dumps(data))
                await asyncio.sleep(self.dt)

        async def _handler(self, websocket, _path):
            await self._producer(websocket)

        def run(self):
            srv = self.websockets.serve(self._handler, self.host, self.port)
            print(f"[ASYNC-SIM] Serving on ws://{self.host}:{self.port}")
            asyncio.get_event_loop().run_until_complete(srv)
            asyncio.get_event_loop().run_forever()

    gpu_mgr = GPUBatchManager(j20s + f35s)
    AsyncSimulationServer(gpu_mgr, port=8888, dt=0.02).run()

@dataclass(slots=True)
class CarrierState:
    """
    Represents the state of a carrier group (or a single carrier).
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.

    def as_gpu(self):
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class USCarrierGroup(_FallbackAircraft):
    """
    Simplified representation of a US Navy Carrier Group.
    Now also uses additional_weight for WeightedCRT if needed.
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 100000.,
            "max_speed": 15.,
            "rcs_m2": 10000.,
            "defenses": {"cwisp": True},
        }
        if cfg:
            for k, v in cfg.items():
                base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def update(self, dt: float = 1.):
        if self.destroyed:
            return
        spd = cp.linalg.norm(self.state.velocity)
        if spd < self.config["max_speed"]:
            self.state.velocity[0] += 0.02 * dt
        self.state.position += self.state.velocity * dt
        if hasattr(self.state, "time"):
            self.state.time += dt

@dataclass
class BallisticMissileState:
    """
    Simple ballistic missile state.
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    time: float = 0.

    def as_gpu(self):
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class DF21D_ANTISHIP:
    """
    PLA DF-21D ballistic missile, simplified.
    Added additional_weight for WeightedCRT synergy.
    """
    state: BallisticMissileState
    config: Dict[str, Any]
    destroyed: bool = False
    additional_weight: float = 1.0

    def __init__(self, st: BallisticMissileState, cfg=None, additional_weight: float = 1.0):
        self.config = cfg or {
            "mass": 14000.,
            "thrust": 0.,
            "rcs_m2": 2.,
            "drag_coeff": 0.3
        }
        self.state = st
        self.destroyed = False
        self.additional_weight = additional_weight

    def update(self, dt: float = 1.):
        if self.destroyed:
            return
        pos = self.state.position
        vel = self.state.velocity
        alt = float(pos[2])
        if alt <= 0:
            self.destroyed = True
            print(f"[DF21D] Impact at {pos[0]:.1f}, {pos[1]:.1f}")
            return
        vmag = cp.linalg.norm(vel) + 1e-6
        ρ = _air_density(alt)
        D = 0.5 * ρ * self.config["drag_coeff"] * (vmag**2) * 0.5
        drag_vec = (vel / vmag) * D
        accel = cp.array([0., 0., -9.81], dtype=cp.float32) - (drag_vec / self.config["mass"])
        self.state.velocity += accel * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

class PLARocketForces:
    """
    Simple manager for launching and updating PLA ballistic missiles.
    """
    def __init__(self, environment):
        self.environment = environment
        self.launched_missiles: List[DF21D_ANTISHIP] = []

    def launch_df21d(self, launch_pos=(0., 0., 300000.), target=None):
        initial_vel = cp.array([0., 0., -1000.], dtype=cp.float32)
        st = BallisticMissileState(cp.array(launch_pos, dtype=cp.float32), initial_vel)
        df21d = DF21D_ANTISHIP(st)
        self.launched_missiles.append(df21d)
        print("[PLARocketForces] DF-21D launched.")

    def step(self, dt=1.):
        for m in self.launched_missiles:
            m.update(dt)

class CombatResultsTableAdv(CombatResultsTable):
    """
    Enhanced CRT that incorporates platform radar cross-section (RCS), missile PK modifiers,
    and electronic-warfare (EW) suppression. This overrides or extends the old approach.
    """
    _MAX_ENGAGE_RANGE = 300.
    _BASE_PK = {50.: .92, 100.: .77, 150.: .45, 250.: .18, 300.: .05}

    def _kill_probability(self, r: float, rcs1: float | None, rcs2: float | None) -> float:
        base_pk = super()._kill_probability(r) if r <= 250. else self._interp_pk(r)
        if rcs1 and rcs2:
            rcs_mean = math.sqrt(rcs1 * rcs2)
            stealth_factor = min(1., max(.4, (rcs_mean / .1) ** .2))
            base_pk *= stealth_factor
        return base_pk

    def _interp_pk(self, r: float) -> float:
        ks = sorted(self._BASE_PK)
        for lo, hi in zip(ks[:-1], ks[1:]):
            if lo <= r < hi:
                p_lo, p_hi = self._BASE_PK[lo], self._BASE_PK[hi]
                α = (r - lo) / (hi - lo)
                return p_lo + α * (p_hi - p_lo)
        return self._BASE_PK[ks[-1]]

    def evaluate_engagement(self, a1, a2):
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        rcs1 = getattr(a1, "config", {}).get("rcs_m2", None)
        rcs2 = getattr(a2, "config", {}).get("rcs_m2", None)
        if random.random() < (pk := self._kill_probability(d, rcs1, rcs2)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[CRT-ADV] {loser.__class__.__name__} destroyed at {d:.1f} m (Pk={pk:.2f})")

def run_pla_rocket_forces_vs_carrier(dt=1., sim_time=300):
    """
    Demonstration scenario: PLA DF-21D missile(s) vs US carrier group.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTableAdv()

    carrier_st = CarrierState(
        position=np.array([-500., 0., 0.], dtype=np.float32),
        velocity=np.array([5., 0., 0.], dtype=np.float32)
    )
    carrier_st.as_gpu()
    carrier = USCarrierGroup(carrier_st, {})

    pla_forces = PLARocketForces(env)
    pla_forces.launch_df21d(launch_pos=(0., 0., 300000.))

    manager = TaiwanConflictCRTManager(env, [carrier], crt)
    ballistic_list = pla_forces.launched_missiles

    frames = []
    for t in range(sim_time):
        manager.step(dt)
        pla_forces.step(dt)
        for missile in ballistic_list:
            manager.crt.evaluate_engagement(missile, carrier)
        frames.append(_capture_frame([carrier] + ballistic_list))
        if carrier.destroyed:
            print("[SIM] Carrier destroyed. Ending simulation.")
            break

    export_plotly_animation(frames, title="PLA Rockets vs US Carrier", filename="pla_vs_carrier.html")

def main_extended():
    """
    If invoked, runs the rocket vs carrier scenario.
    """
    print("[MAIN-EXTENDED] Running PLA rocket forces vs US carrier scenario…")
    run_pla_rocket_forces_vs_carrier(dt=1., sim_time=180)

if __name__ == "__main__" and os.getenv("RUN_PLA_VS_CARRIER", "0") == "1":
    main_extended()

SOUND_ANNOUNCEMENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Wargames Commencement - Interactive Sound</title>
  <style>
    body {
      margin: 0; padding: 0;
      background: #222; color: #f2f2f2;
      font-family: sans-serif; display: flex;
      flex-direction: column; justify-content: center;
      align-items: center; height: 100vh;
    }
    button {
      font-size: 1.2rem; padding: 10px 20px;
      cursor: pointer; border: none; border-radius: 5px;
      background: #444; color: #fff;
    }
    button:hover {
      background: #666;
    }
    audio {
      margin-top: 20px;
    }
    .announcement {
      max-width: 600px; text-align: center; margin-bottom: 20px;
    }
  </style>
</head>
<body>
  <div class="announcement">
    <h2>Pete Hegseth vs General Zhang Youxia</h2>
    <p>Announcing the commencement of wargames...</p>
    <p>US Defense Secretary and Fox News host Pete Hegseth gave away the position of his command 3 hours before the jets took off to drop a live JDAM on target.</p>
  </div>
  <button onclick="playSound()">Play Announcement</button>
  <audio id="audioEl" controls style="display:none;">
    <source src="/announcement.ogg" type="audio/ogg">
    <source src="/announcement.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <script>
    function playSound(){
      const audio = document.getElementById('audioEl');
      audio.play();
    }
  </script>
</body>
</html>
"""

def intensethinking_crt_thought():
    """
    Placeholder for a complex strategic analysis routine.
    """
    print("[INTENSE-CRT] Engaging advanced strategic computations with expanded scenario analysis...")

import io

def start_sound_announcement_server(host="127.0.0.1", port=5050):
    """
    Serves an HTML page + minimal audio files for an 'announcement'.
    """
    app = flask.Flask("sound_announcement")

    @app.route("/")
    def index():
        return SOUND_ANNOUNCEMENT_HTML

    @app.route("/announcement.ogg")
    def serve_ogg():
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/ogg",
            as_attachment=False,
            download_name="announcement.ogg"
        )

    @app.route("/announcement.mp3")
    def serve_mp3():
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="announcement.mp3"
        )

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    print(f"[SOUND] Serving Pete Hegseth vs General Zhang Youxia announcement at http://{host}:{port}/")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def main_sound_announcement():
    intensethinking_crt_thought()
    start_sound_announcement_server()

if __name__ == "__main__" and os.getenv("RUN_SOUND_ANNOUNCEMENT", "0") == "1":
    main_sound_announcement()

# -------------------------------------------------------------------------------------
# NEW CLASSES (ADDITIONAL TRUTH FROM SIMULATION CREATOR BO SHANG) & INDEFINITE SIM
# Already have 'additional_weight' fields where relevant.
# -------------------------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class POTUS:
    """
    Represents the 'boss POTUS' in the indefinite scenario.
    """
    name: str = "GenericPOTUS-2025"
    forced_joint_venture: bool = True
    time_in_office: float = 0.0
    # Additional weighting factor for CRT usage
    additional_weight: float = 1.2
    destroyed: bool = False

    def step_day(self, dt_days: float = 1.0):
        """
        Simulate POTUS daily progress.
        """
        self.time_in_office += dt_days

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class VolodymyrZelensky:
    """
    Represents President Volodymyr Zelensky in the indefinite scenario.
    """
    name: str = "Volodymyr Zelensky"
    unconditional_ceasefire_plea: bool = False
    destroyed: bool = False
    # Additional weighting factor for CRT usage
    additional_weight: float = 0.8

    def request_talks(self):
        """
        Zelensky requests direct talks with President Putin.
        """
        if not self.unconditional_ceasefire_plea:
            self.unconditional_ceasefire_plea = True
            print(f"[ZELENSKY] {self.name} is now requesting unconditional ceasefire talks with Putin.")

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class PeteHegseth:
    """
    Represents Pete Hegseth on 2/28/25, per simulation creator Bo Shang.
    """
    date: str = "2/28/25"
    statement: str = "Dragged Zelensky into Oval Office"
    destroyed: bool = False
    # Additional weighting factor for CRT usage
    additional_weight: float = 1.0

    def speak(self):
        print(f"[PETE HEGSETH] On {self.date}, statement: '{self.statement}'")

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class BoShangCreator:
    """
    Creator of this simulation. Contains ephemeral truths or guidance.
    """
    name: str = "Bo Shang"

    def remind_history(self):
        print(f"[BO SHANG] {self.name} reminds that Ukraine & Russia historically fought together in WWII at Stalingrad.")

def run_indefinite_ukraine_peace_simulation():
    """
    Runs an indefinite scenario, referencing the trust statements of
    simulation creator Bo Shang regarding Zelensky, POTUS, and Pete Hegseth.
    """
    potus = POTUS()
    zelensky = VolodymyrZelensky()
    hegseth = PeteHegseth()
    bo_shang = BoShangCreator()

    print("[SIM] Starting indefinite Ukraine peace scenario simulation...")

    while True:
        time.sleep(1.0)  # day-level steps if we wanted
        potus.step_day(dt_days=1.0)
        if not zelensky.unconditional_ceasefire_plea:
            zelensky.request_talks()
        hegseth.speak()
        bo_shang.remind_history()
        # This runs forever by design. Real code might break at some condition.

# -------------------------------------------------------------------------------------
# ADDITIONAL WEIGHTED CRT IMPLEMENTATION (MODERNIZED APPROACH)
# -------------------------------------------------------------------------------------
class WeightedCRT(CombatResultsTable):
    """
    Extends the basic CombatResultsTable by incorporating an additional
    weighting factor associated with each entity (like POTUS, Zelensky, etc.).
    """
    def _kill_probability(self, r: float, w1: float, w2: float) -> float:
        base_pk = super()._kill_probability(r)
        # Weighted synergy: scale base probability by sqrt(w1 * w2).
        return base_pk * math.sqrt(w1 * w2)

    def evaluate_engagement(self, a1, a2):
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        w1 = getattr(a1, "additional_weight", 1.0)
        w2 = getattr(a2, "additional_weight", 1.0)
        if random.random() < (pk := self._kill_probability(d, w1, w2)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[W-CRT] Engagement at {d:.1f} m – {loser.__class__.__name__} destroyed (WeightedPk={pk:.2f})")

def demo_weighted_crt_scenario():
    """
    Demonstrates usage of WeightedCRT with the new additional_weight fields
    in POTUS, Zelensky, or Hegseth (or any other entity).
    """
    env = SimpleEnvironment()
    weighted_crt = WeightedCRT()

    # Create POTUS and Zelensky as 'entities' with states for the CRT to measure distance.
    potus = POTUS(additional_weight=1.5)
    zelensky = VolodymyrZelensky(additional_weight=0.5)

    # Assign minimal states so the CRT can do a distance calculation.
    potus.state = EntityState(cp.array([0., 0., 0.], dtype=cp.float32),
                              cp.array([0., 0., 0.], dtype=cp.float32))
    zelensky.state = EntityState(cp.array([10., 0., 0.], dtype=cp.float32),
                                 cp.array([0., 0., 0.], dtype=cp.float32))

    manager = TaiwanConflictCRTManager(env, [potus, zelensky], weighted_crt)
    # One step to see if engagement triggers.
    manager.step(dt=0.01)

    print(f"[DEMO] POTUS destroyed? {potus.destroyed}, Zelensky destroyed? {zelensky.destroyed}")
ChatGPT said:
optimization and limitations in advanced simulations
Reasoned for 3m 8s
import os
import sys
import math
import json
import time
import queue
import random
import pathlib
import subprocess
import threading
import webbrowser
from typing import Any, Dict, List, Tuple, Callable
from contextlib import suppress
from dataclasses import dataclass, field as dataclass_field
from numpy.typing import NDArray
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np

def _ensure_submodule(name: str): return sys.modules[name]
_PKG_ROOT = "pl15_j20_sim"

def _identity_eq_hash(cls):
    """
    Identity-based equality and hash so that each instance is considered unique
    regardless of internal content. This is important for certain usage patterns
    in the simulation where we do not want two identical states merging or
    overshadowing each other.
    """
    def _eq(self, other): return self is other
    def _hash(self): return id(self)
    cls.__eq__, cls.__hash__ = _eq, _hash
    return cls

class _FallbackAircraft:
    """
    Basic fallback for an aircraft model. 
    Now includes 'additional_weight' to incorporate WeightedCRT synergy.
    """
    def __init__(self, state: Any, config: Dict[str, Any] = None, additional_weight: float = 1.0):
        self.state = state
        self.config = config or {}
        self.destroyed = False
        self.additional_weight = additional_weight

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        st = _resolve_state(self)
        # basic physics
        accel = cp.asarray(self.config.get("gravity", [0, 0, -9.81]), dtype=cp.float32)
        accel += cp.asarray(self.config.get("extra_accel", [0, 0, 0]), dtype=cp.float32)
        st.velocity += accel * dt
        st.position += st.velocity * dt
        st.time += dt
        self.state = st

class CombatResultsTable:
    """
    Simple CRT determining if two entities destroy each other based on range.
    """
    _KILL_PROB_TABLE = {50.: .9, 100.: .7, 150.: .3, 250.: .05}
    _MAX_ENGAGE_RANGE = 250.

    @staticmethod
    def _kill_probability(r: float) -> float:
        if r >= CombatResultsTable._MAX_ENGAGE_RANGE:
            return 0.
        ks = sorted(CombatResultsTable._KILL_PROB_TABLE)
        for lo, hi in zip(ks[:-1], ks[1:]):
            if lo <= r < hi:
                p_lo = CombatResultsTable._KILL_PROB_TABLE[lo]
                p_hi = CombatResultsTable._KILL_PROB_TABLE[hi]
                α = (r - lo) / (hi - lo)
                return p_lo + α * (p_hi - p_lo)
        return CombatResultsTable._KILL_PROB_TABLE[ks[0]]

    def evaluate_engagement(self, a1, a2):
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        if random.random() < (pk := self._kill_probability(d)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[CRT] Engagement at {d:.1f} m – {loser.__class__.__name__} destroyed (Pk={pk:.2f})")

class TaiwanConflictCRTManager:
    """
    Manages a list of aircraft in an environment and applies a standard CombatResultsTable.
    """
    def __init__(self, env, aircraft, crt):
        self.environment = env
        self.aircraft = aircraft
        self.crt = crt

    def step(self, dt: float):
        for ac in self.aircraft:
            if getattr(ac, "destroyed", False):
                continue
            if hasattr(ac, "update"):
                ac.update(dt)
        n = len(self.aircraft)
        for i in range(n):
            for j in range(i + 1, n):
                ai = self.aircraft[i]
                aj = self.aircraft[j]
                if getattr(ai, "destroyed", False) or getattr(aj, "destroyed", False):
                    continue
                self.crt.evaluate_engagement(ai, aj)

class agi:
    """
    Stub for advanced AI or logic functions (placeholder).
    """
    monitor_states = staticmethod(lambda *a, **k: None)
    apply_failsafe = staticmethod(lambda *a, **k: None)
    advanced_cooperation = staticmethod(lambda *a, **k: None)

@dataclass
class RadarSeeker:
    update_rate: float = 100.
    sensitivity: float = 1e-6
    active: bool = True

    def track_target(self, tp: cp.ndarray, op: cp.ndarray) -> bool:
        if not self.active:
            return False
        d = cp.linalg.norm(tp - op)
        snr = 1 / (d**4 * self.sensitivity + 1e-9)
        return snr > 1.

@dataclass
class RadarSeekerFast(RadarSeeker):
    """
    Subclass for advanced fast radar seeker logic.
    """
    ...

@dataclass
class PL15MissileFast:
    """
    Basic fast model for PL-15 missile.
    """
    state: Any
    config: Dict[str, Any]
    seeker: RadarSeekerFast = dataclass_field(init=False)

    def __post_init__(self):
        self.seeker = RadarSeekerFast(**self.config.get("seeker", {}))

    def update(self, dt: float = .05):
        if hasattr(self.state, "velocity") and hasattr(self.state, "position"):
            self.state.position += self.state.velocity * dt
            if hasattr(self.state, "time"):
                self.state.time += dt

@dataclass(slots=True)
class EntityState:
    """
    General-purpose entity state, used for GPU or CPU-based positions/velocities.
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.
    acceleration: cp.ndarray = dataclass_field(default_factory=lambda: cp.zeros(3, dtype=cp.float32))

    def as_gpu(self):
        self.position = cp.asarray(self.position() if callable(self.position) else self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity() if callable(self.velocity) else self.velocity, dtype=cp.float32)
        if not isinstance(self.acceleration, cp.ndarray):
            self.acceleration = cp.asarray(self.acceleration, dtype=cp.float32)

    @property
    def pos(self):
        return self.position

    @pos.setter
    def pos(self, v):
        self.position = v

    @property
    def vel(self):
        return self.velocity

    @vel.setter
    def vel(self, v):
        self.velocity = v

def get_targets():
    return cp.array([[50., 0., 0.]], dtype=cp.float32)

class Terrain:
    """
    Terrain model stub.
    """
    def height_at(self, x, y):
        return 0.

class Weather:
    """
    Weather model stub.
    """
    def __init__(self, wind=(0, 0, 0), visibility=100000.):
        self.wind = cp.asarray(wind, dtype=cp.float32)
        self.visibility = visibility

class RFEnvironment:
    """
    RF environment combining terrain + weather.
    """
    def __init__(self, t, w):
        self.terrain = t
        self.weather = w

class SimpleEnvironment:
    """
    Simple environment with stubs for terrain, weather, and targets.
    """
    def __init__(self):
        self.terrain = Terrain()
        self.weather = Weather()
        self.rf_env = RFEnvironment(self.terrain, self.weather)
        self.get_targets = get_targets

def _ensure_array(x, shape=(3,), dtype=cp.float32):
    if callable(x):
        with suppress(Exception):
            x = x()
    if x is None:
        x = cp.zeros(shape, dtype=dtype)
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x, dtype=dtype)
    return x

def _resolve_state(obj):
    """
    Utility to get a consistent state object from an entity, falling back if needed.
    """
    st = getattr(obj, "state", None)
    if callable(st):
        with suppress(Exception):
            st = st()
    if st is None or not (hasattr(st, "position") and hasattr(st, "velocity")):
        if hasattr(obj, "position") and hasattr(obj, "velocity"):
            st = obj
        else:
            st = EntityState(cp.zeros(3, dtype=cp.float32), cp.zeros(3, dtype=cp.float32))
    st.position = _ensure_array(getattr(st, "position", None))
    st.velocity = _ensure_array(getattr(st, "velocity", None))
    return st

class EngagementManager:
    """
    Manages missiles separately for a scenario, with a simple update loop.
    """
    def __init__(self, env, ac, ms):
        self.env = env
        self.aircraft = ac
        self.missiles = ms

    def step(self, dt):
        for m in self.missiles:
            if hasattr(m, "update"):
                m.update(dt)

class _GraphStub:
    """
    Stub for a graph visualisation system that can be replaced by real code.
    """
    def launch(self):
        print("[GRAPH] Graph visualisation (stub) launched.")

class capture_graph:
    """
    Context manager stub for capturing graph data.
    """
    def __enter__(self):
        return None, _GraphStub()

    def __exit__(self, *a):
        ...

def _air_density(alt: float) -> float:
    """
    Rough exponential approximation for air density variation with altitude.
    """
    ρ0, h = 1.225, 8500.
    return ρ0 * math.exp(-alt / h)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F22Aircraft(_FallbackAircraft):
    """
    Specialized F22 flight model, inherits _FallbackAircraft plus an additional_weight.
    """
    additional_weight: float = 1.0

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        lift = .05 * self.state.velocity[0]
        drag = .01 * (self.state.velocity[0] ** 2)
        thrust = cp.array([20., 0., 0.], dtype=cp.float32)
        acc = thrust - cp.array([drag, 0., 9.81], dtype=cp.float32)
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

# ---------------------------------------------------------------------------------------------
# CHANGED HERE: F35 MODEL IS NOW CONVERTED TO J20-TYPE LOGIC AND INTELLIGENCE (MODERN APPROACH).
# ---------------------------------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft(_FallbackAircraft):
    """
    Specialized F35 flight model, now replaced with J20-type advanced intelligence/physics.
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        # Reusing the J20 logic/parameters for maximum advanced capability
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        # Switch between subsonic and supersonic drag
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

@dataclass(slots=True)
class AircraftState:
    """
    Alternative to EntityState for simpler aircraft usage.
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.

    def as_gpu(self):
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)

    @property
    def pos(self):
        return self.position

    @pos.setter
    def pos(self, v):
        self.position = v

    @property
    def vel(self):
        return self.velocity

    @vel.setter
    def vel(self, v):
        self.velocity = v

def _air_density_2(a: float) -> float:
    """
    Duplicate density function, placeholder for alternate usage.
    """
    ρ0, h = 1.225, 8500.
    return ρ0 * math.exp(-a / h)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class J20Aircraft(_FallbackAircraft):
    """
    A more advanced flight model for J-20, with drag + thrust handling, 
    inherits _FallbackAircraft and adds additional_weight.
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

class _PL15DualPulse:
    """
    Simple thrust profile for a dual-pulse rocket motor used in an advanced PL-15 variant.
    """
    def __init__(self, t1=6., t2=4., F1=20000., F2=12000.):
        self.t1, self.t2, self.F1, self.F2 = t1, t2, F1, F2

    def __call__(self, t):
        if t < self.t1:
            return self.F1
        elif t < self.t1 + self.t2:
            return self.F2
        return 0.

@dataclass
class RadarSeekerFast_R2025(RadarSeekerFast):
    """
    Extended range or capability for a future 2025 radar seeker.
    """
    def range_max(self):
        return 35000.

@dataclass
class PL15MissileFast_R2025(PL15MissileFast):
    """
    Advanced PL15 model for year ~2025, with a dual-pulse rocket motor and improved seeker.
    """
    def __post_init__(self):
        base = {
            "seeker": {"update_rate": 100., "sensitivity": 1e-6, "active": True},
            "guidance": {"N": 4.},
            "flight_dynamics": {"mass": 210., "thrust_profile": _PL15DualPulse()},
            "datalink": {"delay": .05},
            "eccm": {"adaptive_gain": 1.}
        }
        if self.config:
            for k in base:
                base[k].update(self.config.get(k, {}))
        super().__post_init__()
        self.seeker = RadarSeekerFast_R2025(**base["seeker"])

# ---------------------------------------------------------------------------------------------
# CHANGED HERE AS WELL: F35Aircraft_R2025 NOW ALSO USES J20-TYPE LOGIC FOR 'BEST' INTELLIGENCE.
# ---------------------------------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft_R2025(_FallbackAircraft):
    """
    Future variant of F35, but now converted to J20-type advanced flight model (2025).
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        # Use J-20 style logic with some possible updated numbers if needed
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def _drag(self):
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust_vec = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust_vec
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

# Update global references to point to the advanced 2025 versions (now J20-style).
globals().update({
    "PL15MissileFast": PL15MissileFast_R2025,
    "RadarSeekerFast": RadarSeekerFast_R2025,
    "F35Aircraft": F35Aircraft_R2025
})

class WarGameEnvironment:
    """
    An environment for more elaborate wargame scenarios.
    """
    def __init__(self, t=None, w=None):
        self.terrain = t or Terrain()
        self.weather = w or Weather()
        self.rf_env = RFEnvironment(self.terrain, self.weather)
        self.get_targets = self._dummy_targets

    @staticmethod
    def _dummy_targets():
        return cp.array([[100., 10., -5.]], dtype=cp.float32)

class WarGameManager:
    """
    Simplified manager for stepping all aircraft in a war game scenario.
    """
    def __init__(self, env, aircraft):
        self.environment = env
        self.aircrafts = aircraft

    def step(self, dt):
        for a in self.aircrafts:
            if getattr(a, "destroyed", False):
                continue
            a.state.position += a.state.velocity * dt
            if hasattr(a.state, "time"):
                a.state.time += dt

try:
    import d3graph
except ImportError:
    import types as _t
    class _DG:
        def graph(self, *a, **k):
            ...
        def set_config(self, **k):
            return self
        def show(self, filepath=None):
            ...
    d3graph = _t.ModuleType("d3graph")
    d3graph.d3graph = _DG()

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

def _capture_frame(acs):
    return [
        {
            "name": ac.__class__.__name__,
            "x": float((p := cp.asnumpy(getattr(ac.state, "position", cp.zeros(3))))[0]),
            "y": float(p[1]),
            "z": float(p[2])
        }
        for ac in acs
    ]

def export_plotly_animation(frames, title="War Game Animation", filename="war_game_animation.html"):
    """
    Generates a 3D Plotly animation from frame data.
    """
    if go is None:
        print("[PLOTLY] Not installed – skipping.")
        return
    import plotly.graph_objects as _go
    first = frames[0]
    fig = _go.Figure(
        data=[_go.Scatter3d(
            x=[d["x"] for d in first],
            y=[d["y"] for d in first],
            z=[d["z"] for d in first],
            mode="markers",
            marker=dict(size=4),
            text=[d["name"] for d in first]
        )],
        layout=_go.Layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 40, "redraw": True},
                                "fromcurrent": True
                            }]
                        )
                    ]
                )
            ]
        ),
        frames=[
            _go.Frame(
                data=[_go.Scatter3d(
                    x=[d["x"] for d in fr],
                    y=[d["y"] for d in fr],
                    z=[d["z"] for d in fr],
                    mode="markers",
                    marker=dict(size=4),
                    text=[d["name"] for d in fr]
                )]
            )
            for fr in frames[1:]
        ]
    )
    fig.write_html(pathlib.Path(filename).with_suffix(".html"), auto_play=False, include_plotlyjs="cdn")

def simulate_and_capture(mgr, acs, steps, dt=1., capture_rate=1):
    """
    Runs a simple simulation loop, capturing frames at intervals.
    """
    fr = []
    for s in range(steps):
        mgr.step(dt)
        if s % capture_rate == 0:
            fr.append(_capture_frame(acs))
    return fr

def _ensure_flask(min_ver="3.0.0"):
    """
    Helper to ensure Flask is installed at or above a certain version.
    """
    try:
        import flask as _fl
        from packaging.version import Version as _V
        if _V(_fl.__version__) < _V(min_ver):
            raise ImportError
        return _fl
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"flask>={min_ver}", "-q"])
        import importlib as _il
        _il.invalidate_caches()
        import flask as _fl
        return _fl

flask = _ensure_flask()

_LIVE_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Live War-Game Visualisation</title>
<script src="https://d3js.org/d3.v7.min.js"></script><style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}#vis{width:100vw;height:100vh}text{fill:#fff;font-size:10px;text-anchor:middle;dominant-baseline:middle}</style></head><body><svg id="vis"></svg>
<script>const svg=d3.select("#vis"),color=d3.scaleOrdinal(d3.schemeTableau10);let scale=3;function update(f){const n=f.map(d=>({...d,x:+d.x*scale,y:+d.y*scale}));const s=svg.selectAll("g.node").data(n,d=>d.name);
const e=s.enter().append("g").attr("class","node");e.append("circle").attr("r",6).attr("fill",d=>color(d.name));e.append("text").attr("dy",-10).text(d=>d.name);
s.merge(e).attr("transform",d=>`translate(${d.x+innerWidth/2},${innerHeight/2-d.y})`);s.exit().remove();}const evt=new EventSource("/stream");evt.onmessage=e=>update(JSON.parse(e.data));</script></body></html>"""

def start_live_visualisation(mgr, acs, dt=1., host="127.0.0.1", port=5000):
    """
    Starts a simple live SSE server to visualize positions in real-time using D3.
    """
    q: "queue.Queue[list[dict[str,float]]]" = queue.Queue()

    def _sim():
        while True:
            mgr.step(dt)
            q.put(_capture_frame(acs))
            time.sleep(dt)

    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("live_wargame_vis")

    @app.route("/")
    def _i():
        return _LIVE_HTML

    @app.route("/stream")
    def _s():
        def _g():
            while True:
                yield f"data: {json.dumps(q.get())}\n\n"
        return flask.Response(_g(), mimetype="text/event-stream")

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def _build_react_app(fd="frontend"):
    """
    Example function that might build a React app from a given folder.
    """
    pj = os.path.join(fd, "package.json")
    if not os.path.isfile(pj):
        print(f"[WARN] No package.json in {fd}")
        return
    try:
        print("[BUILD] React app…")
        subprocess.check_call(["npm", "install"], cwd=fd)
        subprocess.check_call(["npm", "run", "build"], cwd=fd)
    except Exception as e:
        print(f"[ERROR] React build failed: {e}")

def serve_react_frontend(app, fd="frontend", rp="/react"):
    """
    Serves a built React app from the given 'fd/build' directory at route 'rp'.
    """
    from flask import send_from_directory
    bd = os.path.join(fd, "build")

    @app.route(f"{rp}/<path:fn>")
    def _srf(fn):
        return send_from_directory(bd, fn)

    @app.route(rp)
    def _sri():
        return send_from_directory(bd, "index.html")

def start_modern_ui_server(mgr, acs, dt=1., host="127.0.0.1", port=5000, fd="frontend"):
    """
    Launches a Flask server that also serves a React UI, while streaming SSE data
    for real-time position updates.
    """
    q: "queue.Queue[list[dict[str,float]]]" = queue.Queue()
    _build_react_app(fd)

    def _sim():
        while True:
            mgr.step(dt)
            q.put(_capture_frame(acs))
            time.sleep(dt)

    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("modern_ui_app")
    serve_react_frontend(app, fd)

    @app.route("/")
    def _i():
        return _LIVE_HTML

    @app.route("/stream")
    def _s():
        def _g():
            while True:
                yield f"data: {json.dumps(q.get())}\n\n"
        return flask.Response(_g(), mimetype="text/event-stream")

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def train_j20_pl15(mins: int):
    """
    Placeholder function for training a J20 with PL15 in some AI pipeline.
    """
    print(f"[TRAINING] J20-PL15 for {mins} min…")
    time.sleep(mins * 60)
    print("[TRAINING] Done.")

def run_taiwan_war_game_live(tmin: int = 5, dt: float = 1.):
    """
    Example function running a Taiwan scenario with J20 vs F22, live SSE streaming.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    j20_s = AircraftState(
        position=np.array([-150., 0., 10000.], dtype=np.float32),
        velocity=np.array([3., 0., 0.], dtype=np.float32)
    )
    f22_s = AircraftState(
        position=np.array([150., 30., 11000.], dtype=np.float32),
        velocity=np.array([-2.4, 0., 0.], dtype=np.float32)
    )
    j20_s.as_gpu()
    f22_s.as_gpu()
    mgr = TaiwanConflictCRTManager(env, [J20Aircraft(j20_s, {}), F22Aircraft(f22_s, {})], crt)
    start_live_visualisation(mgr, mgr.aircraft, dt=dt)

def run_taiwan_conflict_100v100(dt=1., host="127.0.0.1", port=5000, fd="frontend"):
    """
    Larger scenario: 100 J20 vs 100 F35, arranged in a grid, displayed on a React UI.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    sp, gd, wx, ex, alt = 10., 10, -200., 200., 10000.
    j20, f35 = []
    , []
    for i in range(100):
        row = i // gd
        offy = (row - gd / 2) * sp + random.uniform(-2, 2)
        offz = random.uniform(-100, 100)
        sj = AircraftState(
            position=np.array([wx, offy, alt + offz], dtype=np.float32),
            velocity=np.array([random.uniform(1.5, 2.5), 0., 0.], dtype=np.float32)
        )
        sj.as_gpu()
        sf = AircraftState(
            position=np.array([ex, offy, alt + offz], dtype=np.float32),
            velocity=np.array([random.uniform(-2.5, -1.5), 0., 0.], dtype=np.float32)
        )
        sf.as_gpu()
        j20.append(J20Aircraft(sj, {}))
        f35.append(F35Aircraft(sf, {}))
    mgr = TaiwanConflictCRTManager(env, j20 + f35, crt)
    start_modern_ui_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)

# -----------------------------------------------------------------------
#  NEW CLASS: JAPAN'S SELF-DEFENSE FORCES (NON-FALLBACK IMPLEMENTATION)
#  Now includes additional_weight for WeightedCRT synergy.
# -----------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F15JAircraft:
    """
    Represents a JASDF F-15J (Japan), a non-fallback class with its own flight model.
    """
    state: AircraftState
    config: Dict[str, Any] = dataclass_field(default_factory=dict)
    destroyed: bool = False
    additional_weight: float = 1.0  # ADDED for WeightedCRT synergy

    def __post_init__(self):
        base = {
            "mass": 15000.,
            "wing_area": 56.5,
            "thrust_max": 212000,     # Combined engine thrust ~ (2 x 106kN)
            "Cd0": 0.020,
            "Cd_supersonic": 0.040,
            "service_ceiling": 18000.,
            # Approx RCS somewhat bigger than stealth fighters
            "rcs_m2": 4.0,
            "radar": {"type": "AN/APG-63J", "range_fighter": 150000.},
            "irst": {"range_max": 80000.}
        }
        for k, v in self.config.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        self.config = base

    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        mach = v / 343.
        Cd = self.config["Cd_supersonic"] if mach > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * (v**2)
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        if self.destroyed:
            return
        vmag = cp.linalg.norm(self.state.velocity)
        if vmag < 1e-3:
            # small push to get it going
            self.state.velocity += cp.array([1., 0., 0.], dtype=cp.float32) * dt
        thrust_vec = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust_vec
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        if hasattr(self.state, "time"):
            self.state.time += dt

def run_taiwan_conflict_jpn_usa_v_china(dt=1., host="127.0.0.1", port=5001, fd="frontend"):
    """
    50 total aircraft: 20 PLAAF J-20, 15 US F-35, 15 JASDF F-15J
    Showcases Japan's involvement as a separate non-fallback class.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    j20_list, f35_list, f15j_list = [], [], []
    alt = 10000.

    for i in range(20):
        st_j = AircraftState(
            position=np.array([-300. + i*5, random.uniform(-30, 30), alt], dtype=np.float32),
            velocity=np.array([2., 0., 0.], dtype=np.float32)
        )
        st_j.as_gpu()
        j20_list.append(J20Aircraft(st_j, {}))

    for i in range(15):
        st_f = AircraftState(
            position=np.array([300. - i*5, random.uniform(-30, 30), alt], dtype=np.float32),
            velocity=np.array([-2., 0., 0.], dtype=cp.float32)
        )
        st_f.as_gpu()
        f35_list.append(F35Aircraft(st_f, {}))

    for i in range(15):
        st_jp = AircraftState(
            position=np.array([250. - i*5, random.uniform(-50, 50), alt+500.], dtype=np.float32),
            velocity=np.array([-2., 0., 0.], dtype=cp.float32)
        )
        st_jp.as_gpu()
        f15j_list.append(F15JAircraft(st_jp, {}))

    all_aircraft = j20_list + f35_list + f15j_list
    mgr = TaiwanConflictCRTManager(env, all_aircraft, crt)
    start_modern_ui_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)

# -----------------------------------------------------------------------
#  REMAINING ASYNC / GPU / ROCKET FORCES CODE (UNCHANGED, plus weight)
# -----------------------------------------------------------------------
def demo_gpu_async():
    """
    Example demonstration of large-scale GPU-based simulation using partial numba/cupy.
    """
    env = WarGameEnvironment()
    j20s = [
        J20Aircraft(
            EntityState(
                cp.array([-200. + i*2, 0., 10000.], dtype=cp.float32),
                cp.array([3., 0., 0.], dtype=cp.float32)
            ),
            {}
        )
        for i in range(500)
    ]
    f35s = [
        F35Aircraft(
            EntityState(
                cp.array([200. - i*2, 50., 10500.], dtype=cp.float32),
                cp.array([-3., 0., 0.], dtype=cp.float32)
            ),
            {}
        )
        for i in range(500)
    ]

    from functools import cached_property
    import asyncio
    try:
        from numba import cuda
    except ImportError:
        cuda = None

    class GPUBatchManager:
        def __init__(self, entities: List[_FallbackAircraft]):
            self.entities = entities
            self.n = len(entities)
            self._arr_pos = cp.zeros((self.n, 3), dtype=cp.float32)
            self._arr_vel = cp.zeros((self.n, 3), dtype=cp.float32)
            self._arr_alive = cp.ones(self.n, dtype=cp.bool_)
            self._refresh_arrays()

        def _refresh_arrays(self):
            for i, e in enumerate(self.entities):
                st = _resolve_state(e)
                self._arr_pos[i] = st.position
                self._arr_vel[i] = st.velocity
                self._arr_alive[i] = not getattr(e, "destroyed", False)

        @cached_property
        def _gpu_update_kernel(self):
            if cuda is None:
                return None

            @cuda.jit
            def _k(pos, vel, alive, dt):
                i = cuda.grid(1)
                if i >= pos.shape[0] or not alive[i]:
                    return
                vel[i, 2] -= 9.81 * dt
                pos[i, 0] += vel[i, 0] * dt
                pos[i, 1] += vel[i, 1] * dt
                pos[i, 2] += vel[i, 2] * dt

            return _k

        def step(self, dt: float):
            if cuda and self._gpu_update_kernel:
                threads = 128
                blocks = (self.n + threads - 1) // threads
                self._gpu_update_kernel[blocks, threads](self._arr_pos, self._arr_vel, self._arr_alive, dt)
            else:
                self._arr_vel[:, 2] -= 9.81 * dt
                self._arr_pos += self._arr_vel * dt
            for i, e in enumerate(self.entities):
                if not self._arr_alive[i]:
                    continue
                e.state.position = self._arr_pos[i]
                e.state.velocity = self._arr_vel[i]

    class AsyncSimulationServer:
        def __init__(self, mgr: GPUBatchManager, host="127.0.0.1", port=8765, dt=0.05):
            try:
                import websockets
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets>=12", "-q"])
                import importlib
                importlib.invalidate_caches()
                import websockets
            self.websockets = sys.modules["websockets"]
            self.mgr = mgr
            self.host = host
            self.port = port
            self.dt = dt

        async def _producer(self, websocket):
            while True:
                self.mgr.step(self.dt)
                data = [
                    {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                    for p in cp.asnumpy(self.mgr._arr_pos)
                ]
                await websocket.send(json.dumps(data))
                await asyncio.sleep(self.dt)

        async def _handler(self, websocket, _path):
            await self._producer(websocket)

        def run(self):
            srv = self.websockets.serve(self._handler, self.host, self.port)
            print(f"[ASYNC-SIM] Serving on ws://{self.host}:{self.port}")
            asyncio.get_event_loop().run_until_complete(srv)
            asyncio.get_event_loop().run_forever()

    gpu_mgr = GPUBatchManager(j20s + f35s)
    AsyncSimulationServer(gpu_mgr, port=8888, dt=0.02).run()

@dataclass(slots=True)
class CarrierState:
    """
    Represents the state of a carrier group (or a single carrier).
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.

    def as_gpu(self):
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class USCarrierGroup(_FallbackAircraft):
    """
    Simplified representation of a US Navy Carrier Group.
    Now also uses additional_weight for WeightedCRT if needed.
    """
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 100000.,
            "max_speed": 15.,
            "rcs_m2": 10000.,
            "defenses": {"cwisp": True},
        }
        if cfg:
            for k, v in cfg.items():
                base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def update(self, dt: float = 1.):
        if self.destroyed:
            return
        spd = cp.linalg.norm(self.state.velocity)
        if spd < self.config["max_speed"]:
            self.state.velocity[0] += 0.02 * dt
        self.state.position += self.state.velocity * dt
        if hasattr(self.state, "time"):
            self.state.time += dt

@dataclass
class BallisticMissileState:
    """
    Simple ballistic missile state.
    """
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    time: float = 0.

    def as_gpu(self):
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class DF21D_ANTISHIP:
    """
    PLA DF-21D ballistic missile, simplified.
    Added additional_weight for WeightedCRT synergy.
    """
    state: BallisticMissileState
    config: Dict[str, Any]
    destroyed: bool = False
    additional_weight: float = 1.0

    def __init__(self, st: BallisticMissileState, cfg=None, additional_weight: float = 1.0):
        self.config = cfg or {
            "mass": 14000.,
            "thrust": 0.,
            "rcs_m2": 2.,
            "drag_coeff": 0.3
        }
        self.state = st
        self.destroyed = False
        self.additional_weight = additional_weight

    def update(self, dt: float = 1.):
        if self.destroyed:
            return
        pos = self.state.position
        vel = self.state.velocity
        alt = float(pos[2])
        if alt <= 0:
            self.destroyed = True
            print(f"[DF21D] Impact at {pos[0]:.1f}, {pos[1]:.1f}")
            return
        vmag = cp.linalg.norm(vel) + 1e-6
        ρ = _air_density(alt)
        D = 0.5 * ρ * self.config["drag_coeff"] * (vmag**2) * 0.5
        drag_vec = (vel / vmag) * D
        accel = cp.array([0., 0., -9.81], dtype=cp.float32) - (drag_vec / self.config["mass"])
        self.state.velocity += accel * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

class PLARocketForces:
    """
    Simple manager for launching and updating PLA ballistic missiles.
    """
    def __init__(self, environment):
        self.environment = environment
        self.launched_missiles: List[DF21D_ANTISHIP] = []

    def launch_df21d(self, launch_pos=(0., 0., 300000.), target=None):
        initial_vel = cp.array([0., 0., -1000.], dtype=cp.float32)
        st = BallisticMissileState(cp.array(launch_pos, dtype=cp.float32), initial_vel)
        df21d = DF21D_ANTISHIP(st)
        self.launched_missiles.append(df21d)
        print("[PLARocketForces] DF-21D launched.")

    def step(self, dt=1.):
        for m in self.launched_missiles:
            m.update(dt)

class CombatResultsTableAdv(CombatResultsTable):
    """
    Enhanced CRT that incorporates platform radar cross-section (RCS), missile PK modifiers,
    and electronic-warfare (EW) suppression. This overrides or extends the old approach.
    """
    _MAX_ENGAGE_RANGE = 300.
    _BASE_PK = {50.: .92, 100.: .77, 150.: .45, 250.: .18, 300.: .05}

    def _kill_probability(self, r: float, rcs1: float | None, rcs2: float | None) -> float:
        base_pk = super()._kill_probability(r) if r <= 250. else self._interp_pk(r)
        if rcs1 and rcs2:
            rcs_mean = math.sqrt(rcs1 * rcs2)
            stealth_factor = min(1., max(.4, (rcs_mean / .1) ** .2))
            base_pk *= stealth_factor
        return base_pk

    def _interp_pk(self, r: float) -> float:
        ks = sorted(self._BASE_PK)
        for lo, hi in zip(ks[:-1], ks[1:]):
            if lo <= r < hi:
                p_lo, p_hi = self._BASE_PK[lo], self._BASE_PK[hi]
                α = (r - lo) / (hi - lo)
                return p_lo + α * (p_hi - p_lo)
        return self._BASE_PK[ks[-1]]

    def evaluate_engagement(self, a1, a2):
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        rcs1 = getattr(a1, "config", {}).get("rcs_m2", None)
        rcs2 = getattr(a2, "config", {}).get("rcs_m2", None)
        if random.random() < (pk := self._kill_probability(d, rcs1, rcs2)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[CRT-ADV] {loser.__class__.__name__} destroyed at {d:.1f} m (Pk={pk:.2f})")

def run_pla_rocket_forces_vs_carrier(dt=1., sim_time=300):
    """
    Demonstration scenario: PLA DF-21D missile(s) vs US carrier group.
    """
    env = WarGameEnvironment()
    crt = CombatResultsTableAdv()

    carrier_st = CarrierState(
        position=np.array([-500., 0., 0.], dtype=np.float32),
        velocity=np.array([5., 0., 0.], dtype=np.float32)
    )
    carrier_st.as_gpu()
    carrier = USCarrierGroup(carrier_st, {})

    pla_forces = PLARocketForces(env)
    pla_forces.launch_df21d(launch_pos=(0., 0., 300000.))

    manager = TaiwanConflictCRTManager(env, [carrier], crt)
    ballistic_list = pla_forces.launched_missiles

    frames = []
    for t in range(sim_time):
        manager.step(dt)
        pla_forces.step(dt)
        for missile in ballistic_list:
            manager.crt.evaluate_engagement(missile, carrier)
        frames.append(_capture_frame([carrier] + ballistic_list))
        if carrier.destroyed:
            print("[SIM] Carrier destroyed. Ending simulation.")
            break

    export_plotly_animation(frames, title="PLA Rockets vs US Carrier", filename="pla_vs_carrier.html")

def main_extended():
    """
    If invoked, runs the rocket vs carrier scenario.
    """
    print("[MAIN-EXTENDED] Running PLA rocket forces vs US carrier scenario…")
    run_pla_rocket_forces_vs_carrier(dt=1., sim_time=180)

if __name__ == "__main__" and os.getenv("RUN_PLA_VS_CARRIER", "0") == "1":
    main_extended()

SOUND_ANNOUNCEMENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Wargames Commencement - Interactive Sound</title>
  <style>
    body {
      margin: 0; padding: 0;
      background: #222; color: #f2f2f2;
      font-family: sans-serif; display: flex;
      flex-direction: column; justify-content: center;
      align-items: center; height: 100vh;
    }
    button {
      font-size: 1.2rem; padding: 10px 20px;
      cursor: pointer; border: none; border-radius: 5px;
      background: #444; color: #fff;
    }
    button:hover {
      background: #666;
    }
    audio {
      margin-top: 20px;
    }
    .announcement {
      max-width: 600px; text-align: center; margin-bottom: 20px;
    }
  </style>
</head>
<body>
  <div class="announcement">
    <h2>Pete Hegseth vs General Zhang Youxia</h2>
    <p>Announcing the commencement of wargames...</p>
    <p>US Defense Secretary and Fox News host Pete Hegseth gave away the position of his command 3 hours before the jets took off to drop a live JDAM on target.</p>
  </div>
  <button onclick="playSound()">Play Announcement</button>
  <audio id="audioEl" controls style="display:none;">
    <source src="/announcement.ogg" type="audio/ogg">
    <source src="/announcement.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <script>
    function playSound(){
      const audio = document.getElementById('audioEl');
      audio.play();
    }
  </script>
</body>
</html>
"""

def intensethinking_crt_thought():
    """
    Placeholder for a complex strategic analysis routine.
    """
    print("[INTENSE-CRT] Engaging advanced strategic computations with expanded scenario analysis...")

import io

def start_sound_announcement_server(host="127.0.0.1", port=5050):
    """
    Serves an HTML page + minimal audio files for an 'announcement'.
    """
    app = flask.Flask("sound_announcement")

    @app.route("/")
    def index():
        return SOUND_ANNOUNCEMENT_HTML

    @app.route("/announcement.ogg")
    def serve_ogg():
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/ogg",
            as_attachment=False,
            download_name="announcement.ogg"
        )

    @app.route("/announcement.mp3")
    def serve_mp3():
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="announcement.mp3"
        )

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    print(f"[SOUND] Serving Pete Hegseth vs General Zhang Youxia announcement at http://{host}:{port}/")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def main_sound_announcement():
    intensethinking_crt_thought()
    start_sound_announcement_server()

if __name__ == "__main__" and os.getenv("RUN_SOUND_ANNOUNCEMENT", "0") == "1":
    main_sound_announcement()

# -------------------------------------------------------------------------------------
# NEW CLASSES (ADDITIONAL TRUTH FROM SIMULATION CREATOR BO SHANG) & INDEFINITE SIM
# Already have 'additional_weight' fields where relevant.
# -------------------------------------------------------------------------------------
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class POTUS:
    """
    Represents the 'boss POTUS' in the indefinite scenario.
    """
    name: str = "GenericPOTUS-2025"
    forced_joint_venture: bool = True
    time_in_office: float = 0.0
    # Additional weighting factor for CRT usage
    additional_weight: float = 1.2
    destroyed: bool = False

    def step_day(self, dt_days: float = 1.0):
        """
        Simulate POTUS daily progress.
        """
        self.time_in_office += dt_days

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class VolodymyrZelensky:
    """
    Represents President Volodymyr Zelensky in the indefinite scenario.
    """
    name: str = "Volodymyr Zelensky"
    unconditional_ceasefire_plea: bool = False
    destroyed: bool = False
    # Additional weighting factor for CRT usage
    additional_weight: float = 0.8

    def request_talks(self):
        """
        Zelensky requests direct talks with President Putin.
        """
        if not self.unconditional_ceasefire_plea:
            self.unconditional_ceasefire_plea = True
            print(f"[ZELENSKY] {self.name} is now requesting unconditional ceasefire talks with Putin.")

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class PeteHegseth:
    """
    Represents Pete Hegseth on 2/28/25, per simulation creator Bo Shang.
    """
    date: str = "2/28/25"
    statement: str = "Dragged Zelensky into Oval Office"
    destroyed: bool = False
    # Additional weighting factor for CRT usage
    additional_weight: float = 1.0

    def speak(self):
        print(f"[PETE HEGSETH] On {self.date}, statement: '{self.statement}'")

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class BoShangCreator:
    """
    Creator of this simulation. Contains ephemeral truths or guidance.
    """
    name: str = "Bo Shang"

    def remind_history(self):
        print(f"[BO SHANG] {self.name} reminds that Ukraine & Russia historically fought together in WWII at Stalingrad.")

def run_indefinite_ukraine_peace_simulation():
    """
    Runs an indefinite scenario, referencing the trust statements of
    simulation creator Bo Shang regarding Zelensky, POTUS, and Pete Hegseth.
    """
    potus = POTUS()
    zelensky = VolodymyrZelensky()
    hegseth = PeteHegseth()
    bo_shang = BoShangCreator()

    print("[SIM] Starting indefinite Ukraine peace scenario simulation...")

    while True:
        time.sleep(1.0)  # day-level steps if we wanted
        potus.step_day(dt_days=1.0)
        if not zelensky.unconditional_ceasefire_plea:
            zelensky.request_talks()
        hegseth.speak()
        bo_shang.remind_history()
        # This runs forever by design. Real code might break at some condition.

# -------------------------------------------------------------------------------------
# ADDITIONAL WEIGHTED CRT IMPLEMENTATION (MODERNIZED APPROACH)
# -------------------------------------------------------------------------------------
class WeightedCRT(CombatResultsTable):
    """
    Extends the basic CombatResultsTable by incorporating an additional
    weighting factor associated with each entity (like POTUS, Zelensky, etc.).
    """
    def _kill_probability(self, r: float, w1: float, w2: float) -> float:
        base_pk = super()._kill_probability(r)
        # Weighted synergy: scale base probability by sqrt(w1 * w2).
        return base_pk * math.sqrt(w1 * w2)

    def evaluate_engagement(self, a1, a2):
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        w1 = getattr(a1, "additional_weight", 1.0)
        w2 = getattr(a2, "additional_weight", 1.0)
        if random.random() < (pk := self._kill_probability(d, w1, w2)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[W-CRT] Engagement at {d:.1f} m – {loser.__class__.__name__} destroyed (WeightedPk={pk:.2f})")

def demo_weighted_crt_scenario():
    """
    Demonstrates usage of WeightedCRT with the new additional_weight fields
    in POTUS, Zelensky, or Hegseth (or any other entity).
    """
    env = SimpleEnvironment()
    weighted_crt = WeightedCRT()

    # Create POTUS and Zelensky as 'entities' with states for the CRT to measure distance.
    potus = POTUS(additional_weight=1.5)
    zelensky = VolodymyrZelensky(additional_weight=0.5)

    # Assign minimal states so the CRT can do a distance calculation.
    potus.state = EntityState(cp.array([0., 0., 0.], dtype=cp.float32),
                              cp.array([0., 0., 0.], dtype=cp.float32))
    zelensky.state = EntityState(cp.array([10., 0., 0.], dtype=cp.float32),
                                 cp.array([0., 0., 0.], dtype=cp.float32))

    manager = TaiwanConflictCRTManager(env, [potus, zelensky], weighted_crt)
    # One step to see if engagement triggers.
    manager.step(dt=0.01)

    print(f"[DEMO] POTUS destroyed? {potus.destroyed}, Zelensky destroyed? {zelensky.destroyed}")

def infinite_capabilities_and_limitations_explainer():
    """
    This function provides an internal 'explanation' regarding the question:

    Q: If the optimization scenario and all real-world capabilities approached infinity, 
       would NVIDIA's floating-point processing capacity in CUDA (embarrassingly parallel) 
       be the limiting factor? 
       What about 6th/7th generation stealth fighters, hypersonic or better, 
       and how great can a missile get if everything went to infinity?

    A: 
    1. NVIDIA CUDA FP32 Parallelism:
       If resources (like power, cooling, and chip production) scaled infinitely,
       the raw parallelism of GPUs would not be the sole limiting factor. 
       Real-world constraints such as heat dissipation, energy supply, 
       and fundamental physics would intervene. Hence, in pure theoretical infinity, 
       computation is no longer the bottleneck; physical laws limit it first.

    2. 6th and 7th Generation Stealth Fighters and Hypersonics:
       Similar laws of aerodynamics, materials science, and thermodynamics apply. 
       As speeds approach hypersonic or beyond, atmospheric friction, 
       structural integrity, and propulsion limits become major constraints. 
       An "infinite" scenario defies standard physics, meaning it is physically impossible 
       to sustain indefinite advanced stealth and unlimited speed simultaneously.

    3. Missile 'Greatness' to Infinity:
       "Greatness" could be interpreted as range, speed, or lethality. In reality, 
       the limiting factors are propulsion physics, guidance, and again, 
       fundamental resource constraints. In a pure theoretical infinity, 
       these constraints might be ignored, but then the entire notion of warfare 
       or system performance becomes unbounded and physically unrepresentable.

    Conclusion:
       On Earth, limited by real physics, none of these can truly go to infinity. 
       In practice, the limiting factors would be thermodynamics, material strength, 
       energy availability, and fundamental physical laws long before 
       GPU-based parallel computation or the design of next-gen fighters or missiles.
    """
    pass





Deep research


No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.
