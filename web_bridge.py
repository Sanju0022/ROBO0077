"""
web_bridge.py - WallBot Lab
Turns MDP results into JSON for the two browser components in web/ and loads their HTML.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from engine import MOVES, SHORT, level_names

WEB = Path(__file__).parent / "web"

MOVE_COLORS = {"Move-Forward": "#2EC4B6", "Slight-Right-Turn": "#FFB347",
               "Sharp-Right-Turn": "#FF6B6B", "Slight-Left-Turn": "#9D8DF1"}
ARC_COLORS = {"SD_front": "#5AA9E6", "SD_left": "#2EC4B6", "SD_right": "#FFB347", "SD_back": "#9D8DF1"}
SHORT_MOVES = {"Move-Forward": "Forward", "Slight-Right-Turn": "Slight R",
               "Sharp-Right-Turn": "Sharp R", "Slight-Left-Turn": "Slight L"}


def _f(x, nd=4):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def load_html(name, data):
    html = (WEB / name).read_text(encoding="utf-8")
    return html.replace("/*__DATA__*/null", json.dumps(data))


def simulator_data(mdp, sol, reward):
    S = len(mdp.states)
    return {
        "moves": MOVES, "move_colors": MOVE_COLORS, "arc_colors": ARC_COLORS, "short_moves": SHORT_MOVES,
        "used": mdp.used, "short": SHORT,
        "cuts": {d: [float(c) for c in mdp.cuts[d]] for d in mdp.used},
        "names": {d: level_names(mdp.cuts[d]) for d in mdp.used},
        "states": mdp.states,
        "levels": mdp.levels.tolist(),
        "best": {s: MOVES[int(sol["pi"][i])] for i, s in enumerate(mdp.states)},
        "logged": {s: MOVES[int(mdp.behaviour[i])] for i, s in enumerate(mdp.states)},
        "allowed": {s: [MOVES[a] for a in range(len(MOVES)) if mdp.seen[i, a]] for i, s in enumerate(mdp.states)},
        "q": {s: [_f(v, 3) for v in sol["Q"][i]] for i, s in enumerate(mdp.states)},
        "value": {s: _f(sol["V"][i], 3) for i, s in enumerate(mdp.states)},
        "reward": {k: float(v) for k, v in asdict(reward).items()},
        "n_states": S,
    }


def vi_player_data(mdp, sol, gamma, tolerance):
    trace, deltas = sol["trace"], sol["deltas"]
    n = len(trace) - 1
    if n <= 160:
        ks = list(range(n + 1))
    else:
        ks = sorted(set(range(0, 40)) | set(np.unique(np.round(np.geomspace(1, n, 120)).astype(int)).tolist()) | {n})
    frames = [{"k": int(k), "V": [round(float(v), 4) for v in trace[k]], "d": float(deltas[k - 1]) if k else None} for k in ks]
    S, A = mdp.R.shape
    P = [[([[int(j), round(float(mdp.P[s, a, j]), 4)] for j in np.nonzero(mdp.P[s, a])[0]] if mdp.seen[s, a] else [])
          for a in range(A)] for s in range(S)]
    allv = np.concatenate([np.asarray(trace[k]) for k in ks])
    order = np.argsort(-sol["V"]).tolist()
    conv = np.unique(np.linspace(1, n, min(n, 400)).astype(int)).tolist() if n else []
    return {
        "moves": MOVES, "move_colors": MOVE_COLORS, "short_moves": SHORT_MOVES,
        "states": mdp.states, "order": order, "frames": frames,
        "P": P, "R": [[round(float(r), 4) for r in row] for row in mdp.R],
        "seen": mdp.seen.astype(bool).tolist(), "gamma": float(gamma), "tol": float(tolerance),
        "sweeps": int(n), "converged": bool(sol["converged"]),
        "conv": [[int(k), float(deltas[k - 1])] for k in conv],
        "vmin": float(min(0.0, allv.min())), "vmax": float(max(0.0, allv.max())),
        "visits": [int(v) for v in mdp.visits], "final": [round(float(v), 4) for v in sol["V"]],
    }
