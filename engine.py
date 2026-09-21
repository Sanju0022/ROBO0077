"""
engine.py - WallBot Lab
=======================
The MDP behind the wall-following robot, with no Streamlit code in it.

    readings  ->  distances  ->  states  ->  P(s'|s,a), R(s,a)  ->  value iteration  ->  V*, pi*

Design choices in this version
    * States use fixed distance thresholds in metres (editable), or quantiles.
    * Transition probabilities are counts with Laplace (additive) smoothing.
    * Rewards are shaped: a smooth Gaussian bonus for holding the target wall
      distance, graded penalties that grow as obstacles get closer, and a small
      bonus for moving forward.
    * Value iteration keeps the Day 5 notebook update and adds a threshold;
      policy iteration is included to cross-check the answer.

Run on its own:  python engine.py   ->  writes optimal_value_function.csv
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

MOVES = ["Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn"]
US = [f"US{i}" for i in range(1, 25)]

# Sensor arcs that reproduce the dataset's official 4-sensor file
ARCS = {"SD_front": [11, 12, 13, 14, 15], "SD_left": [18, 19, 20],
        "SD_right": [5, 6, 7, 8, 9], "SD_back": [23, 24]}
DIST = list(ARCS)
SHORT = {"SD_front": "F", "SD_left": "L", "SD_right": "R", "SD_back": "B"}

# Mounting angle of every ultrasound sensor (dataset README), degrees
ANGLE = {1: 180, **{i: -180 + 15 * (i - 1) for i in range(2, 14)}, **{i: 15 * (i - 13) for i in range(14, 25)}}


def locate_dataset(start=None):
    """Find sensor_readings_24.csv next to the code, in data/, or in the working folder."""
    roots = [Path(start).parent if start else Path(__file__).parent, Path.cwd()]
    for root in roots:
        for p in (root / "data" / "sensor_readings_24.csv", root / "sensor_readings_24.csv"):
            if p.exists():
                return p
    return None


def load_readings(source):
    """Read the 24-sensor file (header or not) and add the four simplified distances."""
    raw = pd.read_csv(source, header=None).iloc[:, :25]
    try:
        float(raw.iloc[0, 0])
    except (TypeError, ValueError):
        raw = raw.iloc[1:]
    raw.columns = US + ["Class"]
    raw[US] = raw[US].apply(pd.to_numeric, errors="coerce")
    raw["Class"] = raw["Class"].astype(str).str.strip()
    df = raw.dropna()
    df = df[df["Class"].isin(MOVES)].reset_index(drop=True)
    for name, sensors in ARCS.items():
        df[name] = df[[f"US{i}" for i in sensors]].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

LEVELS = {1: ["Close", "Far"], 2: ["Close", "Medium", "Far"],
          3: ["Close", "Medium", "Far", "Very far"], 4: ["Touching", "Close", "Medium", "Far", "Very far"]}

DEFAULT_CUTS = {"SD_front": [0.7, 1.4], "SD_left": [0.55, 0.9], "SD_right": [0.8, 1.6], "SD_back": [0.8, 1.6]}


def quantile_cuts(df, used, n_levels):
    qs = np.linspace(0, 1, n_levels + 1)[1:-1]
    return {d: sorted(set(np.round(df[d].quantile(qs).values, 2).tolist())) for d in used}


def level_names(cuts):
    return LEVELS[len(cuts)]


def state_code(levels, used, cuts):
    """levels = list of level indices aligned with `used` -> 'F:Close L:Medium ...'"""
    return " ".join(f"{SHORT[d]}:{level_names(cuts[d])[i]}" for d, i in zip(used, levels))


def discretise(df, used, cuts):
    """Vectorised: add a 'State' column plus one level column per distance."""
    out = df.copy()
    lv = np.column_stack([np.digitize(out[d].values, cuts[d], right=False) for d in used])
    for j, d in enumerate(used):
        out[f"lvl_{d}"] = lv[:, j]
    out["State"] = [state_code(row, used, cuts) for row in lv.tolist()]
    return out, lv


# ---------------------------------------------------------------------------
# Rewards (shaped)
# ---------------------------------------------------------------------------

@dataclass
class RewardSpec:
    target_left: float = 0.60     # metres from the wall we want to hold
    tolerance: float = 0.20       # width of the Gaussian bonus
    follow_gain: float = 1.0
    safe_front: float = 0.80      # penalty grows once the front is closer than this
    safe_left: float = 0.45
    crash_gain: float = 4.0
    forward_bonus: float = 0.10   # bonus for Move-Forward
    sharp_cost: float = 0.15      # cost for Sharp-Right-Turn

    def reading_reward(self, front, left):
        follow = self.follow_gain * np.exp(-0.5 * ((left - self.target_left) / max(self.tolerance, 1e-3)) ** 2)
        risk_f = np.clip((self.safe_front - front) / max(self.safe_front, 1e-3), 0, 1)
        risk_l = np.clip((self.safe_left - left) / max(self.safe_left, 1e-3), 0, 1)
        return follow - self.crash_gain * (risk_f + risk_l)

    def action_shaping(self):
        return np.array([self.forward_bonus, 0.0, -self.sharp_cost, 0.0])


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------

@dataclass
class RobotMDP:
    states: list
    levels: np.ndarray              # (S, n_used) level index per state
    P: np.ndarray                   # (S, A, S)
    R: np.ndarray                   # (S, A)
    counts: np.ndarray              # (S, A, S) raw counts
    seen: np.ndarray                # (S, A) bool - action observed in state
    used: list
    cuts: dict
    extras: dict = field(default_factory=dict)

    @property
    def visits(self):
        return self.counts.sum(axis=(1, 2))

    @property
    def behaviour(self):
        return self.counts.sum(axis=2).argmax(axis=1)


def build_robot_mdp(df, used, cuts, reward: RewardSpec, smoothing=0.5):
    """Estimate P and R from consecutive log rows. Smoothing is spread over the next
    states that were ever reached from s, so impossible jumps stay impossible."""
    dfs, lv = discretise(df, used, cuts)
    states, inv = np.unique(dfs["State"].values, return_inverse=True)
    states = states.tolist()
    S, A = len(states), len(MOVES)
    lvl_of_state = np.zeros((S, len(used)), dtype=int)
    lvl_of_state[inv] = lv
    act = dfs["Class"].map({m: i for i, m in enumerate(MOVES)}).values
    r_next = reward.reading_reward(dfs["SD_front"].values, dfs["SD_left"].values)

    counts = np.zeros((S, A, S))
    rsum = np.zeros((S, A))
    s, a, s2 = inv[:-1], act[:-1], inv[1:]
    np.add.at(counts, (s, a, s2), 1)
    np.add.at(rsum, (s, a), r_next[1:])

    n_sa = counts.sum(axis=2)
    seen = n_sa > 0
    support = counts.sum(axis=1) > 0                 # next states ever reached from s
    P = np.zeros_like(counts)
    for i in range(S):
        sup = support[i]
        for j in range(A):
            if seen[i, j]:
                row = counts[i, j] + smoothing * sup
                P[i, j] = row / row.sum()
    R = np.where(seen, rsum / np.maximum(n_sa, 1), 0.0) + reward.action_shaping()[None, :]

    dead = ~seen.any(axis=1)                         # only seen on the very last row
    for i in np.where(dead)[0]:
        P[i, 0, i], R[i, 0], seen[i, 0] = 1.0, 0.0, True
    mdp = RobotMDP(states, lvl_of_state, P, R, counts, seen, list(used), dict(cuts))
    mdp.extras["data"] = dfs
    mdp.extras["r_next"] = r_next
    return mdp


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def value_iteration(P, R, gamma, tolerance=1e-6, max_iterations=20_000, seen=None, in_place=False):
    """Notebook update  Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)  with a stopping threshold.
    in_place=True uses each new value immediately (Gauss-Seidel), which usually needs fewer sweeps."""
    S, A = R.shape
    seen = np.ones((S, A), bool) if seen is None else seen
    V = np.zeros(S)
    trace, deltas = [V.copy()], []
    for _ in range(max_iterations):
        V_old = V.copy()
        V_new = V if in_place else np.zeros(S)
        for s in range(S):
            Q_sa = np.full(A, -np.inf)
            for a in range(A):
                if seen[s, a]:
                    Q_sa[a] = R[s][a] + gamma * np.dot(P[s][a], V)
            V_new[s] = np.max(Q_sa)
        V = V_new
        deltas.append(float(np.max(np.abs(V - V_old))))
        trace.append(V.copy())
        if deltas[-1] < tolerance:
            break
    Q = np.where(seen, R + gamma * np.einsum("sat,t->sa", P, V), -np.inf)
    return {"V": V, "Q": Q, "pi": Q.argmax(axis=1), "trace": trace, "deltas": deltas,
            "sweeps": len(deltas), "converged": deltas[-1] < tolerance}


def fast_value_iteration(P, R, gamma, seen, tolerance=1e-8, max_iterations=100_000):
    """Vectorised version used where speed matters (parameter tuning, error baselines)."""
    V = np.zeros(R.shape[0])
    for k in range(max_iterations):
        Q = np.where(seen, R + gamma * np.einsum("sat,t->sa", P, V), -np.inf)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tolerance:
            V = V_new
            break
        V = V_new
    Q = np.where(seen, R + gamma * np.einsum("sat,t->sa", P, V), -np.inf)
    return V, Q, Q.argmax(axis=1)


def policy_iteration(P, R, gamma, seen, max_rounds=200):
    """Howard's policy iteration: exact evaluation by a linear solve, then greedy improvement."""
    S = R.shape[0]
    pi = np.where(seen, R, -np.inf).argmax(axis=1)
    history = []
    for rnd in range(max_rounds):
        P_pi = P[np.arange(S), pi]
        R_pi = R[np.arange(S), pi]
        V = np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)
        Q = np.where(seen, R + gamma * np.einsum("sat,t->sa", P, V), -np.inf)
        new_pi = Q.argmax(axis=1)
        changed = int((new_pi != pi).sum())
        history.append({"Round": rnd + 1, "Policy changes": changed, "Mean V": float(V.mean())})
        # keep the old action when it is tied with the best one (avoids cycling)
        keep = np.isclose(Q[np.arange(S), pi], Q.max(axis=1))
        new_pi = np.where(keep, pi, new_pi)
        if (new_pi == pi).all():
            return V, Q, pi, history
        pi = new_pi
    return V, Q, pi, history


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------

def value_table(mdp: RobotMDP, sol):
    rows = []
    for i, s in enumerate(mdp.states):
        row = {"State": s}
        for d, lvl in zip(mdp.used, mdp.levels[i]):
            row[d] = level_names(mdp.cuts[d])[lvl]
        row["Optimal_Value"] = round(float(sol["V"][i]), 6)
        row["Optimal_Action"] = MOVES[int(sol["pi"][i])]
        for a, m in enumerate(MOVES):
            q = sol["Q"][i, a]
            row[f"Q({m})"] = round(float(q), 6) if np.isfinite(q) else np.nan
        row["Logged_Most_Common"] = MOVES[int(mdp.behaviour[i])]
        row["Visits"] = int(mdp.visits[i])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Optimal_Value", ascending=False).reset_index(drop=True)


def solve(df, used=DIST, cuts=None, gamma=0.9, tolerance=1e-6, reward=None, smoothing=0.5, in_place=False):
    cuts = cuts or {d: DEFAULT_CUTS[d] for d in used}
    reward = reward or RewardSpec()
    mdp = build_robot_mdp(df, used, cuts, reward, smoothing)
    sol = value_iteration(mdp.P, mdp.R, gamma, tolerance, seen=mdp.seen, in_place=in_place)
    return mdp, sol, value_table(mdp, sol)


if __name__ == "__main__":
    path = locate_dataset(__file__)
    if path is None:
        raise SystemExit("sensor_readings_24.csv not found")
    data = load_readings(path)
    mdp, sol, table = solve(data)
    table.to_csv("optimal_value_function.csv", index=False)
    print(f"{len(mdp.states)} states | {sol['sweeps']} sweeps | converged: {sol['converged']}")
    print(table.head(8).to_string(index=False))
    print("Saved optimal_value_function.csv")
