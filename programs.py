"""
programs.py - WallBot Lab
The Day 5 notebook programs, rebuilt as plain Python (no Streamlit):

    1. MDP            - the notebook's 3-state example, value iteration and policy iteration
    2. ADP            - offline Q-learning from the robot's log, and state aggregation
    3. Monte Carlo    - the notebook's grid world: first-visit evaluation and on-policy control
    4. Hooke-Jeeves   - the notebook's pattern search, on test functions and on the robot MDP itself
"""

import numpy as np

import engine as E

# ===========================================================================
# 1. MDP - notebook example
# ===========================================================================

NB_STATES = ["s1", "s2", "s3"]
NB_ACTIONS = ["a1", "a2"]
NB_R = np.array([[5, 10], [2, 3], [8, 1]], dtype=float)
NB_T = np.array([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
                 [[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]],
                 [[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]]])


def normalise_rows(T):
    """Negative entries become 0; every (s, a) row is rescaled to sum to 1 (uniform if all zero)."""
    T = np.clip(np.nan_to_num(np.asarray(T, dtype=float)), 0, None)
    fixed = []
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            tot = T[s, a].sum()
            if tot <= 0:
                T[s, a], _ = 1.0 / T.shape[2], fixed.append((s, a))
            elif abs(tot - 1) > 1e-9:
                T[s, a] = T[s, a] / tot
                fixed.append((s, a))
    return T, fixed


def notebook_value_iteration(T, R, gamma, iterations=1000, tolerance=None):
    """The notebook's loop (fixed number of sweeps), recording V after every sweep."""
    n_s, n_a = R.shape
    V = np.zeros(n_s)
    trace, deltas = [V.copy()], []
    for _ in range(iterations):
        V_new = np.zeros(n_s)
        for s in range(n_s):
            Q_sa = np.zeros(n_a)
            for a in range(n_a):
                Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        deltas.append(float(np.max(np.abs(V_new - V))))
        V = V_new
        trace.append(V.copy())
        if tolerance is not None and deltas[-1] < tolerance:
            break
    Q = R + gamma * np.einsum("sat,t->sa", T, V)
    return V, Q, np.array(trace), deltas


def notebook_policy_iteration(T, R, gamma, max_rounds=50):
    """Policy iteration on the same example; returns the policy and values after each round."""
    n_s = R.shape[0]
    pi = np.zeros(n_s, dtype=int)
    rounds = []
    for k in range(max_rounds):
        V = np.linalg.solve(np.eye(n_s) - gamma * T[np.arange(n_s), pi], R[np.arange(n_s), pi])
        Q = R + gamma * np.einsum("sat,t->sa", T, V)
        rounds.append({"round": k + 1, "policy": pi.copy(), "V": V.copy()})
        new_pi = np.where(np.isclose(Q[np.arange(n_s), pi], Q.max(axis=1)), pi, Q.argmax(axis=1))
        if (new_pi == pi).all():
            break
        pi = new_pi
    return rounds


# ===========================================================================
# 2. ADP
# ===========================================================================

def log_transitions(mdp, reward):
    """(s, a, r, s') tuples straight from consecutive log rows, with the same rewards the MDP uses."""
    dfs = mdp.extras["data"]
    idx = {s: i for i, s in enumerate(mdp.states)}
    s = dfs["State"].map(idx).values
    a = dfs["Class"].map({m: i for i, m in enumerate(E.MOVES)}).values
    r = mdp.extras["r_next"][1:] + reward.action_shaping()[a[:-1]]
    return s[:-1], a[:-1], r, s[1:]


def offline_q_learning(transitions, n_s, n_a, seen, gamma, alpha_mode="decaying", alpha=0.1, epochs=30, seed=0, Q_star=None):
    """Q-learning that replays the logged transitions (no model needed):
         Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    alpha_mode 'decaying' uses 1 / n(s,a)^0.6, where n(s,a) is how often that pair has been updated."""
    rng = np.random.default_rng(seed)
    s_arr, a_arr, r_arr, s2_arr = transitions
    Q = np.zeros((n_s, n_a))
    n_upd = np.zeros((n_s, n_a))
    mask = np.where(seen, 0.0, -np.inf)
    hist = {"epoch": [], "max_error": [], "mean_error": [], "agreement": []}
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(s_arr))
        for i in order:
            s, a, r, s2 = s_arr[i], a_arr[i], r_arr[i], s2_arr[i]
            n_upd[s, a] += 1
            step = alpha if alpha_mode == "constant" else 1.0 / n_upd[s, a] ** 0.6
            target = r + gamma * np.max(Q[s2] + mask[s2])
            Q[s, a] += step * (target - Q[s, a])
        if Q_star is not None:
            err = np.abs(np.where(seen, Q - Q_star, 0.0))
            hist["epoch"].append(ep)
            hist["max_error"].append(float(err.max()))
            hist["mean_error"].append(float(err[seen].mean()))
            hist["agreement"].append(float(((Q + mask).argmax(1) == (Q_star + mask).argmax(1)).mean()))
    return Q + mask, hist


def state_aggregation(mdp, keep, gamma):
    """Merge states that agree on the kept distances, solve the smaller MDP, and map values back."""
    cols = [mdp.used.index(k) for k in keep]
    keys = [tuple(row[cols]) for row in mdp.levels] if cols else [() for _ in mdp.levels]
    groups = sorted(set(keys))
    gid = np.array([groups.index(k) for k in keys])
    G, A = len(groups), mdp.R.shape[1]
    w = mdp.counts.sum(axis=2)                               # visits of (s, a)
    Pg = np.zeros((G, A, G))
    Rg = np.zeros((G, A))
    seen_g = np.zeros((G, A), dtype=bool)
    for g in range(G):
        members = np.where(gid == g)[0]
        for a in range(A):
            wa = w[members, a] * mdp.seen[members, a]
            if wa.sum() <= 0:
                continue
            seen_g[g, a] = True
            p_full = (wa[:, None] * mdp.P[members, a]).sum(0) / wa.sum()
            np.add.at(Pg[g, a], gid, p_full)
            Rg[g, a] = (wa * mdp.R[members, a]).sum() / wa.sum()
    for g in range(G):
        if not seen_g[g].any():
            Pg[g, 0, g], seen_g[g, 0] = 1.0, True
    Vg, _, _ = E.fast_value_iteration(Pg, Rg, gamma, seen_g)
    V_back = Vg[gid]
    Q_back = np.where(mdp.seen, mdp.R + gamma * np.einsum("sat,t->sa", mdp.P, V_back), -np.inf)
    return {"groups": G, "V": V_back, "pi": Q_back.argmax(1), "group_id": gid}


# ===========================================================================
# 3. Monte Carlo - the notebook's grid world
# ===========================================================================

ARROWS = ["↑", "↓", "←", "→"]            # notebook action order: Up, Down, Left, Right


class GridEnvironment:
    """Notebook environment (start (0,0), goal bottom-right, -1 per step, +10 at the goal), plus optional traps."""

    def __init__(self, grid_size=(4, 4), start_state=(0, 0), goal_state=None, traps=(), max_steps=400):
        self.grid_size = grid_size
        self.start_state = start_state
        self.goal = goal_state or (grid_size[0] - 1, grid_size[1] - 1)
        self.traps = set(map(tuple, traps))
        self.max_steps = max_steps
        self.actions = [0, 1, 2, 3]
        self.reset()

    def reset(self):
        self.current_pos, self.t = self.start_state, 0
        return self.current_pos

    def step(self, action):
        x, y = self.current_pos
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.grid_size[1] - 1, y + 1)
        self.current_pos, self.t = (x, y), self.t + 1
        if self.current_pos == self.goal:
            return self.current_pos, 10, True, {"end": "goal"}
        if self.current_pos in self.traps:
            return self.current_pos, -10, True, {"end": "trap"}
        if self.t >= self.max_steps:
            return self.current_pos, -1, True, {"end": "timeout"}
        return self.current_pos, -1, False, {"end": None}


def _safe_path_exists(n, traps):
    """Breadth-first search from the start to the goal that never steps on a trap."""
    blocked, goal = set(traps), (n - 1, n - 1)
    frontier, seen = [(0, 0)], {(0, 0)}
    while frontier:
        x, y = frontier.pop()
        if (x, y) == goal:
            return True
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen and (nx, ny) not in blocked:
                seen.add((nx, ny))
                frontier.append((nx, ny))
    return False


def place_traps(n, count, seed):
    """Random traps that never seal the start off from the goal."""
    rng = np.random.default_rng(seed)
    cells = [(i, j) for i in range(n) for j in range(n) if (i, j) not in ((0, 0), (n - 1, n - 1))]
    count = min(count, len(cells))
    while count > 0:
        for _ in range(300):
            pick = [cells[i] for i in rng.choice(len(cells), size=count, replace=False)]
            if _safe_path_exists(n, pick):
                return pick
        count -= 1                      # too crowded: try with one trap fewer
    return []


def play_episode(env, policy):
    s, steps, end = env.reset(), [], None
    while True:
        a = policy(s)
        s2, r, done, info = env.step(a)
        steps.append((s, a, r))
        s = s2
        if done:
            end = info["end"]
            break
    return steps, s, end


def first_visit_mc(env, policy, episodes, gamma):
    """First-visit Monte Carlo evaluation: V(s) = average return after the first visit to s."""
    n0, n1 = env.grid_size
    total, count = np.zeros((n0, n1)), np.zeros((n0, n1))
    start_trace, returns, lengths, ends = [], [], [], []
    for _ in range(episodes):
        steps, _, end = play_episode(env, policy)
        G, first = 0.0, {}
        rets = [0.0] * len(steps)
        for t in reversed(range(len(steps))):
            G = steps[t][2] + gamma * G
            rets[t] = G
        for t, (s, _, _) in enumerate(steps):
            if s not in first:
                first[s] = rets[t]
        for s, g in first.items():
            total[s] += g
            count[s] += 1
        returns.append(rets[0])
        lengths.append(len(steps))
        ends.append(end)
        start_trace.append(total[env.start_state] / max(count[env.start_state], 1))
    V = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return V, count, np.array(start_trace), np.array(returns), np.array(lengths), ends


def mc_control(env, episodes, gamma, epsilon, seed, window=50):
    """On-policy first-visit Monte Carlo control with an epsilon-greedy policy."""
    rng = np.random.default_rng(seed)
    n0, n1 = env.grid_size
    Q = np.zeros((n0, n1, 4))
    N = np.zeros((n0, n1, 4))

    def policy(s):
        if rng.random() < epsilon:
            return int(rng.integers(4))
        q = Q[s]
        return int(rng.choice(np.flatnonzero(q == q.max())))

    success, curve = [], []
    for ep in range(1, episodes + 1):
        steps, _, end = play_episode(env, policy)
        G, seen_sa = 0.0, set()
        firsts = {}
        for t, (s, a, _) in enumerate(steps):
            if (s, a) not in firsts:
                firsts[(s, a)] = t
        for t in reversed(range(len(steps))):
            s, a, r = steps[t]
            G = r + gamma * G
            if firsts[(s, a)] == t and (s, a) not in seen_sa:
                seen_sa.add((s, a))
                N[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / N[s][a]
        success.append(end == "goal")
        if ep % window == 0:
            curve.append((ep, float(np.mean(success[-window:]))))
    greedy = Q.argmax(axis=2)
    return Q, greedy, curve


def greedy_rollout(env, greedy, limit=200):
    s, pts = env.reset(), [env.start_state]
    for _ in range(limit):
        s, _, done, info = env.step(int(greedy[s]))
        pts.append(s)
        if done:
            return pts, info["end"]
    return pts, "timeout"


# ===========================================================================
# 4. Hooke-Jeeves
# ===========================================================================

TEST_FUNCTIONS = {
    "x² + y² (notebook)": (lambda v: v[0] ** 2 + v[1] ** 2, (-3, 3, -3, 3), [(0, 0)]),
    "Three-hump camel": (lambda v: 2 * v[0] ** 2 - 1.05 * v[0] ** 4 + v[0] ** 6 / 6 + v[0] * v[1] + v[1] ** 2, (-2.5, 2.5, -2.5, 2.5), [(0, 0)]),
    "Booth": (lambda v: (v[0] + 2 * v[1] - 7) ** 2 + (2 * v[0] + v[1] - 5) ** 2, (-4, 6, -4, 6), [(1, 3)]),
    "Himmelblau": (lambda v: (v[0] ** 2 + v[1] - 11) ** 2 + (v[0] + v[1] ** 2 - 7) ** 2, (-5, 5, -5, 5),
                   [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]),
    "Rosenbrock": (lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2, (-2, 2, -1, 3), [(1, 1)]),
}


def hooke_jeeves(func, x0, step_size=0.5, epsilon=1e-6, max_iter=1000, verify_pattern=True):
    """The notebook's algorithm, recording every iteration.
    verify_pattern=False reproduces the notebook exactly (the pattern move is always kept)."""
    x = np.array(x0, dtype=float)
    delta, it, calls = step_size, 0, [0]
    log, probes = [], []

    def f(v):
        calls[0] += 1
        return func(v)

    def explore(x, delta):
        for i in range(len(x)):
            f_val = f(x)
            x[i] += delta
            probes.append(x.copy())
            if f(x) < f_val:
                continue
            x[i] -= 2 * delta
            probes.append(x.copy())
            if f(x) < f_val:
                continue
            x[i] += delta
        return x

    while delta > epsilon and it < max_iter:
        it += 1
        base = np.copy(x)
        x = explore(x, delta)
        if np.array_equal(x, base):
            kind, used = "shrink", delta
            delta /= 2
        else:
            used = delta
            trial = x + (x - base)
            if verify_pattern and not (f(trial) < f(x)):
                kind = "explore"
            else:
                kind, x = "pattern", trial
        log.append({"iter": it, "base": base.copy(), "x": x.copy(), "f": float(func(x)), "step": used, "move": kind})
    return x, log, np.array(probes) if probes else np.zeros((0, len(x0))), calls[0]


class RobotTuner:
    """Objective for tuning the robot MDP with Hooke-Jeeves.
    The data, states and transition model are prepared once; each evaluation only recomputes the
    rewards for a new target wall distance and re-solves for a new discount factor."""

    def __init__(self, df, used, cuts, reward, smoothing):
        self.base = reward
        self.mdp = E.build_robot_mdp(df, used, cuts, reward, smoothing)
        dfs = self.mdp.extras["data"]
        idx = {s: i for i, s in enumerate(self.mdp.states)}
        self.s = dfs["State"].map(idx).values
        self.a = dfs["Class"].map({m: i for i, m in enumerate(E.MOVES)}).values
        self.front, self.left = dfs["SD_front"].values, dfs["SD_left"].values
        self.n_sa = self.mdp.counts.sum(axis=2)
        self.calls = 0

    BOUNDS = ((0.5, 0.99), (0.3, 1.5))

    def clip(self, gamma, target_left):
        return (float(np.clip(gamma, *self.BOUNDS[0])), float(np.clip(target_left, *self.BOUNDS[1])))

    def agreement(self, gamma, target_left):
        gamma, target_left = self.clip(gamma, target_left)
        key = (round(gamma, 4), round(target_left, 4))
        if not hasattr(self, "_memo"):
            self._memo = {}
        if key in self._memo:
            return self._memo[key]
        self.calls += 1
        spec = E.RewardSpec(**{**self.base.__dict__, "target_left": target_left})
        r_next = spec.reading_reward(self.front, self.left)
        S, A = self.mdp.R.shape
        rsum = np.zeros((S, A))
        np.add.at(rsum, (self.s[:-1], self.a[:-1]), r_next[1:])
        R = np.where(self.mdp.seen, rsum / np.maximum(self.n_sa, 1), 0.0) + spec.action_shaping()[None, :]
        _, _, pi = E.fast_value_iteration(self.mdp.P, R, gamma, self.mdp.seen, tolerance=1e-7)
        self._memo[key] = float((pi[self.s] == self.a).mean())
        return self._memo[key]
