"""
WallBot Lab - wall-following robot navigation with MDPs
=======================================================
Streamlit front end. The maths lives in engine.py and programs.py, the two
interactive browser components in web/.

Run:  streamlit run app.py
"""

import io
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import engine as E
import programs as PG
import web_bridge as WB

HERE = Path(__file__).parent
CSV_OUT = HERE / "optimal_value_function.csv"

st.set_page_config(page_title="WallBot Lab", page_icon="🛰️", layout="wide")

# ---------------------------------------------------------------------------
# Version-safe helpers
# ---------------------------------------------------------------------------

try:
    _V = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _V = (0, 0)
MODERN = _V >= (1, 50)
memo = getattr(st, "cache_data", None) or st.cache


def chart(fig, height=None):
    if height:
        fig.update_layout(height=height)
    if MODERN:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def table(df, height=None, index=False):
    kw = {"height": height} if height else {}
    kw.update({"width": "stretch"} if MODERN else {"use_container_width": True})
    if not index and _V >= (1, 23):
        kw["hide_index"] = True
    try:
        st.dataframe(df, **kw)
    except TypeError:
        st.dataframe(df)


def editable(df, key):
    ed = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
    if ed is None:
        table(df, index=True)
        return df
    try:
        return ed(df, key=key, **({"width": "stretch"} if MODERN else {"use_container_width": True}))
    except TypeError:
        return ed(df, key=key)


def rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()


def embed(name, data, height):
    components.html(WB.load_html(name, data), height=height, scrolling=True)


def callout(text):
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def show_code(code, title="Code from the Day 5 notebook"):
    with st.expander(title):
        st.code(code, language="python")


PAPER = "rgba(0,0,0,0)"
INK = "#E6EDF3"
GRIDC = "#243241"
MCOL = WB.MOVE_COLORS


def styled(fig, height=380, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PAPER, plot_bgcolor="#101A24", height=height,
                      font=dict(color=INK), margin=dict(t=46 if title else 16, b=16, l=10, r=10),
                      title=dict(text=title, font=dict(size=14)) if title else None)
    fig.update_xaxes(gridcolor=GRIDC, zerolinecolor=GRIDC)
    fig.update_yaxes(gridcolor=GRIDC, zerolinecolor=GRIDC)
    return fig


st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1300px;}
.callout {background:#16212C; border:1px solid #243241; border-left:4px solid #2EC4B6; padding:.7rem 1rem;
          border-radius:8px; margin-bottom:1rem; color:#C9D4DE;}
div[data-testid="stMetricValue"] {font-size:1.55rem;}
h1, h2, h3 {letter-spacing:-0.01em;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------

@memo(show_spinner=False)
def read_data(blob, path):
    return E.load_readings(io.BytesIO(blob) if blob is not None else path)


@memo(show_spinner="Building the MDP and solving it…")
def solve_robot(df, used, cuts_items, gamma, tol, reward_items, smoothing, in_place):
    cuts = {k: list(v) for k, v in cuts_items}
    rs = E.RewardSpec(**dict(reward_items))
    mdp, sol, tab = E.solve(df, list(used), cuts, gamma, tol, rs, smoothing, in_place)
    V_pi, Q_pi, pi_pi, rounds = E.policy_iteration(mdp.P, mdp.R, gamma, mdp.seen)
    return mdp, sol, tab, {"V": V_pi, "pi": pi_pi, "rounds": rounds}


@memo(show_spinner="Replaying the log with Q-learning…")
def run_q_learning(_mdp0, key, gamma, mode, alpha, epochs, seed, reward_items):
    rs = E.RewardSpec(**dict(reward_items))
    _, Q_star, _ = E.fast_value_iteration(_mdp0.P, _mdp0.R, gamma, _mdp0.seen)
    tr = PG.log_transitions(_mdp0, rs)
    Q, hist = PG.offline_q_learning(tr, len(_mdp0.states), len(E.MOVES), _mdp0.seen, gamma, mode, alpha, epochs, seed, Q_star)
    return Q, Q_star, hist


@memo(show_spinner="Simulating episodes…")
def run_mc_eval(n, traps, policy_kind, eps, episodes, gamma, seed):
    rng = np.random.default_rng(seed)
    goal = (n - 1, n - 1)

    def policy(s):
        if policy_kind == "Random (notebook)" or rng.random() < eps:
            return int(rng.integers(4))
        options = ([1] if s[0] < goal[0] else []) + ([3] if s[1] < goal[1] else [])
        return int(rng.choice(options)) if options else int(rng.integers(4))

    env = PG.GridEnvironment((n, n), traps=traps)
    V, cnt, trace, rets, lens, ends = PG.first_visit_mc(env, policy, episodes, gamma)
    sample = [PG.play_episode(PG.GridEnvironment((n, n), traps=traps), policy) for _ in range(12)]
    return V, cnt, trace, rets, lens, ends, sample


@memo(show_spinner="Learning with Monte Carlo control…")
def run_mc_control(n, traps, episodes, gamma, eps, seed):
    env = PG.GridEnvironment((n, n), traps=traps)
    Q, greedy, curve = PG.mc_control(env, episodes, gamma, eps, seed)
    path, end = PG.greedy_rollout(PG.GridEnvironment((n, n), traps=traps), greedy)
    return Q, greedy, curve, path, end


@memo(show_spinner="Mapping the tuning landscape…")
def tuning_landscape(df, used, cuts_items, reward_items, smoothing, n=13):
    tuner = PG.RobotTuner(df, list(used), {k: list(v) for k, v in cuts_items}, E.RewardSpec(**dict(reward_items)), smoothing)
    gs = np.linspace(0.5, 0.99, n)
    ts = np.linspace(0.3, 1.5, n)
    Z = np.array([[tuner.agreement(g, t) for g in gs] for t in ts])
    return gs, ts, Z


# ---------------------------------------------------------------------------
# Sidebar: navigation + robot MDP settings
# ---------------------------------------------------------------------------

PAGES = ["🏠 Overview", "🤖 Simulator", "🎞️ Value iteration player", "📐 MDP (notebook)", "🧠 ADP",
         "🎲 Monte Carlo", "🧭 Hooke-Jeeves", "📊 Data & results"]

with st.sidebar:
    st.markdown("## 🛰️ WallBot Lab")
    page = st.radio("Go to", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Robot MDP settings**")
    upload = st.file_uploader("Sensor file (optional)", type=["csv", "data"])
    default_path = E.locate_dataset(__file__)
    if upload is None and default_path is None:
        st.error("sensor_readings_24.csv not found. Put it next to app.py (or in data/), or upload it here.")
        st.stop()
    try:
        df = read_data(upload.getvalue() if upload else None, str(default_path) if default_path else None)
    except Exception as exc:
        st.error(f"Could not read the sensor file: {exc}")
        st.stop()
    if len(df) < 20:
        st.error("The sensor file has too few valid rows.")
        st.stop()

    used = st.multiselect("Distances in the state", E.DIST, default=E.DIST, key="used")
    if not used:
        st.warning("Choose at least one distance.")
        st.stop()
    used = [d for d in E.DIST if d in used]
    level_mode = st.radio("How distances become levels", ["Fixed thresholds (metres)", "Quantiles"], key="level_mode")
    cuts = {}
    if level_mode.startswith("Fixed"):
        with st.expander("Thresholds (Close | Medium | Far)", expanded=False):
            for d in used:
                lo, hi = E.DEFAULT_CUTS[d]
                a, b = st.slider(d, 0.2, 3.0, (float(lo), float(hi)), 0.05, key=f"cut_{d}")
                cuts[d] = [a, b] if b > a else [a, a + 0.05]
    else:
        k = st.select_slider("Levels per distance", [2, 3, 4, 5], value=3, key="q_levels")
        cuts = E.quantile_cuts(df, used, k)
        cuts = {d: (c if len(c) >= 1 else [float(df[d].median())]) for d, c in cuts.items()}

    gamma = st.slider("Discount factor γ", 0.50, 0.99, 0.90, 0.01, key="gamma")
    tol = float(st.select_slider("Convergence threshold θ", ["1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"],
                                 value="1e-06", key="tol"))
    smoothing = st.slider("Laplace smoothing α", 0.0, 2.0, 0.5, 0.1, key="smoothing",
                          help="Added to every next state already reached from s, so rare transitions aren't zero.")
    in_place = st.toggle("In-place (Gauss-Seidel) sweeps", value=False, key="in_place") if hasattr(st, "toggle") \
        else st.checkbox("In-place (Gauss-Seidel) sweeps", value=False, key="in_place")
    with st.expander("Reward shaping"):
        d0 = E.RewardSpec()
        rs_vals = {
            "target_left": st.slider("Target left-wall distance (m)", 0.3, 1.5, d0.target_left, 0.05, key="target_left"),
            "tolerance": st.slider("Tolerance around the target (m)", 0.05, 0.6, d0.tolerance, 0.05, key="rw_tol"),
            "follow_gain": st.slider("Wall-following bonus", 0.0, 3.0, d0.follow_gain, 0.1, key="rw_follow"),
            "safe_front": st.slider("Front safety distance (m)", 0.3, 1.5, d0.safe_front, 0.05, key="rw_front"),
            "safe_left": st.slider("Left safety distance (m)", 0.2, 1.0, d0.safe_left, 0.05, key="rw_left"),
            "crash_gain": st.slider("Proximity penalty", 0.0, 10.0, d0.crash_gain, 0.5, key="rw_crash"),
            "forward_bonus": st.slider("Forward bonus", 0.0, 1.0, d0.forward_bonus, 0.05, key="rw_fwd"),
            "sharp_cost": st.slider("Sharp-turn cost", 0.0, 1.0, d0.sharp_cost, 0.05, key="rw_sharp"),
        }

reward = E.RewardSpec(**rs_vals)
cuts_items = tuple((d, tuple(float(x) for x in cuts[d])) for d in used)
reward_items = tuple(rs_vals.items())
mdp, sol, vtab, pi_res = solve_robot(df, tuple(used), cuts_items, gamma, tol, reward_items, smoothing, in_place)
try:
    vtab.to_csv(CSV_OUT, index=False)
    csv_saved = True
except Exception:
    csv_saved = False
dfs = mdp.extras["data"]
best_map = dict(zip(vtab["State"], vtab["Optimal_Action"]))
agree = float((dfs["State"].map(best_map) == dfs["Class"]).mean())
robot_key = (tuple(used), cuts_items, gamma, tol, reward_items, smoothing)


def headline():
    c = st.columns(4)
    c[0].metric("Log readings", f"{len(df):,}")
    c[1].metric("States", len(mdp.states), f"of {int(np.prod([len(cuts[d]) + 1 for d in used]))} possible", delta_color="off")
    c[2].metric("Value-iteration sweeps", sol["sweeps"], "converged" if sol["converged"] else "not converged",
                delta_color="normal" if sol["converged"] else "inverse")
    c[3].metric("Matches the logged move", f"{agree:.0%}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    st.title("WallBot Lab")
    st.caption("Wall-following robot navigation as a Markov Decision Process, solved with value iteration.")
    headline()
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("What the exercise asks for, and where it lives")
        st.markdown(f"""
| Slide item | In WallBot Lab |
|---|---|
| **States** | {", ".join(used)} turned into levels ({level_mode.lower()}); {len(mdp.states)} combinations occur in the log |
| **Action space** | {", ".join(E.MOVES)} |
| **Transition probabilities** | Counted from consecutive readings (9 per second) with Laplace smoothing α = {smoothing} |
| **Rewards** | Gaussian bonus around {reward.target_left:.2f} m from the left wall, graded proximity penalties, forward bonus, sharp-turn cost |
| **Discount factor γ** | {gamma} |
| **Convergence threshold** | {tol:.0e}, value iteration stops when no value changes by more than this |
| **Output** | `optimal_value_function.csv` (download on the Data & results page) |
""")
        callout("Use the sidebar to change any MDP setting. Every page updates straight away: the simulator's robot, the "
                "value iteration player, the results table and the CSV.")
    with right:
        st.subheader("Pipeline")
        try:
            st.graphviz_chart("""digraph { rankdir=TB; bgcolor="transparent";
              node [shape=box, style="rounded,filled", fillcolor="#16212C", color="#2EC4B6", fontcolor="#E6EDF3", fontname="Helvetica", fontsize=11];
              edge [color="#8FA1B3"];
              a [label="24 ultrasound sensors"]; b [label="4 simplified distances (min of each arc)"];
              c [label="Levels -> MDP states"]; d [label="P(s'|s,a) with smoothing, R(s,a) shaped"];
              e [label="Value iteration (θ, γ)"]; f [label="V*, π*  ->  optimal_value_function.csv"];
              a -> b -> c -> d -> e -> f; }""")
        except Exception:
            st.write("24 sensors → 4 distances → states → P and R → value iteration → V*, π*")
        counts = df["Class"].value_counts().reindex(E.MOVES).fillna(0)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                               marker=dict(colors=[MCOL[m] for m in counts.index])))
        chart(styled(fig, 300, "Moves recorded in the log"))


def page_simulator():
    st.title("🤖 Simulator")
    callout("Place a <b>start</b> (●) and a <b>finish</b> (⚑), then press Run. The <b>learned wall-follower</b> reads its "
            "four sensor arcs, turns them into an MDP state and takes the optimal move from value iteration. The "
            "<b>goal seeker</b> plans on the grid itself: its own MDP with 4 moves and slip, solved by value iteration "
            "in your browser. The chart under the room tracks the left and front distances against the target band.")
    embed("simulator.html", WB.simulator_data(mdp, sol, reward), 1000)
    with st.expander("How the simulator works"):
        st.markdown(f"""
- **Room:** 20 × 14 squares of 0.30 m (6.0 m × 4.2 m). Draw or erase walls by clicking and dragging.
- **Sensing:** 13 rays per 60° arc; the shortest is the simplified distance, just like SD_front, SD_left, SD_right and SD_back in the dataset.
- **State and action:** the distances are cut with the same thresholds as the MDP, and the move comes from the optimal policy π*.
- **Finish:** the run ends when the rover is within 0.35 m of the flag. The wall-follower only passes finishes along the wall it follows (clockwise, wall on its left). In the maze it can't get through, which is where the goal seeker comes in.
- **Goal seeker MDP:** states are free squares, actions north/east/south/west, each move costs 1, bumping a wall costs 3 more, γ = 0.98. With slip p the rover goes sideways with probability p/2 each way, and the plan accounts for that.
- **Rewards shown** use the same shaping as the sidebar (target {reward.target_left:.2f} m).
""")


def page_vi():
    st.title("🎞️ Value iteration player")
    callout("Press Play to watch every state's value grow sweep by sweep, and the best move settle. Click a bar to see "
            "its Bellman update worked out with real numbers.")
    embed("vi_player.html", WB.vi_player_data(mdp, sol, gamma, tol), 1080)
    c = st.columns(3)
    other = E.value_iteration(mdp.P, mdp.R, gamma, tol, seen=mdp.seen, in_place=not in_place)
    c[0].metric("Sweeps (current setting)", sol["sweeps"], "in-place" if in_place else "standard", delta_color="off")
    c[1].metric("Sweeps with the other update", other["sweeps"], "standard" if in_place else "in-place", delta_color="off")
    c[2].metric("Policy iteration rounds", len(pi_res["rounds"]),
                f"same policy: {np.mean(pi_res['pi'] == sol['pi']):.0%}", delta_color="off")
    show_code('''def value_iteration(P, R, gamma, tolerance=1e-6, max_iterations=20_000, seen=None, in_place=False):
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(max_iterations):
        V_old = V.copy()
        V_new = V if in_place else np.zeros(S)
        for s in range(S):
            Q_sa = np.full(A, -np.inf)
            for a in range(A):
                if seen[s, a]:                                   # only moves recorded in the log
                    Q_sa[a] = R[s][a] + gamma * np.dot(P[s][a], V)
            V_new[s] = np.max(Q_sa)
        V = V_new
        if np.max(np.abs(V - V_old)) < tolerance:               # convergence threshold
            break
    return V''', "Value iteration used here (same update as the notebook)")


def page_mdp():
    st.title("📐 MDP: the notebook example")
    callout("The Day 5 notebook's three-state, two-action MDP. Edit any number and it is solved again with "
            "<b>value iteration</b> and <b>policy iteration</b>, which should always agree.")
    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("**Rewards R(s, a)**")
        r_df = editable(pd.DataFrame(PG.NB_R, index=PG.NB_STATES, columns=PG.NB_ACTIONS), "nb_R")
        st.markdown("**Transitions P(s' | s, a)**, each row is rescaled to sum to 1")
        rows = [f"{s} · {a}" for s in PG.NB_STATES for a in PG.NB_ACTIONS]
        t_df = editable(pd.DataFrame(PG.NB_T.reshape(6, 3), index=rows, columns=PG.NB_STATES), "nb_T")
        g_nb = st.slider("γ", 0.0, 0.99, 0.9, 0.01, key="nb_gamma")
        mode = st.radio("Stop after", ["1000 sweeps (notebook)", "Δ below 1e-6"], horizontal=True, key="nb_mode")
    try:
        R = pd.DataFrame(r_df).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float).reshape(3, 2)
        T, fixed = PG.normalise_rows(pd.DataFrame(t_df).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float).reshape(3, 2, 3))
    except Exception:
        st.warning("The tables could not be read; the notebook values are used instead.")
        R, (T, fixed) = PG.NB_R.copy(), (PG.NB_T.copy(), [])
    V, Q, trace, deltas = PG.notebook_value_iteration(T, R, g_nb, 1000, None if mode.startswith("1000") else 1e-6)
    rounds = PG.notebook_policy_iteration(T, R, g_nb)
    pi = Q.argmax(1)
    with right:
        if fixed:
            st.caption("Rescaled rows: " + ", ".join(f"{PG.NB_STATES[s]} · {PG.NB_ACTIONS[a]}" for s, a in fixed))
        m = st.columns(3)
        for i, s in enumerate(PG.NB_STATES):
            m[i].metric(f"V*({s})", f"{V[i]:.2f}", f"take {PG.NB_ACTIONS[pi[i]]}", delta_color="off")
        # Sankey of the optimal policy's transitions
        src, tgt, val, colr, lab = [], [], [], [], []
        for s in range(3):
            for t in range(3):
                p = T[s, pi[s], t]
                if p > 0.001:
                    src.append(s); tgt.append(3 + t); val.append(p)
                    colr.append("rgba(46,196,182,0.45)" if pi[s] == 0 else "rgba(255,179,71,0.45)")
                    lab.append(f"{PG.NB_STATES[s]} --{PG.NB_ACTIONS[pi[s]]}--> {PG.NB_STATES[t]}: {p:.2f}")
        fig = go.Figure(go.Sankey(
            node=dict(label=[f"{s} (now)" for s in PG.NB_STATES] + [f"{s} (next)" for s in PG.NB_STATES],
                      color=["#2EC4B6", "#5AA9E6", "#9D8DF1"] * 2, pad=18, thickness=16),
            link=dict(source=src, target=tgt, value=val, color=colr, label=lab)))
        chart(styled(fig, 330, "Where the optimal policy leads (teal = a1, amber = a2)"))
        st.caption(f"Value iteration stopped after {len(deltas)} sweeps. Policy iteration needed {len(rounds)} round(s) "
                   f"and found the same policy: {'yes' if (rounds[-1]['policy'] == pi).all() else 'no'}.")
    st.subheader("Convergence")
    a, b = st.columns([1.3, 1])
    with a:
        kmax = min(len(trace) - 1, 120)
        fig = go.Figure()
        for i, s in enumerate(PG.NB_STATES):
            fig.add_trace(go.Scatter(x=np.arange(kmax + 1), y=trace[: kmax + 1, i], name=s, mode="lines",
                                     line=dict(width=2.5, color=["#2EC4B6", "#5AA9E6", "#9D8DF1"][i])))
        k = st.slider("Inspect sweep k", 0, kmax, min(3, kmax), key="nb_k")
        fig.add_vline(x=k, line_dash="dot", line_color="#FFB347")
        chart(styled(fig, 320, "V_k(s) for the first sweeps"))
    with b:
        Vk = trace[k]
        rows = []
        for s in range(3):
            q = [R[s, a] + g_nb * float(T[s, a] @ Vk) for a in range(2)]
            for a in range(2):
                rows.append({"s": PG.NB_STATES[s], "a": PG.NB_ACTIONS[a],
                             "R + γ·Σ P·V_k": f"{R[s, a]:.1f} + {g_nb:.2f}·({' + '.join(f'{T[s, a, j]:.2f}×{Vk[j]:.1f}' for j in range(3))})",
                             "Q": round(q[a], 3), "best": "★" if a == int(np.argmax(q)) else ""})
        st.markdown(f"**Bellman update at sweep {k}**")
        table(pd.DataFrame(rows))
        pi_rows = [{"Round": r["round"], "Policy": ", ".join(f"{PG.NB_STATES[i]}→{PG.NB_ACTIONS[a]}" for i, a in enumerate(r["policy"])),
                    "V": ", ".join(f"{v:.2f}" for v in r["V"])} for r in rounds]
        st.markdown("**Policy iteration, round by round**")
        table(pd.DataFrame(pi_rows))
    show_code('''states = ['s1', 's2', 's3']; actions = ['a1', 'a2']
R = np.array([[5, 10], [2, 3], [8, 1]])
T = np.array([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
              [[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]],
              [[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]]])
gamma = 0.9

def value_iteration(T, R, gamma, iterations=1000):
    V = np.zeros(len(states))
    for i in range(iterations):
        V_new = np.zeros(len(states))
        for s in range(len(states)):
            Q_sa = np.zeros(len(actions))
            for a in range(len(actions)):
                Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        V = V_new
    return V''')


def page_adp():
    st.title("🧠 Approximate Dynamic Programming")
    callout("Exact value iteration needs the full model P and one value per state. ADP relaxes one of those. "
            "<b>Offline Q-learning</b> learns straight from the logged (s, a, r, s') samples with no model at all. "
            "<b>State aggregation</b> solves a smaller MDP by merging states, then uses those values for the full one.")
    method = st.radio("Method", ["Offline Q-learning from the robot log", "State aggregation"], horizontal=True, key="adp_m")
    if method.startswith("Offline"):
        c = st.columns(4)
        epochs = c[0].slider("Passes over the log (epochs)", 5, 60, 25, 5, key="ql_ep")
        rate = c[1].radio("Learning rate", ["Decaying 1/n(s,a)^0.6", "Constant α"], key="ql_mode")
        amode = "decaying" if rate.startswith("Decaying") else "constant"
        alpha = c[2].slider("Constant α", 0.01, 0.5, 0.05, 0.01, key="ql_alpha", disabled=amode == "decaying")
        seed = c[3].number_input("Shuffle seed", 0, 999, 0, key="ql_seed")
        mdp0 = E.build_robot_mdp(df, used, cuts, reward, smoothing=0.0)
        Q, Q_star, h = run_q_learning(mdp0, robot_key, gamma, amode, alpha, epochs, int(seed), reward_items)
        seen = mdp0.seen
        m = st.columns(4)
        m[0].metric("Samples replayed per epoch", f"{len(df) - 1:,}")
        m[1].metric("Max |Q − Q*|", f"{h['max_error'][-1]:.3f}")
        m[2].metric("Mean |Q − Q*|", f"{h['mean_error'][-1]:.3f}")
        m[3].metric("Same best move as Q*", f"{h['agreement'][-1]:.0%}")
        a, b = st.columns(2)
        with a:
            fig = go.Figure([go.Scatter(x=h["epoch"], y=h["max_error"], name="max error", line=dict(color="#FF6B6B", width=2.5)),
                             go.Scatter(x=h["epoch"], y=h["mean_error"], name="mean error", line=dict(color="#2EC4B6", width=2.5))])
            fig.update_yaxes(type="log")
            chart(styled(fig, 340, "Error against the exact Q* after each pass"))
        with b:
            qs, qe = Q_star[seen], Q[seen]
            fig = go.Figure(go.Scatter(x=qs, y=qe, mode="markers", marker=dict(color="#5AA9E6", size=7, opacity=0.8)))
            lo, hi = float(min(qs.min(), qe.min())), float(max(qs.max(), qe.max()))
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#8FA1B3", dash="dash"))
            fig.update_xaxes(title="exact Q*(s,a)"); fig.update_yaxes(title="learned Q(s,a)")
            chart(styled(fig, 340, "Every (state, move) pair"))
        st.caption("Q* here is the exact solution of the same log without smoothing, the model Q-learning is implicitly "
                   "learning. A decaying learning rate averages out the randomness of individual transitions; a "
                   "constant one keeps chasing it.")
    else:
        keep = st.multiselect("Distances the aggregated MDP keeps", used, default=used[:2], key="agg_keep")
        V_ex, _, pi_ex = E.fast_value_iteration(mdp.P, mdp.R, gamma, mdp.seen)
        res = PG.state_aggregation(mdp, keep, gamma)
        m = st.columns(4)
        m[0].metric("States in the small MDP", res["groups"], f"vs {len(mdp.states)}", delta_color="off")
        m[1].metric("Max |V − V*|", f"{np.abs(res['V'] - V_ex).max():.3f}")
        m[2].metric("Mean |V − V*|", f"{np.abs(res['V'] - V_ex).mean():.3f}")
        m[3].metric("Same best move as exact", f"{np.mean(res['pi'] == pi_ex):.0%}")
        rows = []
        for r in range(len(used) + 1):
            for sub in itertools.combinations(used, r):
                out = PG.state_aggregation(mdp, list(sub), gamma)
                rows.append({"Kept": ", ".join(x.replace("SD_", "") for x in sub) or "(nothing)", "States": out["groups"],
                             "Mean error": float(np.abs(out["V"] - V_ex).mean()), "Policy match": float(np.mean(out["pi"] == pi_ex))})
        tab = pd.DataFrame(rows).sort_values("States")
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Scatter(x=tab["States"], y=tab["Mean error"], mode="markers+text", text=tab["Kept"],
                                       textposition="top center", marker=dict(size=11, color=tab["Policy match"],
                                       colorscale=[[0, "#FF6B6B"], [1, "#2EC4B6"]], showscale=True, colorbar=dict(title="match"))))
            fig.update_xaxes(title="states in the aggregated MDP", type="log"); fig.update_yaxes(title="mean |V − V*|")
            chart(styled(fig, 400, "Fewer states, more error: every possible aggregation"))
        with b:
            fig = go.Figure(go.Scatter(x=V_ex, y=res["V"], mode="markers", marker=dict(color="#9D8DF1", size=8)))
            lo, hi = float(min(V_ex.min(), res["V"].min())), float(max(V_ex.max(), res["V"].max()))
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#8FA1B3", dash="dash"))
            fig.update_xaxes(title="exact V*"); fig.update_yaxes(title="aggregated V")
            chart(styled(fig, 400, "Your selection, state by state"))
        st.caption("The best move is chosen with a one-step look-ahead through the full model, so even rough values "
                   "often pick the right move: most of the decision comes from the immediate reward.")
    show_code('''# "Approximate Dynamic Programming" cell of the Day 5 notebook
def value_iteration(T, R, gamma, iterations=1000):
    V = np.zeros(len(states))
    for i in range(iterations):
        V_new = np.zeros(len(states))
        for s in range(len(states)):
            Q_sa = np.zeros(len(actions))
            for a in range(len(actions)):
                Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        V = V_new
    return V
# Offline Q-learning replaces np.dot(T[s][a], V) by one logged sample r + gamma * max Q(s', .)''')


def grid_animation(n, traps, pts, title):
    base = np.zeros((n, n))
    for t in traps:
        base[t] = -1
    base[n - 1, n - 1] = 1
    xs, ys = [p[1] for p in pts][:300], [p[0] for p in pts][:300]
    bg = go.Heatmap(z=base, showscale=False, hoverinfo="skip", xgap=3, ygap=3, zmin=-1, zmax=1,
                    colorscale=[[0, "#5A2A2E"], [0.5, "#1B2733"], [1, "#1F4D47"]])
    trail = lambda i: go.Scatter(x=xs[: i + 1], y=ys[: i + 1], mode="lines", line=dict(color="#5AA9E6", width=3))
    bot = lambda i: go.Scatter(x=[xs[i]], y=[ys[i]], mode="markers", marker=dict(size=24, color="#FFB347", symbol="diamond", line=dict(color="#0F1720", width=2)))
    fig = go.Figure(data=[bg, trail(0), bot(0)], frames=[go.Frame(data=[bg, trail(i), bot(i)], name=str(i)) for i in range(len(xs))])
    ann = [dict(x=0, y=0, text="S", showarrow=False, font=dict(color="#2EC4B6", size=15)),
           dict(x=n - 1, y=n - 1, text="G", showarrow=False, font=dict(color="#2EC4B6", size=15))]
    ann += [dict(x=t[1], y=t[0], text="✕", showarrow=False, font=dict(color="#FF6B6B", size=15)) for t in traps]
    styled(fig, 500, title)
    fig.update_layout(showlegend=False, annotations=ann, margin=dict(t=70, b=80, l=10, r=10),
                      xaxis=dict(range=[-0.5, n - 0.5], dtick=1, title="Y", constrain="domain", showgrid=False),
                      yaxis=dict(range=[n - 0.5, -0.5], dtick=1, title="X", scaleanchor="x", constrain="domain", showgrid=False),
                      updatemenus=[dict(type="buttons", x=1, y=1.02, xanchor="right", yanchor="bottom", direction="left", showactive=False,
                                        buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=170, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                                                 dict(label="⏸", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
                      sliders=[dict(x=0, len=1, y=-0.12, yanchor="top", currentvalue=dict(prefix="Step "),
                                    steps=[dict(label=str(i), method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i in range(len(xs))])])
    return fig


def page_mc():
    st.title("🎲 Monte Carlo")
    callout("The notebook's grid world: start at S, +10 at G, −1 per step, ✕ traps cost −10 and end the episode. "
            "Monte Carlo methods learn from complete episodes: <b>evaluation</b> estimates how good a policy is, "
            "<b>control</b> improves the policy as it goes.")
    c = st.columns(4)
    n = c[0].slider("Grid size", 4, 10, 5, key="mc_n")
    n_traps = c[1].slider("Traps", 0, 12, 3, key="mc_traps")
    layout_seed = c[2].number_input("Trap layout", 0, 999, 4, key="mc_layout")
    gamma_mc = c[3].slider("γ", 0.5, 0.99, 0.9, 0.01, key="mc_gamma")
    traps = PG.place_traps(n, n_traps, int(layout_seed))
    mode = st.radio("Task", ["Evaluate a policy (first-visit Monte Carlo)", "Learn a policy (Monte Carlo control)"], horizontal=True, key="mc_mode")
    if mode.startswith("Evaluate"):
        c = st.columns(3)
        kind = c[0].selectbox("Policy", ["Random (notebook)", "Head for the goal, ε random"], key="mc_kind")
        eps = c[1].slider("ε", 0.0, 1.0, 0.3, 0.05, key="mc_eps", disabled=kind.startswith("Random"))
        episodes = c[2].select_slider("Episodes", [200, 500, 1000, 2000, 5000], value=1000, key="mc_episodes")
        V, cnt, trace, rets, lens, ends, sample = run_mc_eval(n, tuple(traps), kind, eps, episodes, gamma_mc, 11)
        m = st.columns(4)
        m[0].metric("Estimated V(start)", f"{trace[-1]:.2f}")
        m[1].metric("Mean episode length", f"{lens.mean():.1f}")
        m[2].metric("Reached the goal", f"{ends.count('goal') / len(ends):.0%}")
        m[3].metric("Fell in a trap", f"{ends.count('trap') / len(ends):.0%}")
        a, b = st.columns(2)
        with a:
            txt = np.where(np.isnan(V), "", np.vectorize(lambda v: f"{v:.1f}")(np.nan_to_num(V)))
            for t in traps:
                txt[t] = "✕"
            txt[n - 1, n - 1] = "G"
            fig = go.Figure(go.Heatmap(z=V, text=txt, texttemplate="%{text}", colorscale=[[0, "#FF6B6B"], [0.5, "#1B2733"], [1, "#2EC4B6"]],
                                       xgap=3, ygap=3, colorbar=dict(title="V")))
            fig.update_yaxes(autorange="reversed", title="X"); fig.update_xaxes(title="Y")
            chart(styled(fig, 420, "Value of every square under this policy"))
        with b:
            fig = go.Figure(go.Scatter(x=np.arange(1, len(trace) + 1), y=trace, line=dict(color="#2EC4B6", width=2.5)))
            fig.update_xaxes(title="episodes"); fig.update_yaxes(title="estimate of V(start)")
            chart(styled(fig, 200, "The estimate settles as episodes accumulate"))
            fig = go.Figure(go.Histogram(x=rets, nbinsx=40, marker_color="#9D8DF1"))
            fig.update_xaxes(title="return from the start")
            chart(styled(fig, 200, "Returns of all episodes"))
        pick = st.slider("Replay sample episode", 1, len(sample), 1, key="mc_pick")
        steps, final, end = sample[pick - 1]
        pts = [s for s, _, _ in steps] + [final]
        chart(grid_animation(n, traps, pts, f"Episode {pick}: {len(steps)} steps, ended at {end}"))
    else:
        c = st.columns(2)
        eps = c[0].slider("Exploration ε", 0.01, 0.5, 0.2, 0.01, key="mcc_eps")
        episodes = c[1].select_slider("Training episodes", [300, 1000, 2000, 4000], value=2000, key="mcc_eps_n")
        Q, greedy, curve, path, end = run_mc_control(n, tuple(traps), episodes, gamma_mc, eps, 5)
        m = st.columns(3)
        m[0].metric("Success rate, last 50 episodes", f"{curve[-1][1]:.0%}" if curve else "–")
        m[1].metric("Greedy policy result", end)
        m[2].metric("Greedy path length", f"{len(path) - 1} steps")
        a, b = st.columns(2)
        with a:
            vmax = Q.max(axis=2)
            txt = np.vectorize(lambda a: PG.ARROWS[a])(greedy).astype(object)
            for t in traps:
                txt[t] = "✕"
            txt[n - 1, n - 1] = "G"
            fig = go.Figure(go.Heatmap(z=vmax, text=txt, texttemplate="%{text}", textfont=dict(size=18), xgap=3, ygap=3,
                                       colorscale=[[0, "#FF6B6B"], [0.5, "#1B2733"], [1, "#2EC4B6"]], colorbar=dict(title="max Q")))
            fig.update_yaxes(autorange="reversed", title="X"); fig.update_xaxes(title="Y")
            chart(styled(fig, 430, "Learned policy (arrows) and its values"))
        with b:
            if curve:
                fig = go.Figure(go.Scatter(x=[c_[0] for c_ in curve], y=[c_[1] for c_ in curve], line=dict(color="#2EC4B6", width=2.5), fill="tozeroy", fillcolor="rgba(46,196,182,0.15)"))
                fig.update_yaxes(range=[0, 1.05], tickformat=".0%", title="episodes reaching G"); fig.update_xaxes(title="training episodes")
                chart(styled(fig, 430, "Learning curve (50-episode windows)"))
        chart(grid_animation(n, traps, path, f"Following the learned greedy policy: ended at {end}"))
    show_code('''class MonteCarloPolicySearch:
    def __init__(self, env, policy, gamma=0.99):
        self.env = env; self.policy = policy; self.gamma = gamma

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        while True:
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def evaluate_policy(self, num_episodes=1000):
        returns = []
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
            returns.append(G)
        return np.mean(returns)''')


def hj_contour(func, box, minima, log, probes, x0, title, zlabel="f", transform=None):
    x_lo, x_hi, y_lo, y_hi = box
    gx, gy = np.linspace(x_lo, x_hi, 150), np.linspace(y_lo, y_hi, 150)
    XX, YY = np.meshgrid(gx, gy)
    Z = func([XX, YY])
    Z = transform(Z) if transform else Z
    path_x = [x0[0]] + [r["x"][0] for r in log]
    path_y = [x0[1]] + [r["x"][1] for r in log]
    cont = go.Contour(x=gx, y=gy, z=Z, colorscale=[[0, "#0E3B43"], [0.5, "#2A5D6B"], [1, "#E8A15B"]], showscale=False,
                      line=dict(width=0.5, color="rgba(230,237,243,0.25)"), hoverinfo="skip")
    mins = go.Scatter(x=[m[0] for m in minima], y=[m[1] for m in minima], mode="markers",
                      marker=dict(symbol="star", size=15, color="#FFFFFF", line=dict(color="#0F1720", width=1)))
    pr = go.Scatter(x=probes[:, 0] if len(probes) else [], y=probes[:, 1] if len(probes) else [], mode="markers",
                    marker=dict(size=4, color="rgba(230,237,243,0.35)"))

    def at(i):
        rec = log[i]
        b, d = rec["base"], rec["step"]
        cross = go.Scatter(x=[b[0] - d, b[0] + d, None, b[0], b[0]], y=[b[1], b[1], None, b[1] - d, b[1] + d],
                           mode="lines", line=dict(color="#FFB347", width=3))
        return [cont, pr, mins, go.Scatter(x=path_x[: i + 2], y=path_y[: i + 2], mode="lines+markers",
                                            line=dict(color="#2EC4B6", width=2), marker=dict(size=6)), cross,
                go.Scatter(x=[rec["x"][0]], y=[rec["x"][1]], mode="markers", marker=dict(size=15, color="#FFB347", line=dict(color="#0F1720", width=2)))]

    keep = list(range(len(log))) if len(log) <= 120 else sorted(set(np.linspace(0, len(log) - 1, 120).astype(int).tolist()))
    frames = [go.Frame(data=at(i), name=str(i), layout=go.Layout(title=dict(text=f"Iteration {i + 1}: {log[i]['move']}, step {log[i]['step']:.3g}"))) for i in keep]
    fig = go.Figure(data=at(len(log) - 1), frames=frames)
    styled(fig, 600, title)
    fig.update_layout(showlegend=False, margin=dict(t=70, b=80, l=10, r=10),
                      xaxis=dict(range=[x_lo, x_hi], constrain="domain", showgrid=False),
                      yaxis=dict(range=[y_lo, y_hi], scaleanchor="x", constrain="domain", showgrid=False),
                      updatemenus=[dict(type="buttons", x=1, y=1.02, xanchor="right", yanchor="bottom", direction="left", showactive=False,
                                        buttons=[dict(label="▶ Replay", method="animate", args=[None, dict(frame=dict(duration=240, redraw=True), fromcurrent=False, transition=dict(duration=0))]),
                                                 dict(label="⏸", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
                      sliders=[dict(x=0, len=1, y=-0.1, yanchor="top", currentvalue=dict(prefix="Iteration "),
                                    steps=[dict(label=str(i + 1), method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i in keep])])
    return fig


def page_hj():
    st.title("🧭 Hooke-Jeeves pattern search")
    callout("A derivative-free optimiser: probe each coordinate by ± step (the orange cross), jump further in any "
            "direction that helped (the <b>pattern move</b>), and halve the step when nothing helps. Try it on test "
            "functions, or let it <b>tune the robot MDP itself</b>.")
    mode = st.radio("Use it on", ["Test functions", "Tune the robot MDP"], horizontal=True, key="hj_mode")
    if mode == "Test functions":
        c = st.columns(3)
        name = c[0].selectbox("Function", list(PG.TEST_FUNCTIONS), key="hj_fn")
        func, box, minima = PG.TEST_FUNCTIONS[name]
        default = (1.0, 1.0) if "notebook" in name else (box[0] + 0.25 * (box[1] - box[0]), box[2] + 0.8 * (box[3] - box[2]))
        x0 = c[1].slider("Start x", float(box[0]), float(box[1]), float(default[0]), 0.1, key=f"hj_x_{name}")
        y0 = c[2].slider("Start y", float(box[2]), float(box[3]), float(default[1]), 0.1, key=f"hj_y_{name}")
        c = st.columns(4)
        step = c[0].slider("Initial step", 0.05, 2.0, 0.5, 0.05, key="hj_step")
        eps = float(c[1].select_slider("Stop when step <", ["1e-02", "1e-03", "1e-04", "1e-06", "1e-08"], value="1e-06", key="hj_eps"))
        iters = c[2].slider("Max iterations", 10, 3000, 1000, 10, key="hj_iters")
        verify = c[3].checkbox("Check the pattern move first", value=True, key="hj_verify",
                               help="Untick to run exactly like the notebook, which always keeps the pattern move.")
        x, log, probes, calls = PG.hooke_jeeves(func, [x0, y0], step, eps, iters, verify)
        m = st.columns(4)
        m[0].metric("Optimized parameters", f"({x[0]:.4f}, {x[1]:.4f})")
        m[1].metric("f(x)", f"{func(x):.2e}")
        m[2].metric("Iterations / evaluations", f"{len(log)} / {calls}")
        m[3].metric("Distance to nearest minimum", f"{min(np.hypot(x[0] - a, x[1] - b) for a, b in minima):.2e}")
        if not verify and len(log) >= iters and log[-1]["step"] > eps:
            st.warning("The notebook version used every iteration without shrinking its step: the pattern move keeps "
                       "overshooting, so the search bounces back and forth. Tick 'Check the pattern move first'.")
        if not log:
            st.info("The step is already below the stopping size, so there is nothing to do.")
            return
        chart(hj_contour(func, box, minima, log, probes, (x0, y0), "Search path", transform=lambda Z: np.log10(Z - Z.min() + 1)))
        hist = pd.DataFrame([{"iteration": r["iter"], "f": r["f"], "step": r["step"], "move": r["move"]} for r in log])
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Scatter(x=hist["iteration"], y=np.maximum(hist["f"] - min(0.0, hist["f"].min()), 1e-16), line=dict(color="#2EC4B6", width=2.5)))
            fig.update_yaxes(type="log")
            chart(styled(fig, 280, "Objective value"))
        with b:
            fig = go.Figure(go.Scatter(x=hist["iteration"], y=hist["step"], line=dict(color="#FFB347", width=2.5), line_shape="hv"))
            fig.add_hline(y=eps, line_dash="dash", line_color="#FF6B6B")
            fig.update_yaxes(type="log")
            chart(styled(fig, 280, "Step size (halves when stuck)"))
        with st.expander("Iteration log"):
            table(hist.round(6), height=300)
    else:
        st.markdown("Hooke-Jeeves searches over the **discount factor γ** and the **target wall distance** to make the "
                    "optimal policy agree as often as possible with what the real robot did. Every evaluation "
                    "rebuilds the rewards and re-runs value iteration.")
        c = st.columns(4)
        g0 = c[0].slider("Start γ", 0.5, 0.99, float(gamma), 0.01, key="tune_g0")
        t0 = c[1].slider("Start target (m)", 0.3, 1.5, float(reward.target_left), 0.05, key="tune_t0")
        step = c[2].slider("Initial step", 0.02, 0.3, 0.1, 0.01, key="tune_step")
        iters = c[3].slider("Max iterations", 5, 80, 40, 5, key="tune_iters")
        tuner = PG.RobotTuner(df, used, cuts, reward, smoothing)
        x, log, probes, calls = PG.hooke_jeeves(lambda v: -tuner.agreement(v[0], v[1]), [g0, t0], step, 0.004, iters, True)
        gb, tb = tuner.clip(*x)
        start_score, best_score = tuner.agreement(g0, t0), tuner.agreement(gb, tb)
        m = st.columns(4)
        m[0].metric("Best γ", f"{gb:.3f}")
        m[1].metric("Best target distance", f"{tb:.2f} m")
        m[2].metric("Agreement with the log", f"{best_score:.1%}", f"{(best_score - start_score) * 100:+.1f} pts vs start")
        m[3].metric("Distinct MDPs solved", tuner.calls)
        gs, ts, Z = tuning_landscape(df, tuple(used), cuts_items, reward_items, smoothing)
        land = go.Contour(x=gs, y=ts, z=Z, colorscale=[[0, "#0E3B43"], [0.5, "#2A5D6B"], [1, "#E8A15B"]],
                          colorbar=dict(title="agreement", tickformat=".0%"), line=dict(width=0.5, color="rgba(230,237,243,0.25)"))
        clipped = [tuner.clip(*r["x"]) for r in log]
        fig = go.Figure([land, go.Scatter(x=[g0] + [p[0] for p in clipped], y=[t0] + [p[1] for p in clipped], mode="lines+markers",
                                          line=dict(color="#FFFFFF", width=2), marker=dict(size=7, color="#FFB347"))])
        fig.update_xaxes(title="discount factor γ"); fig.update_yaxes(title="target left-wall distance (m)")
        chart(styled(fig, 520, "Agreement with the logged moves, and the Hooke-Jeeves path"))

        def apply():
            st.session_state["gamma"] = round(gb, 2)
            st.session_state["target_left"] = round(round(tb / 0.05) * 0.05, 2)

        st.button("Use these settings in the sidebar", on_click=apply, key="tune_apply")
        st.caption("Agreement is a step-like objective (a policy either matches a move or it doesn't), which is exactly "
                   "where a derivative-free method like Hooke-Jeeves is useful. It finds a local optimum from its start.")
    show_code('''def hooke_jeeves(func, x0, step_size=0.5, epsilon=1e-6, max_iter=1000):
    x = np.array(x0)
    n = len(x)
    delta = step_size
    iter_count = 0

    def explore(x, delta):
        for i in range(n):
            f_val = func(x)
            x[i] += delta
            if func(x) < f_val:
                continue
            x[i] -= 2 * delta
            if func(x) < f_val:
                continue
            x[i] += delta
        return x

    while delta > epsilon and iter_count < max_iter:
        iter_count += 1
        x_old = np.copy(x)
        x = explore(x, delta)
        if np.array_equal(x, x_old):
            delta /= 2
        else:
            x = x + (x - x_old)
    return x

def objective_function(x):
    return x[0]**2 + x[1]**2
print("Optimized parameters:", hooke_jeeves(objective_function, [1.0, 1.0]))''')


def page_results():
    st.title("📊 Data & results")
    headline()
    tabs = st.tabs(["Sensors", "MDP model", "Value function & CSV", "Policy check"])
    with tabs[0]:
        i = st.slider("Log reading", 0, len(df) - 1, min(1500, len(df) - 1), key="res_row")
        row = df.iloc[i]
        a, b = st.columns([1, 1])
        with a:
            theta = [E.ANGLE[k] for k in range(1, 25)] + [E.ANGLE[1]]
            r = [row[f"US{k}"] for k in range(1, 25)] + [row["US1"]]
            fig = go.Figure(go.Scatterpolar(r=r, theta=theta, mode="lines+markers", fill="toself",
                                            line=dict(color="#2EC4B6"), fillcolor="rgba(46,196,182,0.2)",
                                            text=[f"US{k}" for k in range(1, 25)] + ["US1"], hovertemplate="%{text}: %{r:.2f} m<extra></extra>"))
            for d, sensors in E.ARCS.items():
                fig.add_trace(go.Scatterpolar(r=[row[f"US{k}"] for k in sensors], theta=[E.ANGLE[k] for k in sensors], mode="markers",
                                              marker=dict(size=10, color=WB.ARC_COLORS[d]), name=d))
            fig.update_layout(polar=dict(bgcolor="#101A24", radialaxis=dict(range=[0, 5.1], gridcolor=GRIDC), angularaxis=dict(direction="clockwise", rotation=90, gridcolor=GRIDC)))
            chart(styled(fig, 430, "The 24 ultrasound readings (coloured = the four arcs)"))
        with b:
            st.markdown(f"**Logged move:** {row['Class']}  \n**State:** `{dfs.iloc[i]['State']}`  \n**Optimal move:** {best_map.get(dfs.iloc[i]['State'])}")
            table(pd.DataFrame({"Distance": E.DIST, "Sensors": [", ".join(f"US{k}" for k in E.ARCS[d]) for d in E.DIST],
                                "Metres": [round(float(row[d]), 3) for d in E.DIST]}))
            dsel = st.selectbox("Distribution by move", E.DIST, index=1, key="res_dist")
            fig = go.Figure([go.Violin(y=df.loc[df["Class"] == m, dsel], name=m.replace("-", " "), line_color=MCOL[m], box_visible=True, meanline_visible=True) for m in E.MOVES])
            chart(styled(fig, 300))
    with tabs[1]:
        st.markdown("**Levels**")
        table(pd.DataFrame([{"Distance": d, **{nm: f"{lo} – {hi} m" for nm, lo, hi in zip(E.level_names(cuts[d]), ["0"] + [f"{c:.2f}" for c in cuts[d]], [f"{c:.2f}" for c in cuts[d]] + ["5"])}} for d in used]))
        act = st.selectbox("Transitions for move", E.MOVES, key="res_move")
        a_i = E.MOVES.index(act)
        rows_ = np.where(mdp.seen[:, a_i])[0]
        if len(rows_):
            rows_ = rows_[np.argsort(-mdp.counts[rows_, a_i].sum(1))][:22]
            cols_ = np.where(mdp.P[rows_, a_i].sum(0) > 0)[0]
            fig = go.Figure(go.Heatmap(z=mdp.P[np.ix_(rows_, [a_i], cols_)][:, 0, :], x=[mdp.states[c] for c in cols_], y=[mdp.states[r] for r in rows_],
                                       colorscale=[[0, "#101A24"], [1, "#2EC4B6"]], colorbar=dict(title="P")))
            chart(styled(fig, 560, f"P(s' | s, {act}) for the most common states"))
        rdf = pd.DataFrame(np.where(mdp.seen, mdp.R, np.nan), index=mdp.states, columns=E.MOVES)
        fig = go.Figure(go.Heatmap(z=rdf.values, x=E.MOVES, y=mdp.states, colorscale=[[0, "#FF6B6B"], [0.5, "#1B2733"], [1, "#2EC4B6"]], zmid=0, colorbar=dict(title="R")))
        chart(styled(fig, max(360, 16 * len(mdp.states)), "Rewards R(s, a) (blank = move never taken in that state)"))
    with tabs[2]:
        a, b = st.columns([1.3, 1])
        with a:
            fig = go.Figure(go.Bar(x=vtab["Optimal_Value"], y=vtab["State"], orientation="h",
                                   marker_color=[MCOL[m] for m in vtab["Optimal_Action"]]))
            fig.update_yaxes(autorange="reversed", showticklabels=len(vtab) <= 45)
            chart(styled(fig, max(380, 14 * len(vtab)), "V*(s), coloured by the optimal move"))
        with b:
            d = sol["deltas"]
            fig = go.Figure(go.Scatter(x=np.arange(1, len(d) + 1), y=np.maximum(d, 1e-16), line=dict(color="#2EC4B6", width=2.5)))
            fig.add_hline(y=tol, line_dash="dash", line_color="#FF6B6B")
            fig.update_yaxes(type="log")
            chart(styled(fig, 280, "Largest change per sweep"))
            st.metric("Value iteration vs policy iteration", f"max |ΔV| = {np.abs(pi_res['V'] - sol['V']).max():.1e}",
                      f"same policy in {np.mean(pi_res['pi'] == sol['pi']):.0%} of states", delta_color="off")
        st.markdown("**optimal_value_function.csv**")
        table(vtab, height=380)
        st.download_button("⬇ Download optimal_value_function.csv", vtab.to_csv(index=False).encode(), "optimal_value_function.csv", "text/csv")
        if csv_saved:
            st.caption(f"Also saved as {CSV_OUT.name} next to app.py.")
    with tabs[3]:
        cm = pd.crosstab(dfs["Class"], dfs["State"].map(best_map)).reindex(index=E.MOVES, columns=E.MOVES, fill_value=0)
        fig = go.Figure(go.Heatmap(z=cm.values, x=E.MOVES, y=E.MOVES, text=cm.values, texttemplate="%{text}",
                                   colorscale=[[0, "#101A24"], [1, "#5AA9E6"]], colorbar=dict(title="readings")))
        fig.update_xaxes(title="optimal move"); fig.update_yaxes(title="logged move", autorange="reversed")
        chart(styled(fig, 440, f"Logged vs optimal move ({agree:.1%} agree)"))
        st.caption("The logged moves came from the robot's own controller, not from this reward, so they won't match "
                   "everywhere. Where they differ, the MDP expects a different move to pay off more under the shaping "
                   "set in the sidebar. The Hooke-Jeeves page can tune γ and the target to raise the agreement.")


ROUTES = {PAGES[0]: page_overview, PAGES[1]: page_simulator, PAGES[2]: page_vi, PAGES[3]: page_mdp,
          PAGES[4]: page_adp, PAGES[5]: page_mc, PAGES[6]: page_hj, PAGES[7]: page_results}
try:
    ROUTES[page]()
except Exception as exc:                                   # keep the app usable whatever happens on one page
    st.error(f"This page hit a problem: {exc}")
    st.exception(exc)
