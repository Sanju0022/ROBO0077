# WallBot Lab

Wall-following robot navigation as a Markov Decision Process, solved with value iteration,
plus the Day 5 notebook programs (MDP, ADP, Monte Carlo, Hooke-Jeeves).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Or open `WallBot_Lab.ipynb` and run all cells: it writes these files and launches the app
(works in Jupyter, VS Code and Google Colab).

## Files
| File | Contents |
|---|---|
| `app.py` | Streamlit interface: sidebar settings and eight pages |
| `engine.py` | Data loading, states, smoothed transitions, shaped rewards, value iteration, policy iteration |
| `programs.py` | Notebook MDP, ADP (offline Q-learning, state aggregation), Monte Carlo (evaluation, control), Hooke-Jeeves |
| `web_bridge.py` | Turns results into JSON for the browser components |
| `web/simulator.html` | Interactive room: start, finish, walls, learned wall-follower and grid-MDP goal seeker |
| `web/vi_player.html` | Value iteration animated sweep by sweep |
| `data/sensor_readings_24.csv` | The dataset (24 ultrasound sensors + move label) |

`python engine.py` builds the MDP with the default settings and writes `optimal_value_function.csv`.

## How the robot MDP is built
| Piece | Definition |
|---|---|
| States | SD_front, SD_left, SD_right, SD_back cut at fixed thresholds in metres (default Close / Medium / Far), or quantiles |
| Actions | Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, Slight-Left-Turn |
| Transitions | Counts of consecutive readings with Laplace smoothing over next states reached from s |
| Rewards | Gaussian bonus around the target left-wall distance, penalties that grow as the front or left gets too close, forward bonus, sharp-turn cost |
| Solver | Value iteration (notebook update + threshold), cross-checked with policy iteration |
| Output | `optimal_value_function.csv` |

The simplified distances use the arcs that reproduce the dataset's official 4-sensor file:
front US11–15, left US18–20, right US5–9, back US23–24.
