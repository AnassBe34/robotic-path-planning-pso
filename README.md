# Robotic Path Planning by PSO

Compact Python project that finds collision‑free, length‑efficient cubic‑spline paths in 2D environments using Particle Swarm Optimization (PSO). Includes a matplotlib desktop demo and an interactive Streamlit UI for parameter tuning and visualization.

## Features
- Real-valued PSO optimizer for continuous control points.
- Spline-based path representation (cubic splines).
- 2D environment model with circular obstacles and robot radius handling.
- Cost model combining path length and collision/violation penalties.
- Live plotting utilities (matplotlib) and an interactive Streamlit app.

## Repository structure
- `main.py` — Desktop demo: runs PSO and renders matplotlib updates.
- `streamlit.py` — Interactive Streamlit application for parameter tuning, obstacle editing and live visualization.
- `pso.py` — Particle Swarm Optimization implementation and callback support.
- `path_planning/`
  - `__init__.py` — Package exports.
  - `environment.py` — Environment and obstacle models (`Environment`, `Obstacle`).
  - `solution.py` — Spline path class (`SplinePath`) and utilities.
  - `cost.py` — Cost function and factory (`PathPlanningCost`, `EnvCostFunction`).
  - `plots.py` — Plotting helpers (`plot_environment`, `plot_path`, `update_path`).

## Quick start (Windows)
1. Clone the repo and open a terminal in the project folder
2. Install required packages:
   ```bash
     pip install numpy matplotlib streamlit
4. Run the interactive Streamlit app:
    ```bash
    streamlit run streamlit.py

## Usage notes
- Use the Streamlit UI to edit environment size, start/goal, robot radius, PSO parameters and obstacles. Click "Run PSO" to start optimization and watch live updates.
- main.py runs a similar demo using matplotlib live updates (no Streamlit).

## Tips
- Tune PSO parameters (population size, c1/c2, inertia) to balance exploration vs. exploitation.
- Increase resolution in the cost function for finer collision checks (tradeoff: slower evaluations).
