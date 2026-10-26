import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework Problem: MPPI and Cross Entropy for Trajectory Generation (Dynamic Bicycle Model)

    In this problem, you will implement sampling-based trajectory optimization: **Model Predictive Path Integral (MPPI)** and **Cross Entropy Method (CEM)** for a nonlinear system, specifically the **dynamic bicycle model**.

    ## Dynamics

    The **dynamic bicycle model** describes car-like robot motion as:

    - **State:** $\mathbf{x} = [x,\, y,\, \theta,\, v,\, \delta]$
      - $x, y$: position,
      - $\theta$: heading,
      - $v$: velocity,
      - $\delta$: steering angle.

    - **Control:** $\mathbf{u} = [a,\, \dot{\delta}]$
      - $a$: acceleration,
      - $\dot{\delta}$: steering rate.

    Discrete updates (Euler):


    **Discrete-time update equations:**

    $$
    \begin{aligned}
    x_{t+1}      &= x_t + v_t \cos(\theta_t) \Delta t \\
    y_{t+1}      &= y_t + v_t \sin(\theta_t) \Delta t \\
    \theta_{t+1} &= \theta_t + \frac{v_t}{L} \tan(\delta_t) \Delta t \\
    v_{t+1}      &= v_t + a_t \Delta t \\
    \delta_{t+1} &= \delta_t + \dot{\delta}_t \Delta t
    \end{aligned}
    $$

    ---

    ## Optimization Methods

    ### Model Predictive Path Integral (MPPI)

    MPPI is a **sampling-based stochastic optimal control method**:

    - At each iteration, **sample** $K$ control sequences $\{\mathbf{u}^{(k)}\}_{k=1}^K$ by adding noise to a nominal control sequence.
    - **Roll out** each trajectory according to the system dynamics.
    - **Compute cost** $J_k$ for each sample.
    - **Weight** each control trajectory by its (exponentiated negative) cost:

      $w_k = \frac{\exp\left(-\frac{1}{\kappa} J_k\right)}{\sum_{i=1}^K \exp\left(-\frac{1}{\kappa} J_i\right)}$

      where $\kappa$ is a temperature tuning parameter.

    - **Update the nominal control sequence** by weighted average:

      $\mathbf{u}_t^{\text{new}} = \sum_{k=1}^K w_k\, \mathbf{u}_t^{(k)}$

    - Apply or rollout with the updated controls.

    ---

    ### Cross Entropy Method (CEM)

    CEM optimizes the control sequence distribution by **focusing sampling on elite solutions**:

    - At each iteration, **sample** $K$ control sequences $\{\mathbf{u}^{(k)}\}$ from the current distribution ($\mathcal{N}(\mu,\, \Sigma)$).
    - **Roll out** and **evaluate cost** for each sample.
    - **Select the best (elite) $K_e$ samples** with lowest costs.
    - **Update the distribution parameters**:
      - New mean $\mu_{\text{new}}$ = mean of elites,
      - New std $\sigma_{\text{new}}$ = std of elites.
    - Repeat for a number of iterations.

    ---

    ## Assignment

    - Implement the dynamic bicycle model step.
    - Implement MPPI and CEM.
    - Define a quadratic cost:
    $J = (\mathbf{x}_N - \mathbf{x}_{goal})^T Q_f (\mathbf{x}_N - \mathbf{x}_{goal}) + \sum_{t=0}^{N-1}  ((\mathbf{x}_t - \mathbf{x}_{goal})^T Q (\mathbf{x}_t - \mathbf{x}_{goal}) + \mathbf{u}_t^T R \mathbf{u}_t)$
    - Define an obstacle cost, so that the rover goes around the obstacles
    - **Compare** trajectories produced by both optimizers, especially as you vary the number of samples, iterations, and cost weights.
    - Run a simulation in mujuco

    ## JAX-practices
    The functions in the `MppiCemSolver` class that have the `@eqx.filter_jit` decorator on them must be JIT-compatible.
    Hint: Use `jax.lax.scan` and `jax.vmap`
    """)
    return


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from mppi_cem import MppiCemSolver

    return PatchCollection, jrandom, mo, np, plt


@app.cell
def _(jrandom, np):
    # Problem parameters
    dt = 0.1
    T = 100    # time horizon (steps)
    x0 = np.array([0., 0., 0., 2., 0.])         # [x, y, theta, v, delta]
    x_goal = np.array([6., 14., 0.0, 0., 0.])  
    key = jrandom.key(123)
    obstacles = [(np.array([2.0, 6.0]), 1),]

    # Initialize solver (experiment with different # of samples, iterations, kappa, etc).

    return obstacles, x0, x_goal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. MPPI: Run the solver and plot trajectory

    Run `solve_mppi()` below. Plot the trajectory in $(x, y)$ and the controls over time.
    """)
    return


@app.cell
def _():
    # Get trajectory states, controls
    return


@app.cell
def _(PatchCollection, obstacles, plt, states, x0, x_goal):
    # Plot MPPI traj
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    plt.plot(states[:, 0], states[:, 1], 'b-', label='Trajectory')
    plt.scatter([x0[0], x_goal[0]], [x0[1], x_goal[1]], c=['g','r'], label='Start/Goal')

    # Plot obstacles
    circles = [plt.Circle((center[0], center[1]), radius) for center,radius in obstacles]
    collection = PatchCollection(circles, facecolors='blue', alpha=0.6)
    ax1.add_collection(collection)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("MPPI: State Trajectory")
    plt.legend()
    plt.grid()
    plt.show()
    return (circles,)


@app.cell
def _(controls, plt):
    # Plot controls
    plt.figure(figsize=(6, 3))
    plt.plot(controls[:,0], label="acc")
    plt.plot(controls[:,1], label="steering rate")
    plt.xlabel("Timestep")
    plt.ylabel("Input")
    plt.title("MPPI: Control Inputs")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. CEM: Run the solver and plot trajectory

    Run `solve_cem()` below. Plot the trajectory in $(x, y)$ and the controls over time.
    """)
    return


@app.cell
def _():
    # Generate trajectory with CEM
    return


@app.cell
def _(PatchCollection, circles, obstacles, plt, states_cem, x0, x_goal):
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plt.plot(states_cem[:, 0], states_cem[:, 1], 'm-', label='CEM Trajectory')
    plt.scatter([x0[0], x_goal[0]], [x0[1], x_goal[1]], c=['g','r'], label='Start/Goal')

    # Plot obstacles
    circles2 = [plt.Circle((center[0], center[1]), radius) for center,radius in obstacles]
    collection2 = PatchCollection(circles, facecolors='blue', alpha=0.6)
    ax2.add_collection(collection2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("CEM: State Trajectory")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell
def _(controls_cem, plt):
    plt.figure(figsize=(6, 3))
    plt.plot(controls_cem[:,0], label="acc")
    plt.plot(controls_cem[:,1], label="steering rate")
    plt.xlabel("Timestep")
    plt.ylabel("Input")
    plt.title("CEM: Control Inputs")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot trajectories for both MPPI and CEM and varying key parameters
           (cost function, time horizon, number of iterations, etc.).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
