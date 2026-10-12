import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this problem, we will be implementing the algorithm from [Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem](https://ieeexplore.ieee.org/document/6428631).

    The solution procedure will consist of two steps:

    1. Solve `Problem 3`, the `Convex Relaxed Minimum Landing Error Problem`.
    2. Store this solution to `Problem 3` and use it to construct and solve `Problem 4`, the `Convex Relaxed Minimum Fuel Problem`.

    As a part of this, you will fill out the methods in the `PDGSolver` class:

    1. `__init__`: the class constructor contains all of the parameters you'll need to define the constraints and the numerical values are from `Section IV` of the paper.
    2. `construct_dynamics_matrices`: fill out the expressions for the continuous time dynamics matrices $A$ and $B$ as given in `Equation 2` of the paper.
    3. `construct_constraints` is a shared function that is used to solve both `Problem 3` and `Problem 4`. This will correspond with `Equations 5, 7, 8, 9, 17, 18, 19` from the paper (i.e., all of the constraints for `Problem 3`), but some of these constraints will have to rewritten using change-of-variables techniques (`Equations 33 - 36`).
    4. `solve_minimum_landing_error`: this method solves `Problem 3` by instantiating the necessary `cvxpy` variables, calling `construct_constraints` to construct the necessary constraints array, and then solving the problem.
    5. `solve_minimum_fuel_problem`: this method solves `Problem 4` and uses the output `solve_minimum_landing_error` to define $d_\text{P3}^*$.


    A few further notes:

    - For the dynamics constraints, you are free to use any discretization scheme, but forward Euler should suffice.
    - As discussed in class, the $\frac{\mathbf{T}_c(t)}{m(t)}$ expression in the dynamics is non-convex with respect to $m(t)$ and this necessitates a change-of-variables. Use the reformulation described in `Section III.A` to introduce $z$ and $\sigma$ as additional decision variables and include the constraints from `Equations 33 - 36`.
    - Note that the constraint in `Equation 5` of $\mathbf{x}(t) \in \mathbf{X}$ simply consists of two separate constraints for (1) the maximum speed and (2) the glideslope constraint detailed in `Equation 12`.
    """)
    return


@app.cell
def _():
    import numpy as np
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import marimo as mo

    from pdg import PDGSolver

    return PDGSolver, mo, plt


@app.cell(disabled=True)
def _(PDGSolver):
    pdg_solver = PDGSolver()
    return (pdg_solver,)


@app.cell
def _():
    tf = 45  # end time (s)
    dt = 1   # time interval (s)
    N = int(tf / dt)
    return N, dt


@app.cell
def _(N, dt, pdg_solver):
    x, u, status, cost = pdg_solver.solve_minimum_landing_error(dt, N)
    dP3 = x[1:3, N-1]
    x, u, status, cost = pdg_solver.solve_minimum_fuel_problem(dt, N, dP3)
    print(cost)
    return u, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Once you're done with the implementation, plot the trajectory of the rocket in 3D and the control trajectory for each of the thrusters as well.
    """)
    return


@app.cell
def _(plt, x):
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(top=1.1, bottom=-.1)
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(x[0,:], x[1,:], x[2,:])
    plt.show()
    return


@app.cell
def _(plt, x):
    plt.quiver(x[0,:], x[1,:], x[3,:], x[4,:])
    return


@app.cell
def _(plt, u):
    plt.plot(u[0,:], label='T_x')
    plt.plot(u[1,:], linestyle='--', label='T_y')
    plt.plot(u[2,:], linestyle='-.', label='T_z')
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
