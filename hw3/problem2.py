import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    import cvxpy as cp
    import numpy as np

    import matplotlib.pyplot as plt

    from admm import ADMMSolver

    from abc import ABC, abstractmethod

    from jax import random

    return ADMMSolver, cp, jnp, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Alternating Direction Method of Multipliers

    Consider the convex quadratic program:

    \begin{align}
        \underset{x}{\textrm{minimize}} \,\, & \frac{1}{2}x^TQx + q^Tx \nonumber \\
        \textrm{subject to}
        \,\,& Ax=b\nonumber \\
        \,\,&  Gx \leq h \nonumber\\
    \end{align}

    where $x \in \mathbb{R}^n$, $Q \in \mathbf{S}^n_+$, and $q \in \mathbb{R}^n$.

    The linear equality constraints are given by $A \in \mathbb{R}^{n_\text{eq} \times n}$ and $b \in \mathbb{R}^{n_\text{eq}}$.

    The linear inequality constraints are $G \in \mathbb{R}^{n_\text{ineq} \times n}$ and $h \in \mathbb{R}^{n_\text{ineq}}$.

    Previously, we solved this quadratic program using interior-point methods inspired by, e.g., [`CVXGEN`](https://web.stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf). Here, we will instead employ the approach of alternating direction method of multipliers (ADMM):
    1. [`Distributed Optimization and Statistical
    Learning via the Alternating Direction
    Method of Multipliers`](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
    """)
    return


@app.cell
def _(jnp):
    nx = 8

    Q = jnp.diag(jnp.array([2, 3, 3, 4, 5, 8, 9., 10]))
    q = jnp.array([-2, 3, 3, -14., -2., 2, 2, 17])

    # Set equality constraints
    n_eq = 2
    A = jnp.zeros((n_eq, nx))
    A = A.at[0, :].set(jnp.ones(nx))
    A = A.at[1, :].set(jnp.array([1, -1, 1, -1, 0, 1, 0, 0]))
    b = jnp.array([-120, -14])

    # Set inequality constraints
    n_ineq = 3
    G = jnp.array([[1., -1, -1, 1, 1, 0, 0, 0], [1, -18, 0, 0, 1, 1, -2, 3], [24, -1, 1., -12., 3, 4, 4, 14.]])
    h = jnp.array([-65, -17, -100.0])
    return A, G, Q, b, h, n_eq, nx, q


@app.cell
def _(A, G, Q, b, cp, h, jnp, n_eq, np, nx, q):
    assert jnp.linalg.matrix_rank(A) == n_eq

    # Define and solve the CVXPY problem.
    x_cvx = cp.Variable(nx)
    cons = []
    cons += [np.asarray(A) @ x_cvx == np.asarray(b)]
    cons += [np.asarray(G) @ x_cvx <= np.asarray(h)]
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x_cvx, np.asarray(Q)) + np.asarray(q).T @ x_cvx),
                     cons)
    feasible = prob.solve()

    if feasible:
        print (f'Optimal solution is {x_cvx.value}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Test sensitivity of the solver to $\rho$ using the slider below
    """)
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(start=0.01, stop=1, step=0.01)
    rho
    return (rho,)


@app.cell(disabled=True)
def _(A, ADMMSolver, G, Q, b, h, q, rho):
    max_iter = 200
    eps_tol = 1e-3

    solver = ADMMSolver(rho=rho.value, max_iter=max_iter, tol=eps_tol)
    solver.init_problem(Q, q, A, b, G, h)
    x_admm, has_converged = solver.solve_qp()
    print(has_converged)
    return


if __name__ == "__main__":
    app.run()
