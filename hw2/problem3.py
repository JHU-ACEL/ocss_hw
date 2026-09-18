import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import matplotlib.pyplot as plt

    from solvers import PDIPSolver
    from ocp import OCPSolver
    from qp_tester import PDIPTester

    import marimo as mo

    return OCPSolver, jnp, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3b [15 Points]

    Having implemented the primal-dual interior point algorithm for a generic convex quadratic program, we will now apply it to solve the optimal control problem:

    \begin{align}
        \underset{q_\text{0:N}, u_\text{0:N-1}}{\textrm{minimize}} \,\, & q_N^T Q_f q_N + \sum_{k=0}^{N-1} u_k^T R u_k + q_k ^T Q q_k \nonumber \\
        \textrm{subject to}
        \,\,& q_{k+1} = A_\text{sc} q_k + B_\text{sc} u_k, \quad k = 0, \ldots, N-1\nonumber \\
        \,\,&  q_0 = q_\text{init}, \nonumber\\
        \,\,& q_\text{min} \leq q_k \leq q_\text{max}, \quad k = 1, \ldots, N,\nonumber\\
        \,\,& u_\text{min} \leq u_k \leq u_\text{max}, \quad k = 0, \ldots, N-1,\nonumber
    \end{align}

    where $q_k = (p_k, v_k) \in \mathbb{R}^4$ includes the spacecraft position $p_k \in \mathbb{R}^2$ and velocity $v_k \in \mathbb{R}^2$ as resolve in the inertial frame.
    The spacecraft control $u_k = (u_x, u_y) \in \mathbb{R}^2$ consists of the acceleration applied in the inertial frame.
    Lower and upper bounds are implemented for the state ($q_\text{min}$ and $q_\text{max}$, respectively) and control ($u_\text{min}$ and $u_\text{max}$, respectively) at each time step.
    Finally, the objective function seeks to minimize the control effort expended to drive the system state to the origin.

    Consequently, we use the 2D double integrator dynamics:

    $A_\text{sc} = \begin{pmatrix} I_{2\times 2} & \Delta h I_{2\times 2}\\ 0_{2\times 2} & I_{2\times 2} \end{pmatrix}$

    $B_\text{sc} = \begin{pmatrix} \frac{1}{2}\Delta h^2 I_{2\times 2}\\ \Delta h I_{2\times 2} \end{pmatrix}$

    where $\Delta h$ is the discretization time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to leverage the primal-dual interior point solver we implemented, we must construct convert this double integrator problem into standard form:

    \begin{align}
        \underset{x}{\textrm{minimize}} \,\, & \frac{1}{2}x^TPx + p^Tx \nonumber \\
        \textrm{subject to}
        \,\,& Ax=b\nonumber \\
        \,\,&  Gx \leq h \nonumber\\
    \end{align}

    First determine the vector $x \in \mathbb{R}^n$ corresponding to $q_k$ and $u_k$, i.e.,

    $x = (q_0, u_0, q_1, u_1, \ldots, u_{N-1}, q_N).$

    Now implement the following methods in the `OCP` class:

    1. `compute_cost_term`:

    2. `compute_equality_matrix`: use the double integrator $A_\text{sc}$ and $B_\text{sc}$ matrices to construct the equality constraint matrix $A$.

    3. `compute_equality_bc`: construct the equality constraint vector $b$.

    4. `compute_ineq_con`: use the upper/lower bounds for state ($q_\text{min}$ and $q_\text{max}$) and control ($u_\text{min}$ and $u_\text{max}$) to construct the inequality constraint matrix $G$ and vector $h$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now test out your implementation. The following has a simple set of parameters for the number of time steps $N$ and and time step $\Delta h$.
    """)
    return


@app.cell
def _(jnp):
    nx = 4
    nu = 2
    N = 1

    dh = 0.5
    Ak = jnp.vstack((
        jnp.hstack((jnp.eye(2), dh*jnp.eye(2))),
        jnp.hstack((jnp.zeros((2,2)), jnp.eye(2)))
    ))
    Bk = jnp.vstack((
        0.5*dh**2*jnp.eye(2),
        dh*jnp.eye(2)
    ))
    return Ak, Bk, N, nu, nx


@app.cell
def _(OCPSolver):
    ocp_solver = OCPSolver()
    return (ocp_solver,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define $x_\text{init}$ as the initial position and velocity of the spacecraft in 2D
    """)
    return


@app.cell
def _(jnp):
    x_init = jnp.array([9.4, 7.2, -0.5, 1.0])
    return (x_init,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define cost matrices $Q_\text{mat}$ and $R_\text{mat}$.

    **Note** that these matrices are not the same as the $Q$ and $R$ matrices above! These $Q_\text{mat}$ and $R_\text{mat}$ are the matrices used to compute stagewise cost, i.e., $x_k^T Q_\text{math} x_k$.
    """)
    return


@app.cell
def _(jnp, nu, nx):
    Q_mat = 2*jnp.eye(nx)
    R_mat = jnp.eye(nu)
    return Q_mat, R_mat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instantiate upper and lower bounds for state and control
    """)
    return


@app.cell
def _(jnp):
    x_min = jnp.array([-10.0, -10, -5, -5])
    x_max = jnp.array([10.0, 10, 5, 5])

    u_min = jnp.array([-15.0, -15])
    u_max = jnp.array([15.0, 15])
    return u_max, u_min, x_max, x_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Once you've defined all of the methods in the `OCPSolver` class, activate the cell below and instantiate the QP corresponding to the optimal control problem at the top of the page. For your homework submission, plot a few of the trajectories you generate starting from different initial conditions.
    """)
    return


@app.cell(disabled=True)
def _(
    Ak,
    Bk,
    N,
    Q_mat,
    R_mat,
    jnp,
    nu,
    nx,
    ocp_solver,
    u_max,
    u_min,
    x_init,
    x_max,
    x_min,
):
    A_eq = ocp_solver.compute_equality_matrix(Ak, Bk, N)
    b_eq = ocp_solver.compute_equality_bc(x_init, Ak, N)

    G_ineq, h_ineq = ocp_solver.compute_ineq_con(x_min, x_max, u_min, u_max, N)

    P = ocp_solver.compute_cost_term(Q_mat, R_mat, N)
    p = jnp.zeros(N*(nx+nu))
    return


@app.cell(disabled=True)
def _(PDIPSolver, P, p, A_eq, b_eq, G_ineq, h_ineq):
    
    solver = PDIPSolver()
    solver.init_problem(2*P, p, A_eq, b_eq, G_ineq, h_ineq)
    costs = solver.solve_qp(verbose=True)
    
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
