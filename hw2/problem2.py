import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import matplotlib.pyplot as plt

    from solvers import PDIPSolver
    from qp_tester import PDIPTester

    import marimo as mo

    return PDIPTester, jnp, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2b [15 Points]

    In this homework, we will solve quadratic programs of the form:

    \begin{align}
        \underset{x}{\textrm{minimize}} \,\, & \frac{1}{2}x^TPx + p^Tx \nonumber \\
        \textrm{subject to}
        \,\,& Ax=b\nonumber \\
        \,\,&  Gx \leq h \nonumber
    \end{align}

    where $x \in \mathbb{R}^n$, $P \in \mathbf{S}^n_+$, and $p \in \mathbb{R}^n$.

    The linear equality constraints are given by $A \in \mathbb{R}^{n_\text{eq} \times n}$ and $b \in \mathbb{R}^{n_\text{eq}}$.

    The linear inequality constraints are $G \in \mathbb{R}^{n_\text{ineq} \times n}$ and $h \in \mathbb{R}^{n_\text{ineq}}$.

    To solve this problem, we will be implementing the solver detailed in Section 5 of the [`CVXGEN: a code generator for embedded convex
    optimization`](https://web.stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf) paper.

    Notably, this solver is the same one SpaceX uses to land its rockets! [1](https://ee.stanford.edu/news/2021/jan/stephen-boyd-cvxgen-guides-spacex-falcon), [2](https://discourse-data.ams3.cdn.digitaloceanspaces.com/original/3X/1/1/11dbbcbc9f31e7323f4cbb433cb4a24c81cdad43.pdf)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will do so by implementing the following functions in the `Solver` class:

    1. `init_soln`:
    Solve the following system given in Section 5.2 of the `CVXGEN` paper:
    $\begin{pmatrix} P & G^T & A^T\\ G & -I & 0\\ A & 0 & 0 \end{pmatrix}\begin{pmatrix}x\\ z\\ y \end{pmatrix} = \begin{pmatrix} -p\\ h\\ b \end{pmatrix}$

    Use the solution to this system to initialize $x^{(0)}$ and $z^{(0)}$ and then solve for $s^{(0)}$ and $z^{(0)}$ using the routine described in the paper to ensure non-negativity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. `compute_residuals`: Compute the residuals that we are trying to drive to zero:

    $\begin{pmatrix}r_1\\ r_2\\ r_3\\ r_4\end{pmatrix} = \begin{pmatrix} -(A^T \bar{y} + G^T \bar{z} + P \bar{x} + p)\\ -\text{diag}(\bar{s}) \bar{z}\\ -(G\bar{x}+\bar{s} - h)\\ -(A \bar{x} - b)\end{pmatrix}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. `solve_kkt_system`: Solve the KKT system

    $\begin{pmatrix} P & 0 & G^T & A^T\\ 0 & \text{diag}(\bar{z}) & \text{diag}(\bar{s}) & 0\\ G & I & 0 & 0\\ A & 0 & 0 & 0 \end{pmatrix} \begin{pmatrix} \Delta x\\ \Delta s\\ \Delta z\\ \Delta y \end{pmatrix} = \begin{pmatrix} u_1\\ u_2\\ u_3\\ u_4 \end{pmatrix}$

    **Note**: This `solve_kkt_system` method will be used twice - first to compute $(\Delta x^\text{aff}, \Delta s^\text{aff}, \Delta z^\text{aff}, \Delta y^\text{aff})$ and second to compute $(\Delta x^\text{cc}, \Delta s^\text{cc}, \Delta z^\text{cc}, \Delta y^\text{cc})$. Accordingly, make sure to pass the correct right hand side values $(u_1, u_2, u_3, u_4)$ for each step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4. `compute_centering_plus_corrector`:

    Compute the centering-plus-corrector direction steps:

    $\begin{pmatrix} P & 0 & G^T & A^T\\ 0 & \text{diag}(\bar{z}) & \text{diag}(\bar{s}) & 0\\ G & I & 0 & 0\\ A & 0 & 0 & 0 \end{pmatrix} \begin{pmatrix} \Delta x^\text{cc}\\ \Delta s^\text{cc}\\ \Delta z^\text{cc}\\ \Delta y^\text{cc} \end{pmatrix} = \begin{pmatrix} 0 \\ \sigma\mu\overrightarrow{1} - \text{diag}(\Delta s^\text{aff})\Delta z^\text{aff}\\0\\0 \end{pmatrix}$

    where

    $\sigma = ( \frac{(\bar{s} + \alpha \Delta s^\text{aff})^T (\bar{z} + \alpha \Delta z^\text{aff}) }{\bar{s}^T\bar{z}} )^3$

    and

    $\mu = \frac{\bar{s}^T\bar{z}}{p}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    5. `solve_qp`: This function will use the preceding functions to run through the QP solution routine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Once you're done implementing these class methods, solve the following "simple" QP
    """)
    return


@app.cell(disabled=True)
def _(jnp):
    P = jnp.array([[2.0, 0, 0, 0], [0, 2., 0, 0], [0, 0, 2., 0], [0., 0, 0, 1.]])
    p = jnp.array([1.0, -2., 1.0, 2.])

    A_eq = jnp.array([[1, 1, 0, 0], [1, 0, -1, 0]])
    b_eq = jnp.array([8., -7.0])

    G_ineq = jnp.array([[1., 0, 0, 0],
                        [-1, 0., 0, 0],
                        [0., 1., 0., 0],
                       ])
    h_ineq = jnp.array([15.0, -15, 15.])
    return A_eq, G_ineq, P, b_eq, h_ineq, p


@app.cell(disabled=True)
def _(A_eq, G_ineq, P, PDIPTester, b_eq, h_ineq, p):
    tester = PDIPTester(P, p, A_eq, b_eq, G_ineq, h_ineq)
    tester.compare_solutions()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
