import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt

    return jax, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 4

    A core component of the optimization solvers that we will be implementing in this class is the underlying linear solve used to solve systems of the form $Ax = b$. The performance of these solvers drive the runtime, precision, and robustness of the optimization solvers themselves.

    In this problem, you will implement a version of the [preconditioned conjugate gradient (PCG) solver](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf), an iterative solver for linear systems $Ax = b$ when A is a positive semidefinite matrix (i.e., $A\in\mathbb{S}_+^n$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 4.1 [10 Points]

    Inside `hw1/pcg.py`, complete the implementation for the `PCG.solve` method following the pseudocode provided in Figure 2.5 of [this document](https://www.cfm.brown.edu/faculty/gk/AM258/Handouts/templates.pdf).

    *Note:* Your code will be evaluated under the assumption that it is `jit`-compatible. Take the time to learn more about the relevant `jax` methods here:
    - [`jax.lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html)
    - [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
    - [`jax.lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)
    """)
    return


@app.cell
def _():
    from pcg import PCG

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 4.2 [5 Points]

    Read about what the [condition number](https://en.wikipedia.org/wiki/Condition_number#Matrices) $\kappa$ of a matrix indicates and how it can affect the performance of a linear solver $Ax=b$.

    We have provided a helper function `test_problem` that can generate "random" positive definite matrices with a specified condition number. Generate a histogram comparing the $\ell_2$-distance between the `pcg.solve` and `jax.linalg.solve` solutions when you generate random matrices with condition numbers $\kappa \in \{ 10^2, 10^4, 10^6, 10^8, 10^{10}, 10^{12} \}$. Sample at least 50 random matrices for each condition number and choose matrices $A$ of at least size 25.

    Comment on what your results demonstrate and the performance between the `PCG` solver and `jax`'s native linear solver.
    """)
    return


@app.cell
def _(jax):
    from spd_generator import rand_spd_system
    jax.config.update("jax_enable_x64", True)
    return


if __name__ == "__main__":
    app.run()
