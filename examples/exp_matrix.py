import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    import sympy as sp

    import marimo as mo
    return jnp, mo, sp


@app.cell
def _(mo):
    mo.md(r"""# Matrix exponentials""")
    return


@app.cell
def _(jnp):
    G = jnp.array([[-2, 0], [1, -1]])
    return (G,)


@app.cell
def _(G):
    G @ G
    return


@app.cell
def _(G, jnp):
    jnp.linalg.matrix_power(G, 2)
    return


@app.cell
def _(G):
    G * G
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Double integrator dynamics

    $\dot{x} = \begin{pmatrix} 0 &  1 \\ 0 & 0 \\ \end{pmatrix}x + \begin{pmatrix} 0 \\ 1 \end{pmatrix} u$

    where

    $A = \begin{pmatrix} 0 &  1 \\ 0 & 0 \\ \end{pmatrix}$ and $B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$
    """
    )
    return


@app.cell
def _(sp):
    dh = sp.Symbol('dh', real=True)

    A = sp.Matrix([[0, 1], [0, 0]])
    B = sp.Matrix([[0], [1]]);
    return A, B, dh


@app.cell
def _(A):
    A
    return


@app.cell
def _(B):
    B
    return


@app.cell
def _(A, dh):
    Ak = (A * dh).exp()
    Ak
    return


@app.cell
def _(A, B, dh, sp):
    tau = sp.Symbol('tau', real=True)
    exp_A_tau = (A * tau).exp()
    integrand = exp_A_tau * B
    Bk = sp.integrate(integrand, (tau, 0, dh))
    Bk
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$x_{k+1} = \begin{pmatrix} 1 & \Delta h \\ 0 & 1\end{pmatrix} x_k + \begin{pmatrix} \frac{1}{2}\Delta h^2 \\ \Delta h \end{pmatrix} u_k$""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
