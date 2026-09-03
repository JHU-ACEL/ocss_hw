import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt

    from solver import Solver

    return jax, jnp, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3

    In this problem, we will implement linear and non-linear least squares to predict the planetary motion of Mars as a function of time.

    To start, we will use a toolkit provided by NASA Jet Propulsion Lab to query the ephemeris data for Mars.
    """)
    return


@app.cell
def _(jnp):
    from astroquery.jplhorizons import Horizons

    mars_period_days = 686.98  # Mars sidereal orbital period, in days

    omega = 2 * jnp.pi / mars_period_days  # angular frequency, rad/day

    start_date = '2010-01-01'
    stop_date = '2024-01-01'
    step_size = '5d'
    mars_id = '499'
    mars = Horizons(id=mars_id, location='@sun',
                    epochs={'start':start_date,'stop':stop_date,'step':step_size})
    mars_data = mars.vectors().to_pandas()   # x, y, z in AU; vx, vy, vz in AU/day
    return mars_data, omega


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's start with linear least squares and fit a linear model to map the time $t$ to the $x$ coordinate of the planet:

    $x(t) = A\sin(\omega t) + Bcos(\omega t) + C$,
    where $\omega = 2\pi / T$ and $T \approx 687$ days is Mars's sidereal
    orbital period.

    Here, the vector $f(t) = (\sin(\omega t), \cos(\omega t), 1)$ is known as a "feature vector" -- that is, our hypothesis is that we can find a linear model in the lifted feature space.

    Set up the linear least squares problem necessary to recover $(A, B, C)$.
    """)
    return


@app.cell
def _():
    from nn_model import construct_feature_matrix

    return (construct_feature_matrix,)


@app.cell
def _(construct_feature_matrix, jnp, mars_data, omega):
    t_mars = jnp.array((mars_data['datetime_jd'] - mars_data['datetime_jd'][0]).to_numpy())
    x_mars = jnp.array(mars_data['x'].to_numpy())

    F_mars = construct_feature_matrix(t_mars, omega)
    return F_mars, t_mars, x_mars


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 3.1 [5 Points]

    Use the analytical expression for linear least squares to find the best fit for $x = A\sin(\omega t) + B\cos(\omega t) + C$.
    """)
    return


@app.cell
def _(F_mars, plt, t_mars, theta_mars_analytical, x_mars):
    plt.figure(figsize=(9, 5))
    plt.scatter(t_mars, x_mars, s=8, alpha=0.5, label="Mars x (data)")
    plt.plot(t_mars, F_mars @ theta_mars_analytical, "k-", label="Analytical fit")
    plt.xlabel("t (days since start)")
    plt.ylabel("x (AU)")
    plt.title("Periodic Least Squares Fit to Mars x(t)")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 3.3 [5 Points]

    Once you have completed the write-up in Problem 3b, finish implementing the `solve_with_gauss_newton` method in `hw1/gauss_newton.py`.

    $Hint$: you may find that you will have to add a damping factor $\lambda I$, for some small number $\lambda > 0$, inside the matrix inverse term you derived for Gauss-Newton to improve numerical conditioning.
    """)
    return


@app.cell
def _():
    from gauss_newton import GaussNewtonSolver

    return (GaussNewtonSolver,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we will provide a simple implementation of a feedforward neural network and use Gauss-Newton for training the network.

    The neural network $h_\phi$ maps our constructed feature vector $f(t) = (\sin(\omega t), \cos(\omega t), 1)$ and learns a nonlinear transformation $x(t) = h_\phi (f(t))$. We will use Gauss-Newton to find the optimal parameters $\phi^*$ for this problem.
    """)
    return


@app.cell
def _():
    from nn_model import NNModel

    return (NNModel,)


@app.cell
def _(GaussNewtonSolver, nn, theta0_nn, x_mars):
    nn_gn_solver = GaussNewtonSolver()
    theta_nn_gn = nn_gn_solver.solve_with_gauss_newton_method(theta0_nn, x_mars, nn.forward)
    return (theta_nn_gn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KEEP PLOT BELOW FOR HW RELEASE!
    """)
    return


@app.cell
def _(nn, plt, t_mars, theta_nn_gn, x_mars):
    plt.figure(figsize=(9, 5))
    plt.scatter(t_mars, x_mars, s=8, alpha=0.5, label="Mars x (data)")
    plt.plot(t_mars, nn.forward(theta_nn_gn), "g-", label="NN (Gauss-Newton) fit")
    plt.xlabel("t (days since start)")
    plt.ylabel("x (AU)")
    plt.title("Two-layer NN fit via Gauss-Newton")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
