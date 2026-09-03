import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt

    return jax, jnp, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2

    In this problem, we will implement simple versions of gradient descent and Newton's method.

    Hint: you may find it helpful to scan the [`jax` docs](https://docs.jax.dev/en/latest/automatic-differentiation.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 2.1 [3 Points]

    Consider the multivariable function:

    $f(x, y) = x^2 \cdot y + 3xy + \sin(x) + e^y$

    Start by filling out `gradients.function_eval` to evaluate the above expression and then compute the derivatives of the function analytically, numerically, and using `jax`:

        - First fill out the `analytical_grad` function by computing the gradient by hand
        - Second fill out the `numerical_grad` function by using finite difference approximations of the gradient: $f'(x) = \frac{f(x+\epsilon) - f(x)}{\epsilon}$, where $\epsilon > 0$ is some small number.
        - Finally, fill out the `jax_grad` function and use `jax.gradient` to evaluate the gradient using automatic differentiation.
    """)
    return


@app.cell
def _():
    from gradients import GradientsEval

    return (GradientsEval,)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 2.2 [12 Points]

    Now, we will implement simple versions of gradient descent and Newton's method.
    """)
    return


@app.cell
def _():
    from solver import Solver, function_to_minimize

    return Solver, function_to_minimize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this problem, we will implement simple versions of gradient descent and Newton's method.

    Start by implementing the following function:
    $$f(x,y) = 0.5(10x^2 + y^2) + 5\log(1+e^{-(x+y)})$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start with a simple gradient descent solver using a line search approach and fill out the implementation for `solve_with_gradient_descent` and `compute_step_size_ls`

    Then implement Newton's method for this function and ill out the implementation for `solve_with_newton_method`

    Note: do not modify the function arguments - we expect that the function will be passed directly as an argument and the gradient computed internally (Hint: use `jax.grad`)
    """)
    return


@app.cell
def _(
    function_to_minimize,
    gradient_soln_traj,
    jax,
    jnp,
    newton_soln_traj,
    plt,
):
    xs = jnp.linspace(-10, 10, 100)
    ys = jnp.linspace(-10, 10, 100)

    X, Y = jnp.meshgrid(xs, ys, indexing='ij')  # Shape: (100, 100) each
    coords = jnp.stack([X, Y], axis=-1)  # Shape: (100, 100, 2)

    # First vmap over the second-to-last axis (y-direction)
    vmap_over_y = jax.vmap(function_to_minimize, in_axes=-2)
    # Second vmap over the last remaining spatial axis (x-direction) 
    vmap_over_xy = jax.vmap(vmap_over_y, in_axes=-2)

    results_grid = vmap_over_xy(coords)  # Shape: (100, 100)

    plt.figure(figsize=(8, 6))
    contours = plt.contour(xs, ys, results_grid, levels=20, colors='black', alpha=0.6)
    plt.contourf(xs, ys, results_grid, levels=50, cmap='viridis', alpha=0.8)

    plt.plot(gradient_soln_traj[:,0], gradient_soln_traj[:,1], color='k')
    plt.plot(newton_soln_traj[:,0], newton_soln_traj[:,1], color='r')

    plt.colorbar(label='Function value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Contours')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
