import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell(disabled=True)
def _(mo):
    mo.md(r"""# Just-in-time compilation""")
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import time
    import matplotlib.pyplot as plt

    import sympy as sp
    import marimo as mo
    return jax, jnp, mo, plt, time


@app.cell
def _(jax):
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    return (key,)


@app.cell
def _(jnp):
    def matrix_math_no_jit(A, B, C):
        intermediate = A @ B + C
        intermediate = intermediate @ intermediate.T
        intermediate = jnp.linalg.inv(intermediate)
        return jnp.sin(intermediate) + jnp.exp(intermediate)
    return (matrix_math_no_jit,)


@app.cell
def _(jax, matrix_math_no_jit):
    matrix_math_with_jit = jax.jit(matrix_math_no_jit)
    return (matrix_math_with_jit,)


@app.cell
def _(jax, key):
    matrix_size = 1000
    A = jax.random.normal(key, (matrix_size, matrix_size))
    B = jax.random.normal(jax.random.split(key)[0], (matrix_size, matrix_size))
    C = jax.random.normal(jax.random.split(key)[1], (matrix_size, matrix_size))
    return A, B, C


@app.cell(disabled=True)
def _(A, B, C, matrix_math_no_jit, matrix_math_with_jit):
    matrix_math_no_jit(A, B, C)
    matrix_math_with_jit(A, B, C);
    return


@app.cell(disabled=True)
def _(A, B, C, matrix_math_no_jit, matrix_math_with_jit, time):
    # Benchmark without JIT
    print("Test vanilla implementation:")
    start_time = time.time()
    for i in range(10):
        result_no_jit = matrix_math_no_jit(A, B, C)
    no_jit_time = time.time() - start_time
    print(f"Average time per iteration: {no_jit_time/10:.6f} seconds\n")

    print("Test JIT implementation:")
    start_time = time.time()
    for i in range(10):
        result_jit = matrix_math_with_jit(A, B, C)
    jit_time = time.time() - start_time
    print(f"Average time per iteration: {jit_time/10:.6f} seconds\n")
    return


@app.cell
def _(mo):
    mo.md(r"""# Automatic differentiation""")
    return


@app.cell
def _(jax, jnp):
    @jax.jit
    def quartic_function(x: jnp.array) -> float:
        return 7*x[0]**4 - 3*x[0]**3 + 2*x[0]**2 - x[0]
    return


@app.cell
def _(jax, jnp):
    @jax.jit
    def log_function(x: jnp.array) -> float:
        return 0.5*(10*x[0]**2 + x[1]**2) + 5*jnp.log(1+ jnp.exp(-x[0]-x[1]))
    return (log_function,)


@app.cell
def _(jnp):
    xs = jnp.linspace(-10, 10, 100)
    ys = jnp.linspace(-10, 10, 100)
    return xs, ys


@app.cell
def _(jax, jnp, log_function, xs, ys):
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')  # Shape: (100, 100) each
    coords = jnp.stack([X, Y], axis=-1)  # Shape: (100, 100, 2)

    # First vmap over the second-to-last axis (y-direction)
    vmap_over_y = jax.vmap(log_function, in_axes=-2)
    # Second vmap over the last remaining spatial axis (x-direction) 
    vmap_over_xy = jax.vmap(vmap_over_y, in_axes=-2)

    results_grid = vmap_over_xy(coords)  # Shape: (100, 100)
    return (results_grid,)


@app.cell
def _(plt, results_grid, xs, ys):
    plt.figure(figsize=(8, 6))
    contours = plt.contour(xs, ys, results_grid, levels=20, colors='black', alpha=0.6)
    plt.contourf(xs, ys, results_grid, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Function value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Log Function Contours')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
