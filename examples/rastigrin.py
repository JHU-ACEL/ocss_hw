import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    import marimo as mo

    key = jax.random.PRNGKey(42)
    return LogNorm, jax, jnp, key, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rastigrin function

    If $x \in \mathbb{R}^n$,

    \begin{align}
        f(x) = An + \sum_{i=1}^n x_i^2 - A \cos(2 \pi x_i),
    \end{align}

    where $A=10$.
    """)
    return


@app.cell
def _(jax, jnp):
    @jax.jit
    def rastrigrin_function(x: jnp.array) -> float:
        A = 10.0
        return A * x.shape[0] + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))

    rastrigrin_grad = jax.jit(jax.grad(rastrigrin_function))
    rastrigrin_hessian = jax.jit(jax.hessian(rastrigrin_function))

    @jax.jit
    def one_newton_step(x):
        return x - jnp.linalg.solve(rastrigrin_hessian(x), rastrigrin_grad(x))

    return one_newton_step, rastrigrin_function, rastrigrin_grad


@app.cell
def _(LogNorm, jax, jnp, np, plt, rastrigrin_function):
    xs = jnp.linspace(-5.12, 5.12, 200)
    ys = jnp.linspace(-5.12, 5.12, 200)

    X, Y = jnp.meshgrid(xs, ys, indexing='xy')
    Z = np.asarray(jax.vmap(jax.vmap(lambda x, y: rastrigrin_function(jnp.array([x, y]))))(X, Y))

    levels = jnp.logspace(-2, jnp.log10(Z.max()), 30)

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(xs, ys, Z, levels=levels, norm=LogNorm(), cmap='viridis', extend='min')
    plt.contour(xs, ys, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.6)
    plt.colorbar(cf, label='Function value (log scale)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rastigrin Function')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Try Newton method
    """)
    return


@app.cell
def _(jnp, one_newton_step, rastrigrin_function, rastrigrin_grad):
    x0 = jnp.array([1.5, 2.5])
    g0 = rastrigrin_function(x0)

    max_iter = 30
    tol = 1e-8

    for k in range(max_iter):
        if jnp.linalg.norm(g0) < tol:
            break

        g0 = rastrigrin_grad(x0)
        x0 = one_newton_step(x0)
        x0 = jnp.clip(x0, -5.12, 5.12)

    print(f"Solution: {x0}, Cost: {rastrigrin_function(x0)}")
    return (max_iter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Now let's try cross-entropy method
    """)
    return


@app.cell
def _(jax, jnp, key, max_iter, rastrigrin_function):
    num_particles = 500
    num_elites = 100

    mu0 = jnp.array([1.5, 2.5])
    sigma0 = jnp.array([4.0, 4.0])

    for jj in range(max_iter):
        key_cem, subkey_cem = jax.random.split(key)

        samples = mu0 + sigma0 * jax.random.normal(key_cem, (num_particles, mu0.shape[0]))
        costs_cem = jax.vmap(rastrigrin_function)(samples)

        _, elite_idx = jax.lax.top_k(-costs_cem, num_elites)
        elites = samples[elite_idx]

        mu0 = elites.mean(axis=0)
        mu0 = jnp.clip(mu0, -5.12, 5.12)
        sigma0 = elites.std(axis=0)

    print(f"Solution: {mu0}, Cost: {rastrigrin_function(mu0)}")
    return mu0, num_particles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### And finally, particle swarm optimization
    """)
    return


@app.cell
def _(jax, jnp, key, max_iter, mu0, num_particles, rastrigrin_function):
    w, c1, c2 = 0.7, 1.5, 1.5
    v_init = 0.45

    key_pso, k1, k2 = jax.random.split(key, 3)

    # Initialize positions across the whole domain, velocities small
    x = jax.random.uniform(k1, (num_particles, mu0.shape[0]), minval=-5.12, maxval=5.12)
    v = v_init * jax.random.normal(k2, (num_particles, mu0.shape[0]))

    # Initialize data structures for tracking best value a particle has attained
    costs_pso = jax.vmap(rastrigrin_function)(x)
    pbest, pbest_costs = x, costs_pso
    x_best = x[jnp.argmin(costs_pso)]

    for kk in range(max_iter):
        key_pso, k1, k2 = jax.random.split(key_pso, 3)
        r1 = jax.random.uniform(k1, (num_particles, mu0.shape[0]))
        r2 = jax.random.uniform(k2, (num_particles, mu0.shape[0]))

        # Compute update step
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (x_best - x)
        x = x + v

        # Constraint enforcement
        x = jnp.clip(x, -5.12, 5.12)

        # Compute costs of updated particles
        costs_pso = jax.vmap(rastrigrin_function)(x)

        # Identify which particles have improved their cost
        improved = costs_pso < pbest_costs
        pbest = jnp.where(improved[:, None], x, pbest)
        pbest_costs = jnp.where(improved, costs_pso, pbest_costs)

        # Identify global best particle
        x_best = pbest[jnp.argmin(pbest_costs)]

    print(f"Solution: {x_best}, Cost: {rastrigrin_function(x_best)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
