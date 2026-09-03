import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    import matplotlib.pyplot as plt
    import cvxpy as cp
    import marimo as mo

    key = jax.random.PRNGKey(42)
    return cp, jax, jnp, key, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Let's solve the following quadratic program with equality constraints:

    \begin{align}
        \underset{x}{\textrm{minimize}} \,\, & \frac{1}{2}x^TPx + p^Tx \nonumber \\
        \textrm{subject to} 
        \,\,& Ax=b\nonumber \\ 
    \end{align}

    where $P \in \mathbb{S}_{+}^{n_x}$,  $p \in \mathbb{R}^{n_x}$, $A \in \mathbb{R}^{n_\text{eq} \times n_x}$, and $b \in \mathbb{R}^{n_\text{eq}}$.
    """
    )
    return


@app.cell
def _(jnp):
    nx = 3
    n_eq = 2

    P = jnp.array([[6., 0, 0], [0, 5.0, 0], [0., 0., 1.]])
    p = jnp.array([14.0, -32.0, -3])

    A = jnp.array([[1., 0, 1], [0, 1, 1]])
    b = jnp.array([3., 12.0])
    return A, P, b, n_eq, nx, p


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Let's first check that the KKT matrix will be invertible

    This means we check if (1) $P \in \mathbb{S}_{+}^{n_x}$ and (2) $A$ is full rank.
    """
    )
    return


@app.cell
def _(P, jnp):
    eig_tol = 1e-3
    P_eigval, P_eigvec = jnp.linalg.eig(P)

    print(f'P is posdef!' if jnp.all(jnp.imag(P_eigval) == 0.) and jnp.all(jnp.real(P_eigval) > eig_tol) else f'P is not posdef!')
    return


@app.cell
def _(A, jnp):
    print(f'A is full rank!' if jnp.linalg.matrix_rank(A) == jnp.min(jnp.shape(A)[0]) else 'A is not full rank!')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Let's first solve this using `cvxpy` to make sure a feasible solution exists""")
    return


@app.cell
def _(A, P, b, cp, nx, p):
    x_cvx = cp.Variable(shape=(nx,))

    cost = 0.
    cost += 0.5*cp.quad_form(x_cvx, P) + p.T @ x_cvx

    constraints = []
    constraints += [A @ x_cvx == b]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve();
    return constraints, cost, prob, x_cvx


@app.cell
def _(constraints, cost, prob, x_cvx):
    if prob.status in ['infeasible', 'unbounded']:
        print(f'[WARN] Problem is infeasible!')
    else:
        print(f'Optimal QP value is {cost.value}')
        print(f'Primal solution is {x_cvx.value} and dual solution is {constraints[0].dual_value}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Let's use infeasible start Newton method to iteratively solve this QP""")
    return


@app.cell
def _(jnp):
    def compute_kkt_matrix(A: jnp.array, P: jnp.array):
        n_eq, nx = A.shape
        assert P.shape[0] == nx
        assert P.shape[1] == nx

        KKT_mat = jnp.zeros((nx+n_eq, nx+n_eq))
        KKT_mat = KKT_mat.at[0:nx, 0:nx].set(P)
        KKT_mat = KKT_mat.at[0:nx, nx:].set(A.T)
        KKT_mat = KKT_mat.at[nx:, 0:nx].set(A)

        return KKT_mat

    def compute_residual(P: jnp.array, p: jnp.array, A: jnp.array, b: jnp.array, xbar: jnp.array, ybar: jnp.array):
        return -jnp.concatenate([P @ xbar + p + A.T @ ybar, A @ xbar - b])
    return compute_kkt_matrix, compute_residual


@app.cell
def _(A, P, compute_kkt_matrix):
    KKT_mat = compute_kkt_matrix(A, P) # For a quadratic program, the KKT_matrix need only be computed once
    return (KKT_mat,)


@app.cell
def _():
    max_iter = 1000
    newton_tol = 1e-12
    return max_iter, newton_tol


@app.cell
def _(
    A,
    KKT_mat,
    P,
    b,
    compute_residual,
    jax,
    jnp,
    key,
    max_iter,
    n_eq,
    newton_tol,
    nx,
    p,
):
    xk, yk = jax.random.normal(key, shape=(nx,)), jax.random.normal(key, shape=(n_eq,))
    for ii in range(max_iter):
        residual_vector = compute_residual(P, p, A, b, xk, yk)
        descent_direction = jnp.linalg.solve(KKT_mat, residual_vector)
        xk += descent_direction[:nx]
        yk += descent_direction[nx:]

        if jnp.linalg.norm(compute_residual(P, p, A, b, xk, yk) < newton_tol):
            print(f'Terminating after {ii+1} steps!')
            break
    return xk, yk


@app.cell
def _(max_iter, xk, yk):
    print(f'After {max_iter} iterations, optimal primal is {xk} and optimal dual is {yk}')
    return


@app.cell
def _(constraints, jnp, x_cvx, xk, yk):
    print(f'"Distance" to cvxpy solution is {jnp.linalg.norm(xk-x_cvx.value)} and {jnp.linalg.norm(yk-constraints[0].dual_value)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### For a equality-constrained QP, the solution can also be found analytically in a single step""")
    return


@app.cell
def _(KKT_mat, b, jnp, nx, p):
    rhs_eq = jnp.concatenate([-p, b])
    xy_onestep = jnp.linalg.solve(KKT_mat, rhs_eq)
    x_onestep, y_onestep = xy_onestep[:nx], xy_onestep[nx:]
    return x_onestep, y_onestep


@app.cell
def _(constraints, jnp, x_cvx, x_onestep, y_onestep):
    print(f'"Distance" to cvxpy solution is {jnp.linalg.norm(x_onestep-x_cvx.value)} and {jnp.linalg.norm(y_onestep-constraints[0].dual_value)}')
    return


if __name__ == "__main__":
    app.run()
