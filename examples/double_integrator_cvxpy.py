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
    return cp, jnp, plt


@app.cell
def _(jnp):
    nx = 4
    nu = 2
    N = 51

    dh = 0.1
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
def _(jnp, nu, nx):
    Q = 2*jnp.eye(nx)
    R = jnp.eye(nu)
    return Q, R


@app.cell
def _(jnp):
    x_min = jnp.array([-10., -10, -10, -10])
    x_max = jnp.array([10., 10, 10, 10])

    u_min = jnp.array([-10., -10])
    u_max = jnp.array([10., 10.])
    return u_max, u_min, x_max, x_min


@app.cell
def _(jnp):
    x_init = jnp.array([9.4, 7.2, -0.0, 0.0])
    x_goal = jnp.array([1, -2, 0, 0])
    return x_goal, x_init


@app.cell
def _(Ak, Bk, N, Q, R, cp, nu, nx, u_max, u_min, x_goal, x_init, x_max, x_min):
    X = cp.Variable(shape=(nx, N+1))
    U = cp.Variable(shape=(nu, N))

    cost = 0.0
    constraints = []

    for cp_idx in range(N):
        cost += cp.quad_form(X[:, cp_idx+1] - x_goal, Q) + cp.quad_form(U[:, cp_idx], R)

        constraints += [X[:, cp_idx+1] == Ak @ X[:,cp_idx] + Bk @ U[:,cp_idx]]
        constraints += [X[:, cp_idx+1] <= x_max]
        constraints += [X[:, cp_idx+1] >= x_min]

        constraints += [U[:, cp_idx] <= u_max]
        constraints += [U[:, cp_idx] >= u_min]

    constraints += [X[:,0] == x_init]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    print(prob.value)
    return (X,)


@app.cell
def _(X, plt, x_max, x_min):
    plt.xlim([x_min[0], x_max[0]])
    plt.ylim([x_min[1], x_max[1]])

    plt.quiver(X[0,:].value, X[1,:].value, X[2,:].value, X[3,:].value)
    plt.plot(X[0,:].value, X[1,:].value, color='r')
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
