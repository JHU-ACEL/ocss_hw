import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax import grad

    import numpy as np
    import marimo as mo

    import matplotlib.pyplot as plt

    import cvxpy as cp

    from solvers import PDIPSolver
    from ocp import OCPSolver

    from qp_tester import PDIPTester

    return OCPSolver, PDIPSolver, PDIPTester, cp, jax, jnp, mo, np, plt


@app.cell
def _(jnp):
    nx = 4
    nu = 2
    N = 21

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


@app.cell
def _(N, jnp, nu, nx, ocp_solver):
    Q_mat = 2*jnp.eye(nx)
    R_mat = jnp.eye(nu)

    big_P = ocp_solver.compute_cost_term(Q_mat, R_mat, N)
    return Q_mat, R_mat, big_P


@app.cell
def _(jnp):
    x_min = jnp.array([-10, -10, -5, -5])
    x_max = jnp.array([10, 10, 7, 7])

    u_min = jnp.array([-15, -10])
    u_max = jnp.array([6, 6])
    return u_max, u_min, x_max, x_min


@app.cell
def _(jnp):
    x_init = jnp.array([9.4, 7.2, -0.5, 1.0])
    # x_goal = jnp.array([1, -2, 0, 0])
    return (x_init,)


@app.cell
def _(Ak, Bk, jax, jnp, x_init):
    us = jnp.array([[1.14, 3.23], \
               [0.12, 0.23], \
               [0.14, -0.23], \
               [1.09, 0.0], \
               [-8.14, 0.32], \
               [-0.23, 1.14], \
               [3.18, 2.22], \
               [1.09, 3.84], \
               [-6.15, -0.23], \
               [2.02, 0.0], \
               [0.0, -0.32], \
               [-1.01, 1.01], \
              ])

    def double_integrator(x0, u0):
        new_state = Ak @ x0 + Bk @ u0
        return new_state, new_state

    xs = jax.lax.scan(double_integrator, x_init, us)[1]
    return us, xs


@app.cell
def _(Ak, N, ocp_solver, x_init):
    rhs_eq = ocp_solver.compute_equality_bc(x_init, Ak, N)
    return (rhs_eq,)


@app.cell
def _(N, jnp, nu, nx, us, xs):
    z0_hat = jnp.zeros(N*(nx+nu))

    for ii in range(N):
        start_idx, stop_idx = (nx+nu)*ii, (nx+nu)*ii+nu
        z0_hat = z0_hat.at[start_idx:stop_idx].set(us[ii])

        start_idx, stop_idx = (nx+nu)*ii+nu, (nx+nu)*ii+nx+nu
        z0_hat = z0_hat.at[start_idx:stop_idx].set(xs[ii])
    return (z0_hat,)


@app.cell
def _(Ak, Bk, N, nx, ocp_solver):
    n_eq = N*nx
    A_eq_con = ocp_solver.compute_equality_matrix(Ak, Bk, N)
    assert A_eq_con.shape[0] == n_eq
    return (A_eq_con,)


@app.cell
def _(A_eq_con, jnp, rhs_eq, z0_hat):
    jnp.linalg.norm(A_eq_con @ z0_hat - rhs_eq)
    return


@app.cell
def _(N, ocp_solver, u_max, u_min, x_max, x_min):
    G_ineq, h_ineq = ocp_solver.compute_ineq_con(x_min, x_max, u_min, u_max, N)
    return G_ineq, h_ineq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create cvxpy version
    """)
    return


@app.cell
def _(Ak, Bk, N, Q_mat, R_mat, cp, nu, nx, u_max, u_min, x_init, x_max, x_min):
    X_cvx = cp.Variable(shape=(nx, N+1))
    U_cvx = cp.Variable(shape=(nu, N))

    cost = 0.0
    constraints = []

    for cp_idx in range(N):
        cost += cp.quad_form(X_cvx[:, cp_idx+1], Q_mat) + cp.quad_form(U_cvx[:, cp_idx], R_mat)

        constraints += [X_cvx[:, cp_idx+1] == Ak @ X_cvx[:,cp_idx] + Bk @ U_cvx[:,cp_idx]]
        constraints += [X_cvx[:, cp_idx+1] <= x_max]
        constraints += [X_cvx[:, cp_idx+1] >= x_min]

        constraints += [U_cvx[:, cp_idx] <= u_max]
        constraints += [U_cvx[:, cp_idx] >= u_min]

    constraints += [X_cvx[:,0] == x_init]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    print(prob.value)
    return U_cvx, X_cvx, prob


@app.cell
def _(X_cvx, plt, x_init, x_max, x_min):
    plt.xlim([x_min[0], x_max[0]])
    plt.ylim([x_min[1], x_max[1]])

    plt.quiver(X_cvx[0,:].value, X_cvx[1,:].value, X_cvx[2,:].value, X_cvx[3,:].value)
    plt.scatter(x_init[0], x_init[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Now run cvxpy with our matrices
    """)
    return


@app.cell
def _(A_eq_con, G_ineq, N, big_P, cp, h_ineq, nu, nx, rhs_eq):
    Z_big = cp.Variable(shape=(N*(nx+nu)))

    cost_big = cp.quad_form(Z_big[:], big_P)

    constraints_big = []
    constraints_big += [G_ineq @ Z_big <= h_ineq]
    constraints_big += [A_eq_con @ Z_big == rhs_eq]

    prob_big = cp.Problem(cp.Minimize(cost_big), constraints_big)
    prob_big.solve()
    print(prob_big.value)
    return Z_big, prob_big


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convert the original cvxpy solution into $(u_0, x_0, \ldots, u_N, x_N$) form for comparison

    This check is to make sure that using the CVXpy interface to write out the dynamics/inequality constraints yields the same solution as the one we use to construct our $A$/$b$ and $G$/$h$ matrices.
    """)
    return


@app.cell
def _(N, U_cvx, X_cvx, Z_big, jnp, np, nu, nx, prob, prob_big, x_init):
    X_big = np.zeros((nx, N+1))
    U_big = np.zeros((nu, N))

    X_big[:, 0] = x_init
    for jj in range(N):
        U_big[:,jj] = Z_big[jj*(nx+nu):jj*(nx+nu)+nu].value 
        X_big[:,jj+1] = Z_big[jj*(nx+nu) + nu:jj*(nx+nu) + nu+nx].value

    print(f"Two different versions of CVXpy are {jnp.linalg.norm(X_big - X_cvx.value) + jnp.linalg.norm(U_big - U_cvx.value)} in distance")
    print(f"Two different versions of are {abs(prob_big.value - prob.value)}")
    return U_big, X_big


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # JAX Implementation
    """)
    return


@app.cell
def _(N, jnp, nu, nx):
    big_p = jnp.zeros(N*(nx+nu))
    return (big_p,)


@app.cell
def _(A_eq_con, G_ineq, PDIPSolver, big_P, big_p, h_ineq, rhs_eq):
    solver = PDIPSolver()
    solver.init_problem(2*big_P, big_p, A_eq_con, rhs_eq, G_ineq, h_ineq)
    costs = solver.solve_qp(verbose=False)
    return costs, solver


@app.cell
def _(costs, plt):
    plt.plot(costs)
    return


@app.cell
def _(Z_big, costs, jnp, prob, prob_big, solver):
    print(f'CVXpy and JAX solutions are {jnp.linalg.norm(Z_big.value - solver.x0)} in distance')
    print(f'CVXpy and JAX costs are {abs(prob.value-costs[-1])} and {abs(prob_big.value-costs[-1])} apart')
    return


@app.cell
def _(N, np, nu, nx, solver, x_init):
    X_jax = np.zeros((nx, N+1))
    U_jax = np.zeros((nu, N))

    X_jax[:, 0] = x_init
    for kk in range(N):
        U_jax[:,kk] = solver.x0[kk*(nx+nu):kk*(nx+nu)+nu]
        X_jax[:,kk+1] = solver.x0[kk*(nx+nu) + nu:kk*(nx+nu) + nu+nx]
    return U_jax, X_jax


@app.cell
def _(X_big, X_cvx, X_jax, plt, x_init, x_max, x_min):
    plt.xlim([x_min[0], x_max[0]])
    plt.ylim([x_min[1], x_max[1]])

    plt.plot(X_jax[0,:], X_jax[1,:], 'k')
    plt.plot(X_big[0,:], X_big[1,:], 'r.')
    plt.plot(X_cvx[0,:].value, X_cvx[1,:].value, 'g.-')
    plt.scatter(x_init[0], x_init[1])
    return


@app.cell
def _(U_big, U_cvx, U_jax, plt, u_max, u_min):
    u_idx = 0

    plt.ylim(u_min[u_idx], u_max[u_idx])

    plt.plot(U_jax[u_idx,:], 'rx')
    plt.plot(U_big[u_idx,:], 'k.')
    plt.plot(U_cvx[u_idx,:].value, 'g.-')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test PDIPTester class
    """)
    return


@app.cell
def _(A_eq_con, G_ineq, PDIPTester, big_P, big_p, h_ineq, rhs_eq):
    tester = PDIPTester(2*big_P, big_p, A_eq_con, rhs_eq, G_ineq, h_ineq)
    return (tester,)


@app.cell
def _(tester):
    qp_solved = tester.compare_solutions()
    print(f'QP has been solved!' if qp_solved else f'QP has not been solved!')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
