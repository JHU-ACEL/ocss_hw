import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp

    import matplotlib.pyplot as plt
    import marimo as mo

    jax.config.update("jax_enable_x64", True)
    return jax, jnp, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consider the optimization problem:

    \begin{align}
        \underset{x,y}{\textrm{minimize}} \,\, & f(x,y) = -\exp\left(-(xy+0.5)^2 - (y-1)^2\right) \nonumber \\
        \textrm{subject to} \,\,& h(x,y) = x - y^2 = 0 \nonumber
    \end{align}
    """)
    return


@app.cell
def _(jax, jnp):
    def f(z: jnp.array) -> float:
      x, y = z[0], z[1]
      return -jnp.exp(-(x*y + 0.5)**2 - (y - 1)**2)

    def h(z: jnp.array) -> float:
      x, y = z[0], z[1]
      return x - y**2

    grad_f = jax.jit(jax.grad(f))
    grad_h = jax.jit(jax.grad(h))
    return f, grad_f, grad_h


@app.cell
def _(f, jax, jnp):
    x_lims = (-1.5, 2)
    y_lims = (-0.5, 3.0)

    n_points = 500
    xs = jnp.linspace(x_lims[0], x_lims[1], n_points)
    ys = jnp.linspace(y_lims[0], y_lims[1], n_points)

    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    grid_coords = jnp.stack([X, Y], axis=-1)
    f_grid = jax.vmap(jax.vmap(f))(grid_coords)
    return f_grid, x_lims, xs, y_lims, ys


@app.cell
def _(jnp):
    def constraint_point(t: float) -> jnp.array:
        return jnp.array([t**2, t])

    def constraint_tangent(t: float) -> jnp.array:
        return jnp.array([2*t, 1.0])

    return constraint_point, constraint_tangent


@app.cell
def _(constraint_point, f, jax, jnp):
    t_lims = (0.4, 1.6)

    ts_curve = jnp.linspace(-1.0, 1.75, 300)
    curve_pts = jnp.stack([constraint_point(t) for t in ts_curve])

    def g(t: float) -> float:
        return f(constraint_point(t))

    dg = jax.grad(g)
    d2g = jax.grad(dg)

    t_scan = jnp.linspace(-5.0, 5.0, 401)
    t_star = t_scan[jnp.argmin(jnp.array([g(t) for t in t_scan]))]

    for _ in range(50):
        t_star = t_star - dg(t_star) / d2g(t_star)
    return curve_pts, t_lims, t_star


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot the level curves of $f(x)$ and the constraint surface $h(x)=0$
    """)
    return


@app.cell
def _(COLOR_CURVE, curve_pts, f_grid, plt, x_lims, xs, y_lims, ys):
    fig_, ax_ = plt.subplots(figsize=(7.5, 7.5))

    ax_.contourf(xs, ys, f_grid.T, levels=60, cmap="viridis", alpha=0.85)
    ax_.contour(xs, ys, f_grid.T, levels=18, colors=COLOR_CURVE, alpha=0.35, linewidths=0.6)

    # the feasible set
    ax_.plot(curve_pts[:, 0], curve_pts[:, 1], color=COLOR_CURVE, lw=2.0, label=r"$h(x,y) = x - y^2 = 0$")

    ax_.set_xlabel(r"$x$")
    ax_.set_ylabel(r"$y$")
    ax_.set_xlim(x_lims)
    ax_.set_ylim(y_lims)
    ax_.set_aspect("equal")  # equal aspect, otherwise parallel vectors do not look parallel
    ax_.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax_.spines[["top", "right"]].set_visible(False)

    fig_
    return


@app.cell
def _(constraint_point, f, grad_f, grad_h, jnp, t_star):
    z_star = constraint_point(t_star)
    lambda_star = -grad_f(z_star)[0] / grad_h(z_star)[0]

    print(f"Constrained minimizer at t* = {t_star:.6f}, (x*, y*) = ({z_star[0]:.6f}, {z_star[1]:.6f})")
    print(f"Optimal value f(x*, y*) = {f(z_star):.6f}, multiplier lambda* = {lambda_star:.6f}")
    print(f"Stationarity residual ||grad f + lambda* grad h|| = {jnp.linalg.norm(grad_f(z_star) + lambda_star*grad_h(z_star)):.3e}")
    return (z_star,)


@app.cell
def _(mo, t_lims):
    t_slider = mo.ui.slider(
        start=t_lims[0],
        stop=t_lims[1],
        step=0.002,
        value=0.8,
        label=r"$t$ along the constraint $(x,y) = (t^2, t)$",
        show_value=True,
        full_width=True,
    )
    t_slider
    return (t_slider,)


@app.cell(hide_code=True)
def _(
    constraint_point,
    constraint_tangent,
    curve_pts,
    f_grid,
    grad_f,
    grad_h,
    jnp,
    plt,
    t_slider,
    x_lims,
    xs,
    y_lims,
    ys,
    z_star,
):
    COLOR_GRAD_F = "#0072B2"  # blue
    COLOR_GRAD_H = "#D55E00"  # vermillion
    COLOR_CURVE = "#1a1a1a"
    ARROW_LEN = 0.55

    t = t_slider.value
    z = constraint_point(t)

    gf, gh = grad_f(z), grad_h(z)
    norm_gf, norm_gh = jnp.linalg.norm(gf), jnp.linalg.norm(gh)
    gf_hat, gh_hat = gf / norm_gf, gh / norm_gh

    sin_theta = gf_hat[0]*gh_hat[1] - gf_hat[1]*gh_hat[0]
    angle_deg = jnp.rad2deg(jnp.arctan2(jnp.abs(sin_theta), jnp.dot(gf_hat, gh_hat)))

    tangent_hat = constraint_tangent(t) / jnp.linalg.norm(constraint_tangent(t))
    dir_deriv = jnp.dot(gf, tangent_hat)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    ax.contourf(xs, ys, f_grid.T, levels=60, cmap="viridis", alpha=0.85)
    ax.contour(xs, ys, f_grid.T, levels=18, colors=COLOR_CURVE, alpha=0.35, linewidths=0.6)

    ax.plot(curve_pts[:, 0], curve_pts[:, 1], color=COLOR_CURVE, lw=2.0, label=r"$h(x,y) = x - y^2 = 0$")

    tangent_seg = jnp.stack([z - 0.8*tangent_hat, z + 0.8*tangent_hat])
    ax.plot(tangent_seg[:, 0], tangent_seg[:, 1], color=COLOR_CURVE, lw=1.2, ls="--", alpha=0.7)

    ax.plot(z_star[0], z_star[1], marker="*", ms=18, color="w", mec=COLOR_CURVE, mew=1.2, ls="none",
            label=r"constrained minimizer $(x^\star, y^\star)$", zorder=4)
    ax.plot(z[0], z[1], marker="o", ms=9, color="w", mec=COLOR_CURVE, mew=1.8, ls="none", zorder=5)

    for vec, color, name in ((gf_hat, COLOR_GRAD_F, r"$\nabla f$"), (gh_hat, COLOR_GRAD_H, r"$\nabla h$")):
        ax.annotate(
            "", xy=(z[0] + ARROW_LEN*vec[0], z[1] + ARROW_LEN*vec[1]), xytext=(z[0], z[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6, mutation_scale=22,
                            shrinkA=0, shrinkB=0),
            zorder=6,
        )
        # direct label, so the two arrows are never told apart by color alone
        ax.text(z[0] + 1.14*ARROW_LEN*vec[0], z[1] + 1.14*ARROW_LEN*vec[1], name, color=color,
                fontsize=15, ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="w", ec="none", alpha=0.8))

    parallel = jnp.abs(sin_theta) < 1e-3
    ax.set_title(
        f"t = {t:.3f}    "
        + r"$\angle(\nabla f, \nabla h)$ = "
        + f"{angle_deg:.1f}"
        + r"$^\circ$"
        + f"    $\\|\\nabla f\\| = {norm_gf:.3f}$,  $\\|\\nabla h\\| = {norm_gh:.3f}$\n"
        + (r"gradients are parallel: $\nabla f + \lambda \nabla h = 0$ with $\lambda$ = "
           + f"{-norm_gf/norm_gh * jnp.sign(jnp.dot(gf_hat, gh_hat)):.4f}"
           if parallel else
           r"$\nabla f$ still has a tangential component: $\nabla f^T \hat{p}$ = "
           + f"{dir_deriv:+.4f}"),
        fontsize=12,
        color=("#1a1a1a" if parallel else "#5a5a5a"),
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_aspect("equal")  # equal aspect, otherwise parallel vectors do not look parallel
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig
    return (COLOR_CURVE,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
