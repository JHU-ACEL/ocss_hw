import functools
import warnings
import jax
import jax.numpy as jnp

def rand_spd(nx, key, cond=1e4):
  """
  SPD (or PSD if rank < nx) matrix with the requested condition number.

  Inputs:
    nx (int): dimension of the matrix
    key: jax.random key.
    cond (float): desired condition number.

  Outputs:
    A (jax.Array): nx x nx PSD matrix with condition number `cond`
  """
  assert jax.config.jax_enable_x64

  # Generate list of eigenvalues up to desired cond number
  lam = jnp.logspace(0.0, jnp.log10(cond), nx)
  lam = jnp.sort(lam)

  # Generate random orthogonal matrix
  initializer = jax.nn.initializers.orthogonal()
  Q = initializer(key, (nx, nx), jnp.float64)

  # Evaluate eigendecomposition
  A = (Q * lam) @ Q.T
  A = 0.5 * (A + A.T)
  return A

def rand_spd_system(nx, key, cond):
  """
  Generates ``random'' linear system Ax=b with A
    having desired condition number cond.
  
  Inputs:
    nx (int): dimension of the matrix
    key: jax.random key.
    cond (float): desired condition number.

  Outputs:
    A (jax.Array): nx x nx SPD matrix with condition number `cond`
    b (jax.Array): right-hand side, equal to A @ x_star
    x_star (jax.Array): the known exact solution used to construct b
  """
  k_mat, k_sol = jax.random.split(key)
  A = rand_spd(nx, k_mat, cond)
  x_star = jax.random.normal(k_sol, (nx,))
  return A, A @ x_star, x_star
