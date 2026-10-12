import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class PDGSolver():
  """
  Follows the implementation of the powered descent guidance algorithm presented in 
      "Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem",
      B. Acikmese, J. M. Carson III, and L. Blackmore, IEEE Transactions on Control Systems Technology,
      vol. 21, no. 6, pp. 2104 -- 2113, 2013.

  Attributes:
    m0 (float): initial mass [kg]
    mf (float): final mass [kg]
    Tmax (float): max thrust [N]
    rho1 (float): minimum throttle [N]
    rho2 (float): maximum throttle [N]
    alpha (float): fuel consumption rate [s/m]
    x0 (np.ndarray): initial state
    q (np.ndarray):  target at zero altitude
    e1 (np.ndarray): basis vector along x
    e2 (np.ndarray): basis vector along y
    e3 (np.ndarray): basis vector along z
    E (np.ndarray)
    vmax (float): maximum velocity [m/s]
    gamma_gs (float): glideslope angle [rad]
    theta (float)
    omega (np.ndarray): rotation angular velocity [rad/s]
    g (np.ndarray): gravity [m/s^2]
    zi (float): initial mass in log-space
    zf (float): difference in mass in log-space
    A (np.ndarray): dynamics matrix A from Eq. 2
    B (np.ndarray): dynamics matrix B from Eq. 2
  """
  def __init__(self): 
    """
      All values here are from the parameters listed in Section IV of the paper.
    """
    self.m0 = 2000  # initial mass [kg]
    self.mf = 300   # final mass [kg]

    self.Tmax = 24e3        # max thrust [N]
    self.rho1 = 0.2 * self.Tmax  # minimum throttle [N]
    self.rho2 = 0.8 * self.Tmax  # maximum throttle [N]

    self.alpha = 5e-4           # fuel consumption rate [s/m]

    self.x0 = np.array([2400, 450, -330, -10, -40, 10]) # initial state
    self.q  = np.array([0, 0])

    self.e1 = np.array([1,0,0]).T
    self.e2 = np.array([0,1,0]).T
    self.e3 = np.array([0,0,1]).T
    self.E = np.array([self.e2.T, self.e3.T])

    self.vmax = 90              # maximum velocity [m/s]

    self.gamma_gs = np.deg2rad(30)  # glideslope angle [rad]
    self.theta = np.deg2rad(120)

    self.omega = np.array([2.53e-5, 0, 6.62e-5])    # rotation angular velocity [rad/s]
    self.g = np.array([-3.71, 0, 0])                # gravity [m/s^2]

    self.zi = np.log(self.m0)
    self.zf = np.log(self.m0 - self.mf)

    self.construct_dynamics_matrices()

  def construct_dynamics_matrices(self) -> None:
    """
      Define self.A and self.B dynamics matrices using Equation 2 from the paper.
    """
    raise NotImplementedError("PDGSolver.construct_dynamics_matrices has yet to be implemented")

  def construct_constraints(self, dt: float, N:int, x:cp.Variable, z:cp.Variable, u:cp.Variable, sigma:cp.Variable) -> list[cp.Expression]:
    """
      Constructs a list of the constraints that are shared between Problems 3 and 4.

      Parameters:
        dt (float): time duration for integration.
        N (int): number of discretization points.
        x (cp.Variable): cvxpy array of size 6xN for state trajectory.
        u (cp.Variable): cvxpy array of size 3xN for control trajectory.
        z (cp.Variable): cvxpy array of size 1xN for log-mass trajectory.
        sigma (cp.Variable): cvxpy array of size 1xN for sigma = Gamma / mass term.

      Returns:
        constraints (list): list of cvxpy constraints for Problems 3 and 4.
    """
    raise NotImplementedError("PDGSolver.construct_constraints has yet to be implemented")

  def solve_minimum_landing_error(self, dt: float, N: int) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
      Solves Problem 3: Convex Relaxed Minimum Landing Error Problem

      Parameters:
        dt (float): time duration for integration.
        N (int): number of discretization points.

      Returns:
        x (np.array): optimal state trajectory from cvxpy
        u (np.array): optimal control trajectory from cvxpy
        status: cvxpy problem status
        cost (float): optimal value of cvxpy problem
    """
    raise NotImplementedError("PDGSolver.solve_minimum_landing_error has yet to be implemented")

  def solve_minimum_fuel_problem(self, dt:float, N:int, dp3) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
      Solves Problem 4: Convex Relaxed Minimum Fuel Problem
        using the solution of Problem 3

      Parameters:
        dt (float): time duration for integration.
        N (int): number of discretization points.
        dp3 (np.array): final optimal position coming from Problem 3 (soln of self.solve_minimum_landing_error)

      Returns:
        x (np.array): optimal state trajectory from cvxpy
        u (np.array): optimal control trajectory from cvxpy
        status: cvxpy problem status
        cost (float): optimal value of cvxpy problem
    """
    raise NotImplementedError("PDGSolver.solve_minimum_fuel_problem has yet to be implemented")
