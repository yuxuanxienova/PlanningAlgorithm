import numpy as np
import cvxpy as cvx
import sympy as spy
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters


class Model:
    """
    A 2D path rocket collision avoidance problem.
    """

    sg: RocketGeometry
    sp: RocketParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(
        self,
        sg: RocketGeometry,
        sp: RocketParameters,
        planets,
        satellites,
        K,
    ):
        self.sg = sg
        self.sp = sp
        self.K = K

        self.t_f = 1  # it will update in the planner # 1s works for task 1 2

        self.n_x = 8  # number of states
        self.n_u = 3  # number of inputs
        self.n_sat = len(satellites)  # number of satellites

        self.rocket_radius = (
            self.sg.l / 2 * 1.5
        )  # safe buffer is half of the rocket length

        self.planets = planets
        self.satellites = satellites

        # slack variables for artificial infeasibility
        self.s_prime = []

        # Planet [(x,y),r]
        # Satellite [planet center, tau, orbit radius, omega, satelite radius]
        self.obstacles = []
        for planet in self.planets.values():
            self.obstacles.append([planet.center, planet.radius])

        self.sats = []
        for satellite in self.satellites:
            # Use the name of the satellite to get the planet center
            planet_name = satellite.split("/")[0]  # split the name by the slash
            center = planets[
                planet_name
            ].center  # get the planet center from the dictionary
            self.sats.append(
                [
                    center,
                    self.satellites[satellite].tau,
                    self.satellites[satellite].orbit_r,
                    self.satellites[satellite].omega,
                    self.satellites[satellite].radius,
                ]
            )

        for _ in self.obstacles:
            self.s_prime.append(cvx.Variable((self.K, 1), nonneg=True))

        if len(self.sats) > 0:
            for _ in self.sats:
                self.s_prime.append(cvx.Variable((self.K, 1), nonneg=True))

    def get_position_list(self, time_past, t_f, K):
        """
        This function returns a list of satellite positions for a given satellite and final time
        Satellites is a list containing the planet center, satellite tau, orbit radius, angular velocity, and satellite radius
        t_f is the final time of the simulation
        K is the number of time steps
        """
        dt = t_f / (K - 1)  # time step
        all_pos = np.empty(
            shape=[K * 2, len(self.sats)]
        )  # empty list to store the positions
        for j, sat in enumerate(self.sats):
            center = sat[0]  # planet center
            tau = sat[1]  # satellite tau
            orbit_r = sat[2]  # orbit radius
            omega = sat[3]  # angular velocity
            for k in range(K):  # loop over each time step
                t = time_past + k * dt  # current time
                theta = tau + omega * t  # current angle
                x = center[0] + orbit_r * np.cos(theta)  # x coordinate
                y = center[1] + orbit_r * np.sin(theta)  # y coordinate
                # Fill the all_pos array
                all_pos[2 * k, j] = x
                all_pos[2 * k + 1, j] = y
        return all_pos  # return the numpy array of positions

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        Dynamics:
        0 dx/dt = vx
        1 dy/dt = vy
        2 dθ/dt = vθ
        3 dvx/dt = 1/m*(sin(phi+θ)*F_l + sin(phi-θ)*F_r)
        4 dvy/dt = 1/m*(-cos(phi_l+θ)*F_l + cos(phi-θ)*F_r)
        5 dvθ/dt = 1/I*l2*cos(phi)*(F_r-F_l)
        6 dphi/dt = vphi
        7 dm/dt = -k_l*(F_l+F_r)
        """
        f = spy.zeros(self.n_x, 1)

        x = spy.Matrix(spy.symbols("x y psi vx vy dpsi phi m", real=True))  # states
        u = spy.Matrix(spy.symbols("F_l F_r dphi", real=True))  # inputs

        f[0, 0] = x[3, 0]
        f[1, 0] = x[4, 0]
        f[2, 0] = x[5, 0]

        f[3, 0] = (
            1
            / x[7, 0]
            * (
                spy.sin(x[6, 0] + x[2, 0]) * u[0, 0]
                + spy.sin(x[6, 0] - x[2, 0]) * u[1, 0]
            )
        )
        f[4, 0] = (
            1
            / x[7, 0]
            * (
                -spy.cos(x[6, 0] + x[2, 0]) * u[0, 0]
                + spy.cos(x[6, 0] - x[2, 0]) * u[1, 0]
            )
        )
        f[5, 0] = 1 / self.sg.Iz * self.sg.l_m * spy.cos(x[6, 0]) * (u[1, 0] - u[0, 0])

        f[6, 0] = u[2, 0]
        f[7, 0] = -self.sp.C_T * (u[0, 0] + u[1, 0])

        f = spy.simplify(f)
        A = spy.simplify(f.jacobian(x))
        B = spy.simplify(f.jacobian(u))

        f_func = spy.lambdify((x, u), f, "numpy")
        A_func = spy.lambdify((x, u), A, "numpy")
        B_func = spy.lambdify((x, u), B, "numpy")

        return f_func, A_func, B_func

    def initialize_guess(self, init_state, goal_state):
        """
        Initialize the trajectory.
        """
        # state and input
        X = np.zeros(shape=[self.n_x, self.K])
        U = np.zeros(shape=[self.n_u, self.K])

        x_init = init_state
        x_final = np.zeros_like(x_init)
        x_final[0:6] = goal_state

        for k in range(self.K):
            alpha1 = (self.K - k) / self.K
            alpha2 = k / self.K

            X[:, k] = alpha1 * x_init + alpha2 * x_final
        return X, U

    def get_objective(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """

        slack = 0
        for j in range(len(self.obstacles) + len(self.sats)):
            slack += cvx.sum(self.s_prime[j])

        objective = cvx.Minimize(1e5 * slack)
        # Important to add this term to the objective to regulate the fuel consumption
        objective += cvx.Minimize(cvx.sum(cvx.square(U_v)))
        return objective

    def get_constraints(self, X_v, U_v, X_last_p, U_last_p, time_past, sigma, sats_pos):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :param sigma: cvx variable for final time
        :param time_past: cvx parameter for time past
        :return: A list of cvx constraints
        """
        constraints = []
        constraints = [
            # State constraints:
            cvx.abs(X_v[0, :]) <= 10 - self.rocket_radius,
            cvx.abs(X_v[1, :]) <= 10 - self.rocket_radius,
            X_v[6, :] >= self.sp.phi_limits[0],
            X_v[6, :] <= self.sp.phi_limits[1],
            X_v[7, :] >= self.sp.m_v,  # Fuel
            # Control constraints:
            U_v[0, :] >= self.sp.F_limits[0],
            U_v[0, :] <= self.sp.F_limits[1],
            U_v[1, :] >= self.sp.F_limits[0],
            U_v[1, :] <= self.sp.F_limits[1],
            U_v[2, :] >= self.sp.dphi_limits[0],
            U_v[2, :] <= self.sp.dphi_limits[1],
            # No zero plan time:
            sigma >= 0.1,
            # No plan longer than 40s:
            sigma <= 25,
        ]

        # linearized obstacles (for planets)
        for j, obst in enumerate(self.obstacles):
            p = obst[0]
            r = obst[1] + self.rocket_radius

            lhs = [
                (X_last_p[0:2, k] - p)
                / (cvx.norm((X_last_p[0:2, k] - p)) + 1e-6)
                @ (X_v[0:2, k] - p)
                for k in range(self.K)
            ]
            constraints += [r - cvx.vstack(lhs) <= self.s_prime[j]]

        # linearized obstacles (for satellites)
        if self.n_sat > 0:
            for j, sat in enumerate(self.sats):
                p = sats_pos[:, j]
                r = sat[4] + self.rocket_radius * 1.2  # compensate for uncertainty

                lhs = [
                    (X_last_p[0:2, k] - p[2 * k : (2 * k + 2)])
                    / (cvx.norm((X_last_p[0:2, k] - p[2 * k : (2 * k + 2)])) + 1e-6)
                    @ (X_v[0:2, k] - p[2 * k : (2 * k + 2)])
                    for k in range(self.K)
                ]
                constraints += [
                    r - cvx.vstack(lhs) <= self.s_prime[j + len(self.obstacles)]
                ]

        return constraints

    def get_linear_cost(self):
        cost = 0
        # for j in range(len(self.obstacles) + len(self.sats)):
        #     cost += np.sum(self.s_prime[j].value)
        return cost

    def get_nonlinear_cost(self, X, U=None):
        return 0

    ## Decrypted
    # def get_nonlinear_cost(self, X, U=None):
    #     cost = 0
    #     for obst in self.obstacles:
    #         vector_to_obstacle = X[0:2, :].T - obst[0]
    #         dist_to_obstacle = np.linalg.norm(vector_to_obstacle, 2, axis=1)
    #         is_violated = dist_to_obstacle < obst[1] + self.rocket_radius
    #         violation = obst[1] + self.rocket_radius - dist_to_obstacle
    #         cost += np.sum(is_violated * violation)
    #     if len(self.sats) > 0:
    #         for sat in self.sats:
    #             p = self.get_position_list(sat, self.t_f, K)
    #             vector_to_obstacle = X[0:2, :].T - p
    #             dist_to_obstacle = np.linalg.norm(vector_to_obstacle, 2, axis=1)
    #             is_violated = dist_to_obstacle < sat[4] + self.rocket_radius
    #             violation = sat[4] + self.rocket_radius - dist_to_obstacle
    #             cost += np.sum(is_violated * violation)
    #     return cost
