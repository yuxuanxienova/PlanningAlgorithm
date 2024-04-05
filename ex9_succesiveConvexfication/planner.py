import sys
import numpy as np
import cvxpy as cvx
import sympy as spy
from dataclasses import dataclass, field
from time import time

from numpy.typing import NDArray
from matplotlib import pyplot as plt
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters

from pdm4ar.exercises_def.ex09.utils_params import PlanetParams, SatelliteParams
from pdm4ar.exercises.ex09.rocket import Model
from pdm4ar.exercises.ex09.discretization import FirstOrderHold
from pdm4ar.exercises.ex09.scproblem import SCProblem


# Uitils
def format_line(name, value, unit=""):
    """
    Formats a line e.g.
    {Name:}           {value}{unit}
    """
    name += ":"
    if isinstance(value, (float, np.ndarray)):
        value = f"{value:{0}.{4}}"

    return f"{name.ljust(40)}{value}{unit}"


class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    viz_traj: bool = True  # if True, the trajectory is visualized
    iterations: int = 100  # max algorithm iterations

    # Weight constants
    w_nu: float = 1e5  # virtual control
    w_sigma: float = 10  # flight time  # default 10
    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region variables
    rho_1: float = 0.25
    rho_2: float = 0.9
    alpha: float = 2.0
    beta: float = 3.2

    # Discretization constants
    K: int = 50  # number of discretization steps


class RocketPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: list[PlanetParams]
    satellites: list[SatelliteParams]
    rocket: Model
    sg: RocketGeometry
    sp: RocketParameters

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: RocketGeometry,
        sp: RocketParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        self.m = Model(self.sg, self.sp, self.planets, self.satellites, self.params.K)

        # state and input
        self.X = np.empty(shape=[self.m.n_x, self.params.K])
        self.U = np.zeros(shape=[self.m.n_u, self.params.K])

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        self.tr_radius = self.params.tr_radius

        # START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
        self.integrator = FirstOrderHold(self.m, self.params.K)
        self.problem = SCProblem(self.m, self.params.K)

        self.last_nonlinear_cost = None
        self.converged = False

    def compute_trajectory(
        self,
        init_state: RocketState,
        goal_state: DynObstacleState,
        dynamic_goal,
        time_past_val: float,
        legacy_X: NDArray = None,
        legacy_U: NDArray = None,
        legacy_sigma: float = None,
    ) -> tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """

        if dynamic_goal == None:
            self.goal_state = goal_state.as_ndarray()
        else:
            if legacy_sigma is not None:
                self.goal_state = dynamic_goal.get_target_state_at(
                    legacy_sigma
                ).as_ndarray()
            else:
                self.goal_state = dynamic_goal.get_target_state_at(
                    time_past_val + self.m.t_f
                ).as_ndarray()

        # Initialize
        self.init_state = init_state.as_ndarray()

        if legacy_X is not None:
            self.X = legacy_X
            self.X[0:6, -1] = self.goal_state
            if legacy_U is not None:
                self.U = legacy_U
            else:
                self.U = np.zeros(shape=[self.m.n_u, self.params.K])
        else:
            self.X, self.U = self.m.initialize_guess(self.init_state, self.goal_state)
        if legacy_sigma is not None:
            self.sigma = (
                legacy_sigma - time_past_val
            )  # Theorerically, this should be positive
            if self.sigma < 1e-3:
                self.sigma = self.m.t_f
                print("Warning: sigma is negtive!!!")
        else:
            self.sigma = self.m.t_f
        self.tr_radius = self.params.tr_radius

        if self.m.n_sat > 0:
            self.sats_pos = self.m.get_position_list(
                time_past_val, self.sigma, self.params.K
            )
        else:
            self.sats_pos = 1  # dummy value

        for it in range(self.params.iterations):
            t0_it = time()
            print("-" * 50)
            print("-" * 18 + f" Iteration {str(it + 1).zfill(2)} " + "-" * 18)
            print("-" * 50)

            t0_tm = time()
            (
                A_bar,
                B_bar,
                C_bar,
                S_bar,
                z_bar,
            ) = self.integrator.calculate_discretization(self.X, self.U, self.sigma)
            print(format_line("Time for transition matrices", time() - t0_tm, "s"))

            if self.m.n_sat > 0:
                self.sats_pos = self.m.get_position_list(
                    time_past_val, self.sigma, self.params.K
                )

            self.problem.set_parameters(
                A_bar=A_bar,
                B_bar=B_bar,
                C_bar=C_bar,
                S_bar=S_bar,
                z_bar=z_bar,
                X_last=self.X,
                U_last=self.U,
                time_past=time_past_val,
                sigma_last=self.sigma,
                x_init=self.init_state,
                goal_state=self.goal_state,
                sats_pos=self.sats_pos,
                weight_nu=self.params.w_nu,
                weight_sigma=self.params.w_sigma,
                tr_radius=self.tr_radius,
            )

            while True:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver,
                    solver=self.params.solver,
                    max_iters=200,
                )
                print(format_line("Solver Error", error))

                # get solution
                new_X = self.problem.get_variable("X")
                new_U = self.problem.get_variable("U")
                new_sigma = self.problem.get_variable("sigma")

                # May not have a solution. new_X is None
                try:
                    X_nl = self.integrator.integrate_nonlinear_piecewise(
                        new_X, new_U, new_sigma
                    )

                except:
                    print("No solution found.")
                    self.converged = True
                    break

                linear_cost_dynamics = np.linalg.norm(
                    self.problem.get_variable("nu"), 1
                )
                nonlinear_cost_dynamics = np.linalg.norm(new_X - X_nl, 1)

                linear_cost_constraints = self.m.get_linear_cost()
                nonlinear_cost_constraints = self.m.get_nonlinear_cost(X=new_X, U=new_U)

                linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
                nonlinear_cost = (
                    nonlinear_cost_dynamics + nonlinear_cost_constraints
                )  # L

                if self.last_nonlinear_cost is None:
                    self.last_nonlinear_cost = nonlinear_cost
                    self.X = new_X
                    self.U = new_U
                    self.sigma = new_sigma
                    break

                actual_change = self.last_nonlinear_cost - nonlinear_cost  # delta_J
                predicted_change = self.last_nonlinear_cost - linear_cost  # delta_L

                print("")
                print(format_line("Virtual Control Cost", linear_cost_dynamics))
                print(format_line("Constraint Cost", linear_cost_constraints))
                print("")
                print(format_line("Actual change", actual_change))
                print(format_line("Predicted change", predicted_change))
                print("")
                print(format_line("Trajectory time", self.sigma))
                print("")
                print(format_line("Total time", time_past_val + self.sigma))
                print("")

                if abs(predicted_change) < 1e-4:  # delta 1e-4
                    self.converged = True
                    break
                else:
                    rho = actual_change / predicted_change
                    if rho < self.params.rho_0:
                        # reject solution
                        self.tr_radius /= self.params.alpha
                        print(
                            f"Trust region too large. Solving again with radius={self.tr_radius}"
                        )
                        # Reset the trust region radius and break the loop if the radius is out of range
                        if (
                            self.tr_radius < self.params.min_tr_radius
                            or self.tr_radius > self.params.max_tr_radius
                        ):
                            print("Trust region radius out of range.")
                            self.converged = True
                            break
                    else:
                        # accept solution
                        self.X = new_X
                        self.U = new_U

                        # For dynamic goal only. If the difeference between new sigma and old sigma is too large, we need to reinitialize the guess
                        if dynamic_goal is not None:
                            if abs(new_sigma - self.sigma) > 0.01:
                                self.goal_state = dynamic_goal.get_target_state_at(
                                    time_past_val + new_sigma
                                ).as_ndarray()
                                print("New goal state", self.goal_state)
                                # Important: update the final state
                                self.X[0:6, -1] = self.goal_state
                                # Important: update the trust region radius
                                self.tr_radius = self.params.tr_radius

                        self.sigma = new_sigma
                        print("Solution accepted.")

                        if rho < self.params.rho_1:
                            print("Decreasing radius.")
                            self.tr_radius /= self.params.alpha
                        elif rho >= self.params.rho_2:
                            print("Increasing radius.")
                            self.tr_radius *= self.params.beta

                        self.last_nonlinear_cost = nonlinear_cost
                        break

                self.problem.set_parameters(tr_radius=self.tr_radius)

                print("-" * 50)

            print("")
            print(format_line("Time for iteration", time() - t0_it, "s"))
            print("")

            if self.converged:
                print(f"Converged after {it + 1} iterations.")
                self.converged = False
                break

        # if not self.converged:
        #     print("Maximum number of iterations reached without convergence.")

        if self.params.viz_traj:
            # Create a figure of the trajectory
            X = self.X
            # Define the rocket segment length and width
            seg_len = 0.2
            seg_wid = 0.05

            fig, ax = plt.subplots()

            # Loop through the rocket data and plot each segment
            for i in range(len(X[0])):
                # Calculate the segment endpoints based on the position and angle
                x1 = X[0, i] - seg_len / 2 * np.cos(X[2, i])
                y1 = X[1, i] - seg_len / 2 * np.sin(X[2, i])
                x2 = X[0, i] + seg_len / 2 * np.cos(X[2, i])
                y2 = X[1, i] + seg_len / 2 * np.sin(X[2, i])

                # Plot the segment as a line with a given width and color
                ax.plot([x1, x2], [y1, y2], linewidth=seg_wid * 100, color="red")

            plt.savefig("traj.png")

        mycmds, mystates = self._extract_seq_from_array(
            self.X, self.U, time_past_val, self.sigma, self.params.K
        )

        return mycmds, mystates, self.sats_pos, self.goal_state

    @staticmethod
    def _extract_seq_from_array(
        X, U, pre_t_f, t_f, K
    ) -> tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """

        # interpolate timsstamps with final time t_f and number of steps K
        ts = np.linspace(pre_t_f, pre_t_f + t_f, K)

        # Control inputs
        F_l = U[0, :]
        F_r = U[1, :]
        dphi = U[2, :]

        cmds_list = [RocketCommands(l, r, dp) for l, r, dp in zip(F_l, F_r, dphi)]
        mycmds = DgSampledSequence[RocketCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.transpose(X)
        states = [RocketState(*v) for v in npstates]
        mystates = DgSampledSequence[RocketState](timestamps=ts, values=states)
        return mycmds, mystates
