from dataclasses import dataclass
from typing import Any, Optional, Sequence, Sequence, Tuple, Union, List, Mapping
import numpy as np
import scipy
from cvxpy import (
    Variable,
    Minimize,
    quad_form,
    sum_squares,
    Problem,
    hstack,
    vstack,
    Parameter,
)

from dg_commons import DgSampledSequence, PlayerName, Color
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters

from pdm4ar.exercises.ex09.planner import RocketPlanner
from pdm4ar.exercises_def.ex09.goal import RocketTarget, SatelliteTarget
from pdm4ar.exercises_def.ex09.utils_params import PlanetParams, SatelliteParams

from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.models.vehicle import VehicleState


@dataclass(frozen=True)
class Pdm4arAgentParams:
    """
    Definition space for additional agent parameters.
    """

    param1: float = 0.2


class RocketAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: RocketState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[RocketCommands]
    state_traj: DgSampledSequence[RocketState]
    myname: PlayerName
    planner: RocketPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: RocketGeometry
    sp: RocketParameters

    def __init__(
        self,
        init_state: RocketState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only at the beginning of each simulation.
        Provides the RocketAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets
        self.replan_counter = 0
        self.cached_expected_state = None

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        Feel free to add additional methods, objects and functions that help you to solve the task
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.replan_dist = 0.5  # self.sg.l / 3
        self.planner = RocketPlanner(
            planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp
        )
        _, self.A, self.B = self.planner.m.get_equations()

        # Get goal from Targets (either moving (SatelliteTarget) or static (RocketTarget))
        if isinstance(init_sim_obs.goal, SatelliteTarget):
            self.goal_state = init_sim_obs.goal.get_target_state_at(self.planner.m.t_f)
            self.dynamic_goal = init_sim_obs.goal
            print("offset_r", init_sim_obs.goal.offset_r)
        elif isinstance(init_sim_obs.goal, RocketTarget):
            self.goal_state = init_sim_obs.goal.target
            self.dynamic_goal = None

        # Implement Compute Trajectory

        (
            self.cmds_plan,
            self.state_traj,
            self.sat_pos,
            self.goal_state_out,
        ) = self.planner.compute_trajectory(
            self.init_state, self.goal_state, self.dynamic_goal, time_past_val=0
        )

    # Liearize the dynamics around the expected state and control input
    def linearize_dynamics(self, expected_state, control_input):
        A = self.A(expected_state, control_input)
        B = self.B(expected_state, control_input)
        return A, B

    # MPC controller
    def get_commands(self, sim_obs: SimObservations) -> RocketCommands:
        """
        This is called by the simulator at every time step. (0.1 sec)
        Do not modify the signature of this method.
        """

        dt = 0.1  # (0.1 sec)
        current_state = sim_obs.players[self.myname].state.as_ndarray()
        expected_state = self.state_traj.at_interp(sim_obs.time).as_ndarray()
        self.cached_expected_state = self.state_traj.at_interp(float(sim_obs.time))

        # Replan if current state is far from expected state: maximum_replan_num=3
        print("Difference", np.linalg.norm(current_state[0:3] - expected_state[0:3]))
        if np.linalg.norm(current_state[0:3] - expected_state[0:3]) > self.replan_dist:
            print("Off track. Replaning...", self.replan_counter)
            self.replan_counter += 1
            self.init_state = sim_obs.players[self.myname].state
            print("init.state", self.init_state)

            # Warm start
            # state and input and TIME !!!
            legacy_X = np.zeros(shape=[self.planner.m.n_x, self.planner.params.K])
            legacy_U = np.zeros(shape=[self.planner.m.n_u, self.planner.params.K])

            legacy_X[:, 0] = self.init_state.as_ndarray()
            legacy_X[0:6:, -1] = self.goal_state.as_ndarray()

            dt = (self.state_traj.get_end() - float(sim_obs.time)) / (
                self.planner.params.K - 1
            )

            for i in range(self.planner.params.K - 2):
                legacy_X[:, i + 1] = self.state_traj.at_interp(
                    float(sim_obs.time) + (i + 1) * dt
                ).as_ndarray()

            for i in range(self.planner.params.K):
                legacy_U[:, i] = self.cmds_plan.at_interp(
                    float(sim_obs.time) + (i) * dt
                ).as_ndarray()

            legacy_sigma = self.state_traj.get_end()

            (
                self.cmds_plan,
                self.state_traj,
                self.sat_pos,
                self.goal_state_out,
            ) = self.planner.compute_trajectory(
                self.init_state,
                self.goal_state,
                self.dynamic_goal,
                time_past_val=float(sim_obs.time),
                legacy_X=legacy_X,
                legacy_U=legacy_U,
                legacy_sigma=legacy_sigma,  # for moving target
            )

        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)
        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)

        control_input = cmds.as_ndarray()

        # Linearize the dynamics around the expected state and control input
        A, B = self.linearize_dynamics(expected_state, control_input)

        # Convert the continuous-time system matrices to discrete-time
        A_d = scipy.linalg.expm(A * dt)  # matrix exponential
        B_d, err = scipy.integrate.quad_vec(
            lambda tau: scipy.linalg.expm(A * tau) @ B, 0, dt
        )  # matrix integral

        Q = np.diag([5, 5, 5, 1, 1, 1, 0.001, 0.001])  # state error weights
        R = np.diag([1, 1, 1])  # control input error weights

        # Define the prediction horizon P and the control horizon M
        P = 1  # number of steps to predict ahead

        # Define the control constraints
        u_min = np.array(
            [self.sp.F_limits[0], self.sp.F_limits[0], self.sp.dphi_limits[0]]
        )  # minimum input values
        u_max = np.array(
            [self.sp.F_limits[1], self.sp.F_limits[1], self.sp.dphi_limits[1]]
        )  # maximum input values

        # Define the decision variable
        U = Variable((3, P))  # matrix of control inputs over the control horizon

        # Define the objective function
        obj = 0  # initialize the objective function
        x = current_state  # initialize the state vector
        ref_state = self.state_traj.at_interp(
            float(sim_obs.time)
            - dt  # important, the roket is always ahead of the expected state (slow it down)
        ).as_ndarray()  # initialize the reference state vector
        for i in range(P):
            # Predict the next state using the discrete-time model
            x = A_d @ x + B_d @ U[:, i]
            # Calculate the state and input errors
            e_x = x - ref_state
            e_u = U[:, i] - control_input
            # Add the weighted errors to the objective function
            obj += quad_form(e_x, Q) + quad_form(e_u, R)

        # Define the constraints
        constr = []  # initialize the constraint list
        for i in range(P):
            # Add the input constraints
            constr += [u_min <= U[:, i], U[:, i] <= u_max]
        x = current_state  # initialize the state vector

        # Solve the optimization problem
        prob = Problem(Minimize(obj), constr)  # define the problem
        prob.solve()  # solve the problem

        # Check the status of the solution
        if prob.status == "optimal":
            # Use the first optimal control input
            u = U[:, 0].value
            # print("change:", sum(abs(u - control_input)))
        else:
            # Use the nominal control input
            print("MPC not solvable")
            u = control_input

        # Return the control input
        cmds.F_left = u[0]
        cmds.F_right = u[1]
        cmds.dphi = u[2]
        return cmds

    # # Discrete-time LQR controller
    # def get_commands(self, sim_obs: SimObservations) -> RocketCommands:
    #     """
    #     This is called by the simulator at every time step. (0.1 sec)
    #     Do not modify the signature of this method.
    #     """

    #     dt = 0.1
    #     current_state = sim_obs.players[self.myname].state.as_ndarray()
    #     expected_state = self.state_traj.at_interp(sim_obs.time).as_ndarray()

    #     # # Replan if current state is far from expected state: maximum_replan_num=3
    #     print("Difference", np.linalg.norm(current_state[0:3] - expected_state[0:3]))
    #     # if (
    #     #     self.replan_dist <= 0.6
    #     #     and np.linalg.norm(current_state[0:3] - expected_state[0:3])
    #     #     > self.replan_dist
    #     # ):
    #     #     print("Off track. Replan number", self.replan_counter)
    #     #     print("Replan distance", self.replan_dist)
    #     #     self.replan_counter += 1
    #     #     self.replan_dist += 0.1
    #     #     self.init_state = sim_obs.players[self.myname].state
    #     #     print("init.state", self.init_state)
    #     #     self.cmds_plan, self.state_traj = self.planner.compute_trajectory(
    #     #         self.init_state,
    #     #         self.goal_state,
    #     #         self.dynamic_goal,
    #     #         time_past_val=float(sim_obs.time),
    #     #     )

    #     # FirstOrderHold
    #     cmds = self.cmds_plan.at_interp(sim_obs.time)
    #     # ZeroOrderHold
    #     # cmds = self.cmds_plan.at_or_previous(sim_obs.time)

    #     control_input = cmds.as_ndarray()

    #     # Linearize the dynamics around the expected state and control input
    #     A, B = self.linearize_dynamics(expected_state, control_input)

    #     # Convert the continuous-time system matrices to discrete-time
    #     A_d = scipy.linalg.expm(A * dt)  # matrix exponential
    #     B_d, err = scipy.integrate.quad_vec(
    #         lambda tau: scipy.linalg.expm(A * tau) @ B, 0, dt
    #     )  # matrix integral

    #     # Define the weighting matrices Q and R
    #     Q = np.diag(
    #         [0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    #     )  # state error weights
    #     R = np.diag([1, 1, 1])  # control input error weights

    #     try:
    #         # Solve the Riccati equation for P
    #         P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)

    #         # Calculate the feedback gain matrix K
    #         K = scipy.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d

    #         # Calculate the control input
    #         u = control_input - K @ (current_state - expected_state)

    #         # # if u out of bounds, use control input
    #         # if (
    #         #     u[0] < self.sp.F_limits[0]
    #         #     or u[0] > self.sp.F_limits[1]
    #         #     or u[1] < self.sp.F_limits[0]
    #         #     or u[1] > self.sp.F_limits[1]
    #         #     or u[2] < self.sp.dphi_limits[0]
    #         #     or u[2] > self.sp.dphi_limits[1]
    #         # ):
    #         #     print("LQR out of bounds")
    #         #     u = control_input

    #     except:
    #         print("LQR not solvable")
    #         u = control_input

    #     # Return the control input
    #     cmds.F_left = u[0]
    #     cmds.F_right = u[1]
    #     cmds.dphi = u[2]
    #     return cmds
    def on_get_extra(self) -> Optional[Sequence[Tuple[Trajectory, Color]]]:
        plottable: List[Tuple[Trajectory, Color]] = []
        # Define the rocket segment length and width
        seg_len = 0.2
        seg_wid = 0.05

        def state_to_plot(state: RocketState):
            x1 = state.x - seg_len / 2 * np.cos(state.psi)
            y1 = state.y - seg_len / 2 * np.sin(state.psi)
            x2 = state.x + seg_len / 2 * np.cos(state.psi)
            y2 = state.y + seg_len / 2 * np.sin(state.psi)
            values = [
                VehicleState(x=x1, y=y1, psi=0, vx=0, delta=0),
                VehicleState(x=x2, y=y2, psi=0, vx=0, delta=0),
            ]
            traj = Trajectory(timestamps=np.arange(0, len(values)) * 0.1, values=values)
            return traj

        def goalStateToPlot(goal: np.array):
            x1 = goal[0] - seg_len / 2 * np.cos(goal[2])
            y1 = goal[1] - seg_len / 2 * np.sin(goal[2])
            x2 = goal[0] + seg_len / 2 * np.cos(goal[2])
            y2 = goal[1] + seg_len / 2 * np.sin(goal[2])
            values = [
                VehicleState(x=x1, y=y1, psi=0, vx=0, delta=0),
                VehicleState(x=x2, y=y2, psi=0, vx=0, delta=0),
            ]
            traj = Trajectory(timestamps=np.arange(0, len(values)) * 0.1, values=values)
            return traj

        def satPosToPlot(sat_pos: List[float]) -> List[Trajectory]:
            x_coords = []
            y_coords = []
            for i in range(len(sat_pos)):
                if i % 2 == 0:
                    x_coords.append(sat_pos[i])
                if i % 2 == 1:
                    y_coords.append(sat_pos[i])
            coordinates = list(zip(x_coords, y_coords))
            traj_list = []
            for pos in coordinates:
                values = [
                    VehicleState(x=pos[0], y=pos[1], psi=0, vx=0, delta=0),
                    VehicleState(x=pos[0] + 0.2, y=pos[1], psi=0, vx=0, delta=0),
                ]
                traj = Trajectory(
                    timestamps=np.arange(0, len(values)) * 0.1, values=values
                )
                traj_list.append(traj)
            return traj_list

        for state in self.state_traj.values:
            contour_plot = state_to_plot(state)
            plottable.append((contour_plot, "black"))
        if self.cached_expected_state:
            contour_plot = state_to_plot(self.cached_expected_state)
            plottable.append((contour_plot, "blue"))

        if isinstance(self.sat_pos, np.ndarray):
            num = self.sat_pos.shape[1]
            for i in range(num):
                traj_list = satPosToPlot(self.sat_pos[:, i])
                for traj in traj_list:
                    plottable.append((traj, "green"))

        contour_plot_goal = goalStateToPlot(self.goal_state_out)
        plottable.append((contour_plot_goal, "red"))
        return plottable
