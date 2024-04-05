from dataclasses import dataclass
import random
from typing import Sequence

from dg_commons import PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import (
    DiffDriveGeometry,
    DiffDriveParameters,
)
from dg_commons.sim.models.obstacles import StaticObstacle


import numpy as np
import matplotlib.pyplot as plt
from shapely import STRtree
from shapely import box
from shapely.geometry.base import BaseGeometry
from shapely import LineString
from shapely import LinearRing
from shapely import Point
from shapely import Polygon
from typing import Optional, TypeVar, Set, Mapping, Tuple, List
import queue
from pqdict import pqdict
import numpy as np
import matplotlib.pyplot as plt
import random
import queue
from shapely import geometry as geo
from shapely import wkt
from shapely import ops
import networkx as nx

# -------------------My function----------------------------


# -----------------1. utilities-----------------
def rotateZ(vec: np.array, theta: float):
    # theta in radius
    # vector dim = (2,1)
    vec = vec.reshape((2, 1))
    RoatMat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    RoatMat = RoatMat.reshape((2, 2))
    return RoatMat @ vec


def vecToAngle(vec):
    if vec[0] == 0:
        if vec[1] > 0:
            return np.pi / 2
        if vec[1] < 0:
            return np.pi * (3 / 2)
    if vec[1] == 0:
        if vec[0] > 0:
            return 0
        if vec[0] < 0:
            return np.pi
    theta = np.arctan2(vec[1], vec[0])
    return theta


def angleNormalize(in_angle):
    # input: angle in rad
    # output: angle in rad in range[-pi , pi]
    if in_angle >= np.pi:
        out_angle = in_angle - 2 * np.pi
    elif in_angle <= -1 * np.pi:
        out_angle = in_angle + 2 * np.pi
    else:
        out_angle = in_angle
    return out_angle


# -----------------------------------------------


# --------------------------------------------------------3.Planner and Controller Module------------------------------------------------------------------
# class Controller:
#     def __init__(self, pos_cur: np.array, psi_cur: np.array) -> None:
#         self.pos_target = pos_cur
#         self.psi_target = psi_cur
#         self.e_psi_I = 0  # cumulated psi error

#     def setTargetPosition(self, _pos_target: np.array):
#         self.pos_target = _pos_target

#     def turnToDirection(self, psi_target, psi_cur, speed=2):
#         # use PI control
#         p_gain = 0.5
#         i_gain = 0.03
#         e_psi = psi_target - psi_cur
#         self.e_psi_I = np.clip(self.e_psi_I + e_psi, -2.0 / i_gain, 2.0 / i_gain)
#         flag = 0
#         # --------shift the angle so it is from -pi to pi---------
#         if e_psi >= np.pi:
#             e_psi = e_psi - 2 * np.pi
#         if e_psi <= -1 * np.pi:
#             e_psi = e_psi + 2 * np.pi
#         # ---------------------------------------------------------
#         if e_psi < 0.01 and e_psi > -0.01:
#             omega1 = 0
#             omega2 = 0
#             flag = 1
#         else:
#             omega1 = -1.0 * speed * (p_gain * e_psi + i_gain * self.e_psi_I)
#             omega2 = 1.0 * speed * (p_gain * e_psi + i_gain * self.e_psi_I)

#         return omega1, omega2, flag

#     def moveToPosition(
#         self, pos_target: np.array, pos_cur: np.array, psi_cur: float, speed=2
#     ):
#         distance = np.linalg.norm(pos_target - pos_cur)
#         if distance <= 0.01:
#             omega1_forward = 0
#             omega2_forward = 0
#             flag_forward = 1
#             psi_target = psi_cur
#         else:
#             flag_forward = 0
#             rela_pos = pos_target - pos_cur
#             rela_pos_norm = rela_pos / distance
#             theta_rela = vecToAngle(rela_pos_norm)
#             psi_target = theta_rela
#             omega1_forward = speed
#             omega2_forward = speed

#         omega1_turn, omega2_turn, flag_turn = self.turnToDirection(psi_target, psi_cur)
#         omega1 = omega1_turn + omega1_forward
#         omega2 = omega2_turn + omega2_forward
#         flag = flag_forward and flag_turn

#         return omega1, omega2, flag

#     def update(self, pos_cur, psi_cur):
#         omega1, omega2, flag = self.moveToPosition(self.pos_target, pos_cur, psi_cur)
#         return omega1, omega2


class LocalPlanner:
    def __init__(
        self, name, pos_cur: Tuple, psi_cur: float, polygons_static: List[Polygon]
    ) -> None:
        self.name = name
        self.pos_target = pos_cur
        self.psi_target = psi_cur
        self.polygons_static: List[Polygon] = polygons_static

    def SetLocalGoal(self, pos_goal: Tuple):
        self.pos_target = pos_goal

    def update(
        self,
        pos_cur: Tuple,
        psi_cur: float,
        v_cur: Tuple,
        w_cur: float,
        polygons_dynamic: List[Polygon],
    ):
        if polygons_dynamic == None:
            polygons_dynamic = []

        # 1.update str tree
        polygons = self.polygons_static + polygons_dynamic
        str_tree = STRtree(polygons)
        # 2. calculate omega1 omega2 using dynamic window approach

        omega1_opti, omega2_opti = DWA_update(
            pos_cur, psi_cur, self.pos_target, v_cur, w_cur, str_tree
        )
        # print(
        #     "[{0}][Mode2]LocalPlannerUpdate,pos_target:{1}".format(
        #         self.name, self.pos_target
        #     )
        # )
        return omega1_opti, omega2_opti


class GlobalPlanner:
    def __init__(
        self,
        name: str,
        pos_goal: Tuple,
        _path: List[Tuple],
        polygons_static: List[Polygon],
    ) -> None:
        self.name = name
        self.pos_goal = pos_goal
        self.path = _path
        self.plan = _path
        self.finishedFlag = 0
        self.polygons_static: List[Polygon] = polygons_static
        self.mode1_flag = 0
        print(
            "[{0}]GlobalPlannerInitialized,self.plan:{1}".format(self.name, self.plan)
        )

    def changePath(self, _path: List[Tuple]):
        print("[{0}]GlobalPlannerChangePath,self.plan:{1}".format(self.name, self.plan))
        self.path = _path
        self.plan = _path

    def update(
        self,
        localPlanner: LocalPlanner,
        pos_cur: np.array,
        psi_cur: float,
    ):
        # -------------Error Handle-----------
        if self.plan == None:
            print("[{0}]Error Plan is None".format(self.name))
            self.finishedFlag = 1
            return self.finishedFlag

        # -----------When the plan is finished and there some distance to final goal-------
        if len(self.plan) == 0:
            # plan finish
            if np.linalg.norm(pos_cur - self.pos_goal) < 0.01:
                # reach goal
                print("[{0}]Reach Goal!".format(self.name))
                self.finishedFlag = 1
                return self.finishedFlag
            else:
                # still have some distance
                localPlanner.SetLocalGoal(self.pos_goal)
                self.finishedFlag = 0
                return self.finishedFlag

        # ------change plan--------------
        if np.linalg.norm(pos_cur - self.plan[0]) < 0.6:
            print("ChangePlan!RemovePoint{0}".format(self.plan[0]))
            # reach a point in path, we remove point from plan
            self.plan.remove(self.plan[0])
            if len(self.plan) == 0:
                print("[{0}]Plan Finished!".format(self.name))
                localPlanner.SetLocalGoal(self.pos_goal)
                return self.finishedFlag
            else:
                localPlanner.SetLocalGoal(self.plan[0])

        # print(
        #     "[{0}]GlobalPlannerUpdate,cur_target:{1},cur_pos:{2}".format(
        #         self.name, self.plan[0], pos_cur
        #     )
        # )
        return self.finishedFlag


# ------------------------------------------------------------------------FMT------------------------------------------------------------------------


# --------utilities--------------------------
def plotPointList(ax, point_list, style="b."):
    x_coords = [point[0] for point in point_list]  # Extract x coordinates
    y_coords = [point[1] for point in point_list]  # Extract y coordinates
    ax.plot(x_coords, y_coords, style)


def plotLine(ax, p1, p2):
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    ax.plot(x, y, "r-")


def plotPolygon(ax, poly: Polygon):
    coord_s = poly.exterior.coords[0]
    coord_prev = coord_s
    for coord in poly.exterior.coords:
        if coord == coord_prev:
            continue
        # print("coord_prev:{0}".format(coord_prev))
        # print("coord:{0}".format(coord))
        plotLine(ax, p1=coord_prev, p2=coord)
        coord_prev = coord


def plot_circle(ax, center, radius):
    # Create a circle
    circle = plt.Circle(center, radius, edgecolor="b", facecolor="none")

    # Plot the circle
    ax.add_patch(circle)


# Function to generate Halton sequence for a single dimension
def halton_sequence(size, base):
    sequence = np.zeros(size)
    for i in range(size):
        n, denom = i + 1, 1
        while n > 0:
            denom *= base
            sequence[i] += (n % base) / denom
            n //= base
    return sequence


# Function to scale and shift Halton sequence to a specific range
def scale_halton_sequence(sequence, min_val, max_val):
    return min_val + sequence * (max_val - min_val)


# --------Define Data Structure necessary for the Algorithm-----------------


class Node:
    def __init__(self, x, y):
        self.id = None
        self.pos = (x, y)
        self.costToTerminal = 0.0
        self.costToReachInGraph = 9999999.0
        self.parent: Node = None
        self.childrens: List[Node] = []


class SampleSet:
    def __init__(self) -> None:
        pass


def manhattan_distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return abs(x1 - x2) + abs(y1 - y2)


def norm(p1, p2):
    d = np.linalg.norm(np.array(p1) - np.array(p2))
    return d


idToNodeDict: dict[int:Node] = {}
posTupleToID: dict[Tuple:int] = {}


def neighbourPointsQuery_shapely(
    query_point_shapely: Point, r: float, pointsSet_shapely: List[Point]
) -> List[Point]:
    query_circle = query_point_shapely.buffer(r)
    Neighbour_mask = query_circle.contains(pointsSet_shapely)
    points_nparray = np.array(pointsSet_shapely, dtype=object)
    Neighbour_points = points_nparray[Neighbour_mask]
    return Neighbour_points


def shapelyPointListToListTuple(pointsList_shapely: List[Point]) -> List[Tuple]:
    tupleList = []
    for point_shapely in pointsList_shapely:
        tupleList.append((point_shapely.x, point_shapely.y))
    return tupleList


def shapelyPointListToListID(pointsList_shapely: List[Point]) -> List[int]:
    idList = []
    for point_shapely in pointsList_shapely:
        position = (point_shapely.x, point_shapely.y)
        id = posTupleToID[position]
        idList.append(id)
    return idList


def lineSegmentToShapelyPointList(p1, p2, step=0.1):
    ps = np.array([p1[0], p1[1]])
    pe = np.array([p2[0], p2[1]])
    vec_se = pe - ps
    dis = np.linalg.norm(pe - ps)
    vec_se_norm = vec_se / dis
    i = 0
    pointList_s = []
    while i * step <= dis:
        pointList_s.append(Point(ps + i * step * vec_se_norm))
        i = i + 1
    return pointList_s


def FMT_Search(_p_init: Tuple, _p_goal: Tuple, tree: STRtree):
    # Define the bounds
    x_min, x_max = -11, 11
    y_min, y_max = -11, 11

    # Number of points to sample
    num_points = 3000

    # Generate Halton sequences for x and y dimensions (using different prime bases)
    x_sequence = halton_sequence(num_points, 2)
    y_sequence = halton_sequence(num_points, 3)

    # Scale the Halton sequences to the desired range (-15 to 15)
    x_coords = scale_halton_sequence(x_sequence, x_min, x_max)
    y_coords = scale_halton_sequence(y_sequence, x_min, x_max)

    points_safe = []
    points = list(zip(x_coords, y_coords))
    r = 0.7
    for point in points:
        index_query, distance = tree.query_nearest(Point(point), return_distance=True)
        if distance[distance <= r].size == 0:
            points_safe.append(point)

    # ------------initialization------------------
    points_safe_shapely = []
    for point in points_safe:
        points_safe_shapely.append(Point(point))

    # set initial and goal position
    p_init = _p_init
    p_goal = _p_goal

    # ---initialize containers--
    V: dict = {}
    V_unvisited: List[int] = []
    V_closed: List[int] = []

    # initialize initial node
    Node_init = Node(p_init[0], p_init[1])
    Node_init.costToTerminal = manhattan_distance(Node_init.pos, p_goal)
    Node_init.costToReachInGraph = 0.0
    Node_init.id = 0
    idToNodeDict[Node_init.id] = Node_init
    posTupleToID[p_init] = Node_init.id

    i = 0
    for point in points_safe:
        i = i + 1
        node = Node(point[0], point[1])
        disToGoal = manhattan_distance(point, p_goal)
        # ----------initialize node attributes---------------------
        # print(i)
        node.id = i
        node.costToTerminal = disToGoal
        # ---------------------------------------------------------
        V[node.id] = node
        idToNodeDict[node.id] = node
        posTupleToID[point] = node.id
        V_unvisited.append(node.id)

    # initialize goal node
    Node_goal = Node(p_goal[0], p_goal[1])
    Node_goal.costToTerminal = 0
    Node_goal.id = i + 1
    idToNodeDict[Node_goal.id] = Node_goal
    posTupleToID[p_goal] = Node_goal.id
    print(Node_goal.id)

    # add initial and goal point to points_safe and points_safe_shapely
    points_safe.append(p_init)
    points_safe.append(p_goal)
    points_safe_shapely.append(Point(p_init))
    points_safe_shapely.append(Point(p_goal))

    # ---initialize containers--
    V_open = pqdict()  # key:id ; value: cost to goal
    V_open[Node_init.id] = 0.0

    r_neighbour = 2

    while len(V_open) != 0:
        z_id = V_open.top()  # 2:Find lowest-cost node z in V_open
        z = idToNodeDict[z_id]
        N_z_shapely = neighbourPointsQuery_shapely(
            query_point_shapely=Point(z.pos),
            r=r_neighbour,
            pointsSet_shapely=points_safe_shapely,
        )
        N_z_id = shapelyPointListToListID(N_z_shapely)
        X_sets_id = list(set(N_z_id) & set(V_unvisited))
        for x_id in X_sets_id:  # 3: For each of z's neighbours x in V_unvisited
            x = idToNodeDict[x_id]
            N_x_shapely = neighbourPointsQuery_shapely(
                query_point_shapely=Point(x.pos),
                r=r_neighbour,
                pointsSet_shapely=points_safe_shapely,
            )
            N_x_id = shapelyPointListToListID(N_x_shapely)
            Y_sets_id = list(
                set(N_x_id) & set(V_open)
            )  # 4. Find Neighbour of Node x, also in V_open,  denoted as Y
            Y_sets_cost = [
                V_open[y_id] for y_id in Y_sets_id
            ]  # Note: What is locally optimal?
            y_min_id = Y_sets_id[
                np.argmin(Y_sets_cost)
            ]  # 5. Find locally-optimal one-step connection to x from among nodes y
            y_min = idToNodeDict[y_min_id]
            # 6. if that connection is collision-free add it to tree of path
            segment_shapely_PointList = lineSegmentToShapelyPointList(
                p1=x.pos, p2=y_min.pos
            )
            _, distance = tree.query_nearest(
                segment_shapely_PointList, return_distance=True
            )
            if distance[distance <= r].size == 0:
                # no collision, add this edge to the graph
                # set y_min node as parent of x
                x.parent = y_min
                y_min.childrens.append(x)
                # 7. remove successfully connected nodes x from V_unvisited and add them to V_open
                V_unvisited.remove(x_id)
                V_open[x_id] = (
                    V_open[y_min_id] + norm(y_min.pos, x.pos) + x.costToTerminal
                )

        # 8:Remove z from V_open and add it to V_closed
        V_open.pop(z_id)
        V_closed.append(z_id)

        # 9: Repeat until either:(1)V_open is empty->report failure(2)lowesr-cost node z in V_open is in X_goal return unique path to z and erport success
        if len(V_open) == 0:
            print("[INFO][FMT Search]search Fail")
            break
        if idToNodeDict[V_open.top()].costToTerminal < 0.4:
            z_id = V_open.pop()
            V_closed.append(z_id)
            print("[INFO][FMT Search]search Success")
            node_final = idToNodeDict[z_id]
            node = node_final
            path = []
            while node.id != 0:
                path = [node.pos] + path
                node = node.parent
            path = [node.pos] + path
            return path
    return None


# --------------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------DWA ---------------------------------------------------------------------
# calculate trajectory
class State:
    def __init__(self) -> None:
        self.pos: Tuple = None
        self.dir: float = None
        self.v: float = None
        self.w: float = None

    # def plot(self, ax):
    #     plot_robot(ax, pos=self.pos, psi=self.dir)


def calculateTrajectory(
    pos_init,
    psi_init: float,
    v: float,
    w: float,
    T: float,
    dt=0.1,
) -> List[State]:
    # Calculate the Trajectory given:
    # We assume velocity and angular velocity
    # inputs:
    # initial position in global frame: pos_init [m]
    # initial direction in global frame: psi_init [rad][0,2pi]
    # forward velocity in robot frame: v [m/s]
    # angular velocity in robot frame: w [rad/s]
    # time horizon: T [s]
    # time resolution: dt [s]

    # outputs:
    # trajectory: Tra
    # -----input type handle-------
    if isinstance(pos_init, np.ndarray):
        pos_init = pos_init.flatten()
    # -----------------------------

    x = pos_init[0]
    y = pos_init[1]
    theta = psi_init

    num_steps = round(T / dt)

    Tra: List[State] = []
    for i in range(num_steps):
        x = x + v * dt * np.cos(theta)
        y = y + v * dt * np.sin(theta)
        theta = theta + w * dt

        # record the trajectory
        s = State()
        s.pos = (x, y)
        s.dir = theta
        s.v = v
        s.w = w
        Tra.append(s)

    return Tra


def TrajectoryToShapelyPointList(Tra: List[State]) -> List[Point]:
    pointList_shapely = []
    for state in Tra:
        pointList_shapely.append(Point(state.pos))
    return pointList_shapely


def omegaPairToVelocity(omega1: float, omega2: float):
    # Note: for per unit d_omega=omega2-omega1, the angular velocity w = 0.2 rad/s
    w = (omega2 - omega1) * 0.2
    v = (omega2 + omega1) / 2.0
    return v, w


def velocityToOmegaPair(v: float, w: float):
    omega1 = (2 * v - 5 * w) / 2
    omega2 = (5 * w + 2 * v) / 2
    return omega1, omega2


# if a trajectory hit an obstacle, return True
# Note: radius of robot is 0.35m, we identified as collision if the centre to obstacle < 0.5m
# str_tree contains all the obstacles
def trajectoryCollision(Tra: List[State], str_tree: STRtree) -> bool:
    pointList_shapely = TrajectoryToShapelyPointList(Tra)
    index_query, distance = str_tree.query_nearest(
        pointList_shapely, return_distance=True
    )
    if distance[distance <= 0.63].size == 0:
        # No Collision
        return False, min(distance)
    else:
        return True, 0


# Evaluate every admissible trajectory and return the optimal one
# Evaluate According to:
# (1) distance of final state position to target
# (2) difference of final state heading to ideal heading
# (3) line velocity, larger is better
def normalize_array_minmax(arr: np.array):
    normalized = np.interp(arr, (arr.min(), arr.max()), (0, 1))
    return normalized


def scorePosition(pos_final: Tuple, pos_target: Tuple) -> float:
    eps = 0.7
    dis = np.linalg.norm(np.array(pos_final) - np.array(pos_target))
    score = 1 / (dis + eps)
    return score


def scoreDistanceToObstacle(d: float) -> float:
    L = 1
    if d < L:
        score = d
    else:
        score = L
    return score


def scoreHeading(psi_final: float, pos_final: Tuple, pos_target: Tuple) -> float:
    vec_relative = np.array(pos_target) - np.array(pos_final)
    psi_ideal = vecToAngle(vec_relative)

    psi_error = abs(angleNormalize(psi_final - psi_ideal))
    score = np.pi - psi_error
    return score


def scoreVelocity(v) -> float:
    score = np.abs(v)
    return score


def CalculateOptimalTrajectory(
    TraList_Safe: List[List[State]], TraList_Safe_dist: List[float], pos_target: Tuple
) -> List[State]:
    # Hyperparameter
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    eta = 1.0

    num_tra = len(TraList_Safe)

    scorePositionArr: np.array = np.zeros(num_tra)
    scoreHeadingArr: np.array = np.zeros(num_tra)
    scoreVelocityArr: np.array = np.zeros(num_tra)
    scoreDistanceArr: np.array = np.zeros(num_tra)

    for i in range(num_tra):
        Tra = TraList_Safe[i]
        s_final = Tra[-1]
        pos_final = s_final.pos
        psi_final = s_final.dir
        v = s_final.v

        scorePositionArr[i] = scorePosition(pos_final, pos_target)
        scoreHeadingArr[i] = scoreHeading(psi_final, pos_final, pos_target)
        scoreVelocityArr[i] = scoreVelocity(v)
        scoreDistanceArr[i] = scoreDistanceToObstacle(TraList_Safe_dist[i])

    # Normalize Each Score Array
    # print("[Debug]num_tra:{0}".format(num_tra))
    scorePositionArr_normalized = normalize_array_minmax(scorePositionArr)
    scoreHeadingArr_normalized = normalize_array_minmax(scoreHeadingArr)
    scoreVelocityArr_normalized = normalize_array_minmax(scoreVelocityArr)
    scoreDistArr_normalized = normalize_array_minmax(scoreDistanceArr)

    # calculate the final score
    scoreArr = (
        alpha * scorePositionArr_normalized
        + beta * scoreHeadingArr_normalized
        + gamma * scoreVelocityArr_normalized
        + eta * scoreDistArr_normalized
    )

    index_max = np.argmax(scoreArr)
    return TraList_Safe[index_max]


def DWA_update(
    pos_cur: Tuple,
    psi_cur: float,
    pos_target: Tuple,
    v_cur: Tuple,
    w_cur: float,
    tree: STRtree,
):
    # parameters
    T = 0.4  # sample horizon[s]
    pos_init = pos_cur
    psi_init = psi_cur
    # print(
    #     "[DWA_update]pos_cur:{0},psi_cur:{1},pos_target:{2},v_cur:{3},w_cur:{4}".format(
    #         pos_cur, psi_cur, pos_target, v_cur, w_cur
    #     )
    # )
    # -------------1. Sample Velocities------------------------------
    # the limit on each omega is[-5,5]

    resolution = 20
    num_point = resolution**2
    # Define the ranges for x and y
    x_range = np.linspace(-5, 5, resolution)  # Adjust the number of points as needed
    y_range = np.linspace(-5, 5, resolution)  # Adjust the number of points as needed

    # Create a grid of points
    x, y = np.meshgrid(x_range, y_range)

    # Flatten the grid to get a list of points
    points = np.column_stack((x.flatten(), y.flatten()))

    # we only filter out velocity that violate acceleration constrain

    # --------------define current velocity------------
    # v_cur = 1.0
    # w_cur = 0.0

    dt = 0.1
    v_dot_up = 10  # the upper bound on line acceleration i.e. maximum acceleration
    v_dot_low = 10  # the lower bound on line acceleration i.e. maximum deceleration
    w_dot_up = 20  # the upper bound on angular acceleration i.e. maximum acceleration
    w_dot_low = 20  # the lower bound on angular acceleration i.e. maximum acceleration
    velocity_points = []
    for point in points:
        omega1 = point[0]
        omega2 = point[1]
        v, w = omegaPairToVelocity(omega1, omega2)

        if (
            v_cur - v_dot_low * dt <= v
            and v <= v_cur + v_dot_up * dt
            and w_cur - w_dot_low * dt <= w
            and w <= w_cur + w_dot_up * dt
        ):
            velocity_points.append(point)
    # -------------2. Calculate Trajectory from sampling and abandon infeasible one-----------------------------------------
    TraList_Safe: List[List[State]] = []
    TraList_Safe_dist: List[float] = []
    for i in range(len(velocity_points)):
        omega1 = velocity_points[i][0]
        omega2 = velocity_points[i][1]
        v, w = omegaPairToVelocity(omega1, omega2)
        Tra = calculateTrajectory(pos_init, psi_init, v, w, T)
        # If there is collision , we abandom this trajectory
        collision_flag, dis = trajectoryCollision(Tra, str_tree=tree)
        if not collision_flag:
            # If no collision, we add it to TraList_Safe
            TraList_Safe.append(Tra)
            TraList_Safe_dist.append(dis)

    num_tra = len(TraList_Safe)
    # print("[Debug]num_tra:{0}".format(num_tra))
    # -------------no safe velocity------
    if num_tra == 0:
        return 0.0, 0.0

    Tra_optimal = CalculateOptimalTrajectory(
        TraList_Safe, TraList_Safe_dist, pos_target
    )
    v_opti = Tra_optimal[0].v
    w_opti = Tra_optimal[0].w
    omega1_opti, omega2_opti = velocityToOmegaPair(v=v_opti, w=w_opti)
    return omega1_opti, omega2_opti


def getInitialPosition(name: str) -> Tuple:
    if name == "PDM4AR_1":
        pos_init = (0, 10)
    elif name == "PDM4AR_2":
        pos_init = (-2, -4)
    elif name == "PDM4AR_3":
        pos_init = (-9, 9)
    elif name == "PDM4AR_4":
        pos_init = (-9, -9)

    return pos_init


def getMapEdgePolygon() -> List[Polygon]:
    polygons = []
    coords1 = ((-11, -11), (11, -11), (11, -12), (-11, -12))  # lower edge
    polygons.append(Polygon(coords1))
    coords2 = ((-11, 11), (11, 11), (11, 12), (-11, 12))  # upper edge
    polygons.append(Polygon(coords2))
    coords3 = ((-11, -11), (-11, 11), (-12, 11), (-12, -11))  # left edge
    polygons.append(Polygon(coords3))
    coords4 = ((11, -11), (11, 11), (12, 11), (12, -11))  # left edge
    polygons.append(Polygon(coords4))
    return polygons


# --------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    def __init__(self):
        # feel free to remove/modify  the following
        self.t = 0
        self.params = Pdm4arAgentParams()
        self.replan_interval = 1
        self.countDown_Replan = self.replan_interval  # [s]
        self.initializeFinishedFlag = 0
        self.velocity_line: float = 0.0
        self.velocity_angular: float = 0.0

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.static_obstacles = list(init_obs.dg_scenario.static_obstacles)
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: DiffDriveState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        # todo implement here your planning stack
        self.t += 0.1
        # print("[INFO] time:{0}".format(self.t))
        # -----------------------Method Used to Debug--------------------
        # if self.t > 4.1:
        #     return DiffDriveCommands(omega_l=0, omega_r=0)

        # ---------------------------------------------------------------

        name = self.name
        player_x = sim_obs.players[name].state.x  # CoG x location [m]
        player_y = sim_obs.players[name].state.y  # CoG y location [m]
        player_psi = sim_obs.players[name].state.psi  # CoG heading [rad] from 0 to 2*pi
        pos_cur = np.array([player_x, player_y])
        pos_cur_tuple = (player_x, player_y)
        psi_cur = player_psi

        if not self.initializeFinishedFlag:
            print("[{0}]StartInitialization!".format(self.name))
            # -------------extract initial and goal position-----------
            pos_init = pos_cur_tuple

            poly_goal = self.goal.goal
            pos_goal = (poly_goal.centroid.x, poly_goal.centroid.y)
            print("[{0}]pos_goal:{1}".format(self.name, pos_goal))

            # ------------set initial str tree for collision detect---------
            polygons = []
            polygons_mapEdge = getMapEdgePolygon()
            polygons = polygons + polygons_mapEdge
            for obstacle in self.static_obstacles:
                if type(obstacle.shape) == Polygon:
                    polygons.append(obstacle.shape)
            str_tree = STRtree(polygons)

            # ---------------initialize planner and controller---------

            psi_init = 1
            initial_path = FMT_Search(_p_init=pos_init, _p_goal=pos_goal, tree=str_tree)
            self.globalPlanner = GlobalPlanner(
                self.name, pos_goal, initial_path, polygons_static=polygons
            )
            self.localPlanner = LocalPlanner(
                self.name, pos_cur=pos_init, psi_cur=psi_init, polygons_static=polygons
            )
            # self.controller = Controller(
            #     pos_cur=np.array(pos_init), psi_cur=np.array(psi_init)
            # )
            # ---------------------------------------------------------
            self.initializeFinishedFlag = 1

        # -------extract other players polygon-------
        polygons_dynamics = []
        players = sim_obs.players
        for player_name in players:
            if player_name != name:
                poly_otherPlayer = sim_obs.players[player_name].occupancy
                polygons_dynamics.append(poly_otherPlayer)
        # -------rebuild str tree-------------
        polygons = self.globalPlanner.polygons_static + polygons_dynamics
        str_tree = STRtree(polygons)
        # ------update countdown for replanning-----
        self.countDown_Replan = self.countDown_Replan - 0.1
        if self.countDown_Replan <= 0.1:
            self.countDown_Replan = self.replan_interval
            # update plan in planner

            # ------------Debug  Use----------------------
            if self.name == "PDM4AR_2":
                print("[INFO] time:{0}".format(self.t))
                print(
                    "[{0}]pos_cur_tuple = {1},psi_cur = {2}, self.globalPlanner.pos_goal = {3},self.localPlanner.pos_target={4} , self.velocity_line = {5},self.velocity_angular ={6}".format(
                        self.name,
                        pos_cur_tuple,
                        psi_cur,
                        self.globalPlanner.pos_goal,
                        self.localPlanner.pos_target,
                        self.velocity_line,
                        self.velocity_angular,
                    )
                )
            # --------------------------------------------

            # path = FMT_Search(
            #     _p_init=pos_cur_tuple,
            #     _p_goal=self.globalPlanner.pos_goal,
            #     tree=str_tree,
            # )
            # if path is not None:
            #     self.globalPlanner.changePath(path)

        # -------------------------------------------

        finished_flag = self.globalPlanner.update(self.localPlanner, pos_cur, psi_cur)
        if not finished_flag:
            omega1, omega2 = self.localPlanner.update(
                pos_cur=pos_cur_tuple,
                psi_cur=psi_cur,
                v_cur=self.velocity_line,
                w_cur=self.velocity_angular,
                polygons_dynamic=polygons_dynamics,
            )
        else:
            omega1 = 0
            omega2 = 0

        # Record velocity in this step
        v, w = omegaPairToVelocity(omega1, omega2)
        self.velocity_line = v
        self.velocity_angular = w
        # Note: omega limit: min/max rotational velocity of wheels [rad/s] = (-5, 5)
        # Note: for per unit d_omega=omega2-omega1, the angular velocity w = 0.2 rad/s
        # Note: radius is 0.6
        return DiffDriveCommands(omega_l=omega1, omega_r=omega2)
