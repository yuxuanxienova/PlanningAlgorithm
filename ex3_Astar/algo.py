from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq    # you may find this helpful

from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed
from abc import abstractmethod, ABC
from typing import Optional, TypeVar, Set, Mapping, Tuple, List, Dict
from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes

import math

@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass

@dataclass
class UniformCostSearch(InformedGraphSearch):
    
    def path(self, start: X, goal: X) -> Path:
        # todo
        #0. Define Some Containers,and initialize them
        
        Parents : Mapping[X , Optional[X]] = {}
        CostToReach : Mapping[X , Optional[float]] = {}
        Q = []
        OpenedNodes : List[X] = []
        
        #initialize
        for node in self.graph.adj_list:
            CostToReach[node] = 999999
        
        #1. Processing the first node
        heapq.heappush(Q, (0 , start))
        CostToReach[start] = 0
        Parents[start] = ""
        
        #2. Looping to pop node in Q
        while(Q):
            U = heapq.heappop(Q)[1]
            # print("visit:{0}".format(U))
            OpenedNodes.append(U)
            if U == goal:
                # print("reach the goal!")
                break
            
            graph_U_list = list(self.graph.adj_list[U])
            for N in reversed(graph_U_list):
                newCostToReachN = CostToReach[U] + self.graph.weights[(U , N)]
                if( newCostToReachN < CostToReach[N]):
                    CostToReach[N] = newCostToReachN  
                    Parents[N] = U
                    #we update the priority of N if it is already in the queue                 
                    if( N in Q ):
                            for priority, value in Q:
                                if(N == value):
                                    Q[count] = (newCostToReachN , N)
                                count+=1
                    #else we add N to the queue
                    else:
                        heapq.heappush(Q, (newCostToReachN, N))
                    
        # Check if a path was found
        Path : List[X] = []
        if goal not in OpenedNodes:
            # print("Path:{0}".format(Path))
            # print("OpenedNodes:{0}".format(OpenedNodes))
            return [], OpenedNodes
                    
        #3. Starting from the goal back to start to find the path
        
        
        parentNode = Parents[goal]
        Path.append(goal)
        while(parentNode != ""):
            Path.insert(0, parentNode)#insert at the front
            parentNode = Parents[parentNode]
            
        # print("Path:{0}".format(Path))
        # print("OpenedNodes:{0}".format(OpenedNodes))
                    
        return Path

@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0
    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def heuristic(self, u: X, v: X) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        # todo
        u_pos = self.graph.get_node_coordinates(u)
        v_pos = self.graph.get_node_coordinates(v)
        dis_relative_x = u_pos[0] - v_pos[0]
        dis_relative_y = u_pos[1] - v_pos[1]
        distance = (dis_relative_x**2 + dis_relative_y**2)**0.5
        
        
        
        return 0
        
    def path(self, start: X, goal: X) -> Path:
        # todo
        #0. Define Some Containers,and initialize them
        
        Parents : Mapping[X , Optional[X]] = {}
        CostToReach : Mapping[X , Optional[float]] = {}
        Q = []
        OpenedNodes : List[X] = []
        
        #initialize
        for node in self.graph.adj_list:
            CostToReach[node] = 999999
        
        #1. Processing the first node
        heapq.heappush(Q, (0 , start))
        CostToReach[start] = 0
        Parents[start] = ""
        
        #2. Looping to pop node in Q
        while(Q):
            U = heapq.heappop(Q)[1]
            # print("visit:{0}".format(U))
            OpenedNodes.append(U)
            if U == goal:
                # print("reach the goal!")
                break
            graph_U_list = list(self.graph.adj_list[U])
            for N in reversed(graph_U_list):
                newCostToReachN = CostToReach[U] + self.graph.weights[(U , N)]
                if( newCostToReachN < CostToReach[N]):
                    CostToReach[N] = newCostToReachN  
                    Parents[N] = U
                    #we update the priority of N if it is already in the queue                 
                    if( N in Q ):
                            for priority, value in Q:
                                if(N == value):
                                    Q[count] = (newCostToReachN + self.heuristic(U,N), N)
                                count+=1
                    #else we add N to the queue
                    else:
                        heapq.heappush(Q, (newCostToReachN + self.heuristic(U,N), N))
                    
        # Check if a path was found
        Path : List[X] = []
        if goal not in OpenedNodes:
            # print("Path:{0}".format(Path))
            # print("OpenedNodes:{0}".format(OpenedNodes))
            return [], OpenedNodes
                    
        #3. Starting from the goal back to start to find the path
        
        
        parentNode = Parents[goal]
        Path.append(goal)
        while(parentNode != ""):
            Path.insert(0, parentNode)#insert at the front
            parentNode = Parents[parentNode]
            
        # print("Path:{0}".format(Path))
        # print("OpenedNodes:{0}".format(OpenedNodes))
                    
        return Path


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
