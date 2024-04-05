from abc import abstractmethod, ABC
from typing import Tuple

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes
from typing import Optional, TypeVar, Set, Mapping, Tuple, List, Dict

class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        # todo implement here your solution
        
        #0. Define Some Containers
        Parents : Mapping[X , Optional[X]] = {}
        Q : List[X] = []
        V : List[X] = []
        OpenedNodes : List[X] = []
        
        #1. Processing the first node
        Q.append(start)
        V.append(start)
        Parents[start] = ""
        
        #2. Looping to pop node in Q
        while(Q):
            U = Q.pop()

            OpenedNodes.append(U)
            if U == goal:
                
                break
            graph_U_list = list(graph[U])
            for N in reversed(graph_U_list):
                if N in V:
                    continue
                else:
                    
                    Q.append(N)
                    V.append(N)
                    Parents[N] = U
                    
        # Check if a path was found
        Path : List[X] = []
        if goal not in OpenedNodes:

            return [], OpenedNodes
                    
        #3. Starting from the goal back to start to find the path
        
        
        parentNode = Parents[goal]
        Path.append(goal)
        while(parentNode != ""):
            Path.insert(0, parentNode)#insert at the front
            parentNode = Parents[parentNode]
            
                    
        return Path, OpenedNodes
  
    
class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        
        #0. Define Some Containers
        Parents : Mapping[X , Optional[X]] = {}
        Q : List[X] = []
        V : List[X] = []
        OpenedNodes : List[X] = []
        
        #1. Processing the first node
        Q.append(start)
        V.append(start)
        Parents[start] = ""
        
        #2. Looping to pop node in Q
        while(Q):
            U = Q.pop(0)

            OpenedNodes.append(U)
            if U == goal:
           
                break
            
            for N in graph[U]:
                if N in V:
                    continue
                else:
                    
                    Q.append(N)
                    V.append(N)
                    Parents[N] = U
                    
        # Check if a path was found
        Path : List[X] = []
        if goal not in OpenedNodes:

            return [], OpenedNodes
                    
        #3. Starting from the goal back to start to find the path
        
        
        parentNode = Parents[goal]
        Path.append(goal)
        while(parentNode != ""):
            Path.insert(0, parentNode)#insert at the front
            parentNode = Parents[parentNode]
            

                    
        return Path, OpenedNodes  
     
class IterativeDeepening(GraphSearch):
    def DFSSearchWithDepth(self, graph: AdjacencyList, start: X, goal: X, depthGoal: int) -> Tuple[Path, OpenedNodes]:
        #0. Define Some Containers
        Parents : Mapping[X , Optional[X]] = {}
        Q : List[X] = []
        V : List[X] = []
        OpenedNodes : List[X] = []
        D : Mapping[X, Optional[int]] = {}
        
        #1. Processing the first node
        Q.append(start)
        V.append(start)
        D[start] = 0
        Parents[start] = ""
        
        #2. Looping to pop node in Q
        while(Q):
            U = Q.pop()
            
            # if the depth of U larger than the depthGoal, then skip
            if D[U] > depthGoal:
                continue

            OpenedNodes.append(U)
            if U == goal:
                break

            graph_U_list = list(graph[U])
            for N in reversed(graph_U_list):
                if N in V:
                    continue
                else:
                    
                    Q.append(N)
                    V.append(N)
                    Parents[N] = U
                    D[N] = D[U] + 1
                    
        # Check if a path was found
        Path : List[X] = []
        if goal not in OpenedNodes:

            return [], OpenedNodes
                    
        #3. Starting from the goal back to start to find the path
        
        
        parentNode = Parents[goal]
        Path.append(goal)
        while(parentNode != ""):
            Path.insert(0, parentNode)#insert at the front
            parentNode = Parents[parentNode]
            
                    
        return Path, OpenedNodes
        
        
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        # todo implement here your solution
        d = 0
        OpenedNodes: List[X] = []
        maxCount = 100
        count = 0
        while(count<maxCount):

            (Path, OpenedNodes) = self.DFSSearchWithDepth( graph, start, goal, depthGoal = d)
            if Path != []:

                return Path, OpenedNodes
            
            d += 1  
            count += 1
     
        return [], OpenedNodes  
