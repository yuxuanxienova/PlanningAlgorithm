from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc
#------------------
from pdm4ar.exercises.ex04.structures import Cell
def OutOfMap(state:State, grid: NDArray[np.int64])-> bool:
    if(state[0]>=np.shape(grid)[0] or state[1]>=np.shape(grid)[1] or state[0] <0 or state[1] <0):
        return True
    else:
        return False
def CalculateNextState(cur_state:State , act_taken: Action, startState:State ) -> State:
    next_state: State 
    if(act_taken == Action.EAST):
        next_state = (cur_state[0],cur_state[1] + 1)
        return next_state
    if(act_taken == Action.SOUTH):
        next_state = (cur_state[0] + 1,cur_state[1])
        return next_state
    if(act_taken == Action.WEST):
        next_state = (cur_state[0],cur_state[1] - 1)
        return next_state
    if(act_taken == Action.NORTH):
        next_state = (cur_state[0] - 1,cur_state[1])
        return next_state
    if(act_taken == Action.STAY):
        next_state = (cur_state[0],cur_state[1])
        return next_state
    if(act_taken == Action.ABANDON):
        next_state = startState
        return next_state
def StartPosOfMap(grid: NDArray):
    for i in range(np.shape(grid)[0]):
        for j in range(np.shape(grid)[1]):
            if(grid[i][j] == Cell.START):
                return (i , j)
def ActionTaken(state: State, next_state: State, startState: State ) -> Action:
    if(next_state[0] == state[0]  and next_state[1] == state[1] + 1):#Move to East
        return Action.EAST
    if(next_state[0] == state[0] + 1 and next_state[1] == state[1]):#Move to South
        return Action.SOUTH
    if(next_state[0] == state[0]  and next_state[1] == state[1] - 1):#Move to West
        return Action.WEST
    if(next_state[0] == state[0] -1 and next_state[1] == state[1] ):#Move to North
        return Action.NORTH
    if(next_state[0] == state[0]  and next_state[1] == state[1]):#Stay
        return Action.STAY
    #if the current state is not near the start state but end out next state in the start, we regard it as abandon
    if(~IsNeighbourOf(state, startState) and next_state == startState):#Abandon
        return Action.ABANDON
    #error handle
    print("[ERROR][INVALID ACTION] state:{0} ; next_state:{1}".format(state,next_state))
    return -1

def IsNeighbourOf(A: State, B:State) -> bool:
    if(abs(A[0]-B[0])==1 and abs(A[1]-B[1])==1):
        return False
    if(abs(A[0]-B[0])>1 or abs(A[1]-B[1])>1):
        return False
    return True            
#-------------------
class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""
        
        self.startState = StartPosOfMap(self.grid)
        
        self.FlagPrintInfo = False
        
        #-------define action_actionTaken matrix in different cell----------
        #--------------stores transition probability---------------------
        # Action List:
        # NORTH = 0
        # WEST = 1
        # SOUTH = 2
        # EAST = 3
        # STAY = 4
        # ABANDON = 5
        #----------------------------------------------------------------
        # A_A_Map
        #dim1: 6 (action choice)
        #dim2: 6 (action actually take)
        #----------------------------------------------------------------
        
        #0. Out of the Map
        self.A_A_Map_OUT = np.zeros((6,6))
        self.A_A_Map_OUT[:][Action.ABANDON] = 1
        
        #1. In GRASS:
        self.A_A_Map_GRASS = np.zeros((6,6))
        self.A_A_Map_GRASS = np.array([[   0.75   ,    0.25/3  ,   0.25/3  ,  0.25/3  ,  0  ,  0  ],
                                       [  0.25/3  ,     0.75   ,   0.25/3  ,  0.25/3  ,  0  ,  0  ],
                                       [  0.25/3  ,    0.25/3  ,    0.75   ,  0.25/3  ,  0  ,  0  ],
                                       [  0.25/3  ,    0.25/3  ,   0.25/3  ,    0.75  ,  0  ,  0  ],
                                       [    0     ,       0    ,     0     ,     0    ,  0  ,  0  ],
                                       [    0     ,       0    ,     0     ,     0    ,  0  ,  1  ]])
        #2. In SWAMP:
        self.A_A_Map_SWAMP = np.zeros((6,6)) 
        self.A_A_Map_SWAMP = np.array([[   0.50   ,    0.25/3  ,   0.25/3  ,  0.25/3  ,  0.2  ,  0.05  ],
                                       [  0.25/3  ,     0.50   ,   0.25/3  ,  0.25/3  ,  0.2  ,  0.05  ],
                                       [  0.25/3  ,    0.25/3  ,    0.50   ,  0.25/3  ,  0.2  ,  0.05  ],
                                       [  0.25/3  ,    0.25/3  ,   0.25/3  ,    0.50  ,  0.2  ,  0.05  ],
                                       [    0     ,       0    ,     0     ,     0    ,  0.0  ,  0     ],
                                       [    0     ,       0    ,     0     ,     0    ,  0.0  ,  1     ]])
        #3. In GOAL:
        self.A_A_Map_GOAL = np.zeros((6,6)) 
        self.A_A_Map_GOAL = np.array([[  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                                      [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                                      [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                                      [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ],
                                      [  0  ,  0  ,  0  ,  0  ,  1  ,  0  ],
                                      [  0  ,  0  ,  0  ,  0  ,  0  ,  0  ]])
        #4. In START:
        self.A_A_Map_START = np.zeros((6,6))
        self.A_A_Map_START = np.array([[   0.75   ,    0.25/3  ,   0.25/3  ,  0.25/3  ,  0  ,  0  ],
                                       [  0.25/3  ,     0.75   ,   0.25/3  ,  0.25/3  ,  0  ,  0  ],
                                       [  0.25/3  ,    0.25/3  ,    0.75   ,  0.25/3  ,  0  ,  0  ],
                                       [  0.25/3  ,    0.25/3  ,   0.25/3  ,    0.75  ,  0  ,  0  ],
                                       [    0     ,       0    ,     0     ,     0    ,  0  ,  0  ],
                                       [    0     ,       0    ,     0     ,     0    ,  0  ,  1  ]])
    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:#the action here is the action choice
        """Returns P(next_state | state, action)"""
        # todo
        #0. Out of Map
        if(OutOfMap(state,self.grid)):
            actionTaken = ActionTaken(state, next_state, self.startState)
            return self.A_A_Map_OUT[action][actionTaken]
        #1. In GRASS:
        if(self.grid[state[0]][state[1]] == Cell.GRASS):
            if(self.FlagPrintInfo):print("--------[CheckPoint][get_transition_prob][In GRASS]---------------------------------------------")

            actionTaken = ActionTaken(state, next_state, self.startState)
            if(self.FlagPrintInfo):print("[CkeckPoint] action:{0}".format(action))
            if(self.FlagPrintInfo):print("[CkeckPoint] actionTaken:{0}".format(actionTaken))
            if(self.FlagPrintInfo):print("[CkeckPoint] self.A_A_Map_GRASS[action][actionTaken]:{0}".format(self.A_A_Map_GRASS[action][actionTaken]))
            return self.A_A_Map_GRASS[action][actionTaken]               
        #2. In SWAMP:
        if(self.grid[state[0]][state[1]] == Cell.SWAMP):
            actionTaken = ActionTaken(state, next_state, self.startState)
            return self.A_A_Map_SWAMP[action][actionTaken] 

        #3. In GOAL:
        if(self.grid[state[0]][state[1]] == Cell.GOAL):
            actionTaken = ActionTaken(state, next_state, self.startState)
            return self.A_A_Map_GOAL[action][actionTaken] 

        #4. In START:
        if(self.grid[state[0]][state[1]] == Cell.START):
            actionTaken = ActionTaken(state, next_state, self.startState)
            return self.A_A_Map_START[action][actionTaken] 

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:#the action here is also the action choice
        # todo
        #We need to first find the action actually take
        act_taken:Action = ActionTaken(state, next_state, self.startState)
        #0. Out of Map
        if(OutOfMap(next_state, self.grid)):
            reward = -10.
            return reward

        #1. In GRASS:
        if(self.grid[state[0]][state[1]] == Cell.GRASS):
            if(act_taken == Action.EAST ):
                reward = -1.
                return reward
            if(act_taken == Action.SOUTH):
                reward = -1.
                return reward
            if(act_taken == Action.WEST ):
                reward = -1.
                return reward                
            if(act_taken == Action.NORTH):
                reward = -1.
                return reward
            if(act_taken == Action.STAY ):
                reward = -1.
                return reward
            if(act_taken == Action.ABANDON):
                reward = -10.
                return reward
        #2. In SWAMP:
        if(self.grid[state[0]][state[1]] == Cell.SWAMP):
            if(act_taken == Action.EAST ):
                reward = -2.
                return reward
            if(act_taken == Action.SOUTH):
                reward = -2.
                return reward
            if(act_taken == Action.WEST ):
                reward = -2.
                return reward
            if(act_taken == Action.NORTH):
                reward = -2.
                return reward
            if(act_taken == Action.STAY ):
                reward = -1.
                return reward
            if(act_taken == Action.ABANDON):
                reward = -10.
                return reward
        #3. In GOAL:
        if(self.grid[state[0]][state[1]] == Cell.GOAL):
            if(act_taken == Action.EAST ):
                pass
            if(act_taken == Action.SOUTH):
                pass
            if(act_taken == Action.WEST ):
                pass
            if(act_taken == Action.NORTH):
                pass
            if(act_taken == Action.STAY ):
                reward = 50.
                return reward
            if(act_taken == Action.ABANDON):
                pass
            
            return 0
        #4. In START:
        if(self.grid[state[0]][state[1]] == Cell.START):
            if(act_taken == Action.EAST ):
                reward = -1.
                return reward
            if(act_taken == Action.SOUTH):
                reward = -1.
                return reward
            if(act_taken == Action.WEST ):
                reward = -1.
                return reward                
            if(act_taken == Action.NORTH):
                reward = -1.
                return reward
            if(act_taken == Action.STAY ):
                reward = -1.
                return reward
            if(act_taken == Action.ABANDON):
                reward = -10.
                return reward


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        pass
