from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc
#--------------
from pdm4ar.exercises.ex04.structures import Cell
from typing import Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function



def StartPosOfMap(grid: NDArray) -> Tuple[int,int]:
    for i in range(np.shape(grid)[0]):
        for j in range(np.shape(grid)[1]):
            if(grid[i][j] == 1):
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
    
    

def IsNeighbourOf(A: State, B:State) -> bool:
    if(abs(A[0]-B[0])==1 and abs(A[1]-B[1])==1):
        return False
    if(abs(A[0]-B[0])>1 or abs(A[1]-B[1])>1):
        return False
    return True
        
        


if __name__ == '__main__':
    grid: NDArray[np.int64] = [[1, 3, 1],
                            [2,  5, 1]]
    startState = (0,0)
    state: State = (1 , -1)
    state2: State = (1,2)
    print("state[0]>=np.shape(grid)[0]-1:{0}".format((state[0]>=np.shape(grid)[0]-1)))
    print("state[1]>=np.shape(grid)[1]-1:{0}".format((state[1]>=np.shape(grid)[1]-1)))
    print("state[0] <0:{0}".format((state[0] <0)))
    print("state[1] <0:{0}".format((state[1] <0)))
    
    
   

    if(state[0]>=np.shape(grid)[0] or state[1]>=np.shape(grid)[1] or state[0] <0 or state[1] <0):
        print("out!!!")
    
  



