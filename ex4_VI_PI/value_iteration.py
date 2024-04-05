from typing import Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function
#-------------------
import pdm4ar.exercises.ex04.mdp as mdp

class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        cur_value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)
        # todo implement here
        
        num_iterMax = 100
        epsilon = 0
        count = 0
        while(count < num_iterMax):
            count += 1
            cur_value_func = np.copy(value_func)
            
            for i in range(np.shape(value_func)[0]):
                for j in range(np.shape(value_func)[1]):
                    #-------------DEBUG INFO PRINTING CONTROL--------------
                    if(False): debugInfoFlag = 1
                    else: debugInfoFlag = 0
                    #------------------------------------------------------
                    
                    for act_choice in range(6):#6 different action choices
                        new_value = 0
                        if(debugInfoFlag):print("[DEBUG]--------ITER:{0} ; act_choice:{1}-------".format(count,act_choice))
                        for act_taken in range(6):#6 different action taken
                            #-------------DEBUG INFO PRINTING CONTROL--------------
                            if(act_choice==1 and act_taken ==5 and debugInfoFlag): grid_mdp.FlagPrintInfo = 1
                            else: grid_mdp.FlagPrintInfo = 0
                            #------------------------------------------------------                           
                            cur_state = (i, j)
                            
                            #in cell near the start point the agent will not act aboandon, we skip this turn to avid misclassify aboandon action as other action, which lead to wrong transition prob
                            if(mdp.IsNeighbourOf(cur_state,grid_mdp.startState) and act_taken == 5):
                                continue
                            
                            next_state = mdp.CalculateNextState(cur_state,act_taken,grid_mdp.startState)
                            tran_prob = grid_mdp.get_transition_prob(cur_state, act_choice, next_state)
                            reward = grid_mdp.stage_reward(cur_state, act_choice, next_state)
                            
                            #Deal with case when next state out of the map
                            if(mdp.OutOfMap(next_state,grid_mdp.grid)):
                                next_state = grid_mdp.startState
                            
                            # if(debugInfoFlag):print("[DEBUG]-----act_taken:{0}----".format(act_taken))
                            # if(debugInfoFlag):print("[DEBUG]cur_state:{0}".format(cur_state))
                            # if(debugInfoFlag):print("[DEBUG]next_state:{0}".format(next_state))                                    
                            # if(debugInfoFlag):print("[DEBUG]tran_prob:{0}".format(tran_prob))
                            # if(debugInfoFlag):print("[DEBUG]reward:{0}".format(reward))                           
                            # if(debugInfoFlag):print("[DEBUG]value_func[next_state[0]][next_state[1]:{0}".format(value_func[next_state[0]][next_state[1]]))
                                               
                            new_value += tran_prob * (reward + grid_mdp.gamma * value_func[next_state[0]][next_state[1]])
                        #update
                        if(new_value > value_func[i][j]):
                            
                            value_func[i][j] = new_value
                            policy[i][j] = act_choice
                            
                            # if(debugInfoFlag):print("[DEBUG][UPDATE!]value_func[i][j]:{0}".format(value_func[i][j])) 
                            # if(debugInfoFlag):print("[DEBUG][UPDATE!]policy[i][j]:{0}".format(policy[i][j]))  
                            
                                                   
            #Early termination
            difference = abs(np.sum(cur_value_func - value_func))
            print("[DEBUG]count:{0}".format(count))
            print("[DEBUG]difference:{0}".format(difference))
            if(difference < epsilon):
                break

        return value_func, policy
