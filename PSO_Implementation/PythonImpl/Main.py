"""
Traditional Optimization Algorithm Main File
Pavel Berezovsky
Meng Project

 
INITIALIZATION
Initilize different values of func based on cost function to be tested
Func = 1 -> Rastrigin Function
Func = 2 -> Shifted Sphere Function
Func = 3 -> Griewank Function
Func = 4 -> Shifted Rosenbrock
Func = 5 -> Shifted Rotated Ackley

"""

import numpy as np
from Traditional_PSO import Traditional_PSO

"""
Sequence of parameters to be optimized:
    LearningRate, Slack_Coeff, ConvergenceThresh
"""

if __name__ == "__main__":
    Population = 20
    num_variables = 3
    Iterations = 100

    GlobalBest, GlobalPos = Traditional_PSO(Population, num_variables, Iterations)
        
    print("\n GlobalBest : " + str(GlobalBest) +  " GlobalPos : " + str(GlobalPos) + "\n")

