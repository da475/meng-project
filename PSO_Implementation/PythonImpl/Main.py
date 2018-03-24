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

if __name__ == "__main__":
    Population = 50
    num_variables = 2
    Iterations = 2000
    Data = np.zeros((5, 2))
    Mean = np.zeros(5)
    Std = np.zeros(5)

    for i in range(1, 6):
        Func = i;
        print("Function" + str(i))

        for k in range(0, 2):
            Data[i, k] = Traditional_PSO(Func, Population, num_variables, Iterations)
        
        Mean[i] = np.mean(Data[i, :])
        Std[i] = np.std(Data[i, :])

