"""
Traditional Path Swarm Optimization
Pavel Berezovsky
Meng Project

"""

import numpy as np
import random
from cost_function import cost_function

class Particle():
    def __init__(self, pos, number_of_variables):
        self.position = pos
        self.best_position = pos
        self.best_cost = 0
        self.velocity = np.zeros(number_of_variables)

# PSO
def Traditional_PSO(Func, Population, number_of_variables, Iterations):

    # Upper and lower bounds of the function chosen
    Lower_Bound, Upper_Bound, cost = cost_function(np.zeros(number_of_variables), number_of_variables, Func)
 
    # Setting Upper and Lower Bounds for position and velocity vector
    Bounds_p = np.array([Lower_Bound, Upper_Bound])
    Bounds_v = np.array([-(Bounds_p[1]-Bounds_p[0]), Bounds_p[1]-Bounds_p[0]])
    Global_best_cost = float('inf')        # The initial best cost
    Global_best_position = float('inf')    # Initial best position
 
    Particles = []
    random_vec = np.array([random.random() for i in range(number_of_variables)])

    for i in range(0, Population):
        # Setting initial position of each particle within the bounds of Func
        Particles.append(Particle(Bounds_p[0] + 2 *Bounds_p[1] * random_vec, number_of_variables))
        # Best position of each particle is its current initial position

        # Obtaining best cost of each particle
        discard1, discard2, Particles[i].best_cost = cost_function(Particles[i].position, number_of_variables, Func)

        # If best cost of each particle is less than the global cost set
        # global cost
        if Particles[i].best_cost < Global_best_cost:
            Global_best_cost = Particles[i].best_cost
            Global_best_position = Particles[i].best_position
 
    # PSO LOOP
    # iteration count
    count = 0
    # Inertial coefficient (memory)
    w = .8
    # Coefficients 
    c1 = 2
    c2 = 2
 
    Total = [0 for i in range(0, Iterations)]

    while (count < Iterations):
        for i in range(0, Population):
            # Random elements added to generate velocity
            r1 = random.random()
            r2 = random.random()

            # Equation of particle velocity
            Particles[i].velocity = w*Particles[i].velocity \
                + c1 * r1 * (Particles[i].best_position - Particles[i].position) \
                + c2 * r2 * (Global_best_position - Particles[i].position)

            # If particle velocity exceeds bounds it is set to bounds
            for j in range(0, number_of_variables):
                if Particles[i].velocity[j] > Bounds_v[1]:
                   Particles[i].velocity[j] = Bounds_v[1]
                elif Particles[i].velocity[j] < Bounds_v[0]:
                   Particles[i].velocity[j] = Bounds_v[0]
 
            # Equation for particle position
            Particles[i].position = Particles[i].position + Particles[i].velocity
            
            # If particle position exceeds bounds it is set to bounds
            for j in range(0, number_of_variables):
                if Particles[i].position[j] > Bounds_p[1]:
                   Particles[i].position[j] = Bounds_p[1]
                elif Particles[i].position[j] < Bounds_p[0]:
                   Particles[i].position[j] = Bounds_p[0]

            # Finding temporary cost of the particle
            discard1, discard2, Temp_cost = cost_function(Particles[i].position, number_of_variables, Func)
            
            # If temporary cost is less than particle best cost
            if Temp_cost < Particles[i].best_cost:
                Particles[i].best_cost = Temp_cost
                Particles[i].best_position = Particles[i].position

            # If global best cost is greater than particle best cost 
            if Particles[i].best_cost < Global_best_cost:
                Global_best_cost = Particles[i].best_cost
                Global_best_position = Particles[i].best_position
        
        # Tracking global cost
        Total[count] = Global_best_cost
        
        # Incrementing iterations 
        count = count + 1
        
    return Global_best_cost
 
