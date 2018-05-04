"""
Traditional Path Swarm Optimization
Pavel Berezovsky
Meng Project

"""

import numpy as np
import random
from CNN_Model import main_cnn

class Particle():
    def __init__(self, pos, number_of_variables):
        self.position = pos
        self.best_position = pos
        self.best_cost = 0
        self.velocity = np.zeros(number_of_variables)

# PSO
def Traditional_PSO(Population, number_of_variables, Iterations):

    # Upper and lower bounds of the function chosen
    # 0 - number of features in conv layer
    # 1 - number of features in fc layer
    # 2 - learning rate for adam optimizer

    Lower_Bound = np.array([8, 128, 0.0005])
    Upper_Bound = np.array([16, 256, 0.005])

    # Setting Upper and Lower Bounds for position and velocity vector
    Bounds_p = np.array([Lower_Bound, Upper_Bound])

    # TODO check this
    Bounds_v = np.array([-(Bounds_p[1]-Bounds_p[0]), Bounds_p[1]-Bounds_p[0]])
    Global_best_cost = float('inf')        # The initial best cost
    Global_best_position = float('inf')    # Initial best position
 
    Particles = []

    for i in range(0, Population):

        # TODO : we need separate random seeds for each of the components of the Position
        random_vec = np.array([random.random() for ii in range(number_of_variables)])

        # float to int
        #Particles[i].position[0] = int(Particles[i].position[0])
        #Particles[i].position[1] = int(Particles[i].position[1])

        # Setting initial position of each particle within the bounds of Func
        Particles.append(Particle(Bounds_p[0] + np.multiply((Bounds_p[1] - Bounds_p[0]), random_vec), number_of_variables))
        # Best position of each particle is its current initial position

        # Obtaining best cost of each particle
        Particles[i].best_cost = main_cnn(Particles[i].position)
        #Particles[i].best_cost = svm_model.evaluate(Particles[i].position[0], Particles[i].position[1], Particles[i].position[2])

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
                if Particles[i].velocity[j] > Bounds_v[1, j]:
                   Particles[i].velocity[j] = Bounds_v[1, j]
                elif Particles[i].velocity[j] < Bounds_v[0, j]:
                   Particles[i].velocity[j] = Bounds_v[0, j]
 
            # Equation for particle position
            Particles[i].position = Particles[i].position + Particles[i].velocity

            # float to int
            #Particles[i].position[0] = int(Particles[i].position[0])
            #Particles[i].position[1] = int(Particles[i].position[1])
            
            # If particle position exceeds bounds it is set to bounds
            for j in range(0, number_of_variables):
                if Particles[i].position[j] > Bounds_p[1, j]:
                   Particles[i].position[j] = Bounds_p[1, j]
                elif Particles[i].position[j] < Bounds_p[0, j]:
                   Particles[i].position[j] = Bounds_p[0, j]

            # Finding temporary cost of the particle
            #Temp_cost = svm_model.evaluate(Particles[i].position[0], Particles[i].position[1], Particles[i].position[2])
            Temp_cost = main_cnn(Particles[i].position)
            
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
       
        # Debug Info
        print("\n PSO's Iteration Count : " + str(count) + "\n")

        # Incrementing iterations 
        count = count + 1
        
    return Global_best_cost, Global_best_position, Total
 
