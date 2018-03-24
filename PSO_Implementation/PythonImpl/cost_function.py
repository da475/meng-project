"""
All possible cost functions

"""

import numpy as np

def cost_function(particle, num_variables, function_choice):
    if function_choice == 1:
        # Rastrigin Function
        cos_array = np.zeros(particle.size)
        for j in range(0, particle.size):
            cos_array[j] = np.cos(2 * np.pi * particle[j])
        cost = 10 * num_variables + np.sum(particle**2 - 10 * cos_array)
        lower_bound = -5.12
        upper_bound = 5.12
    elif function_choice == 2:
        # Shifted Sphere Function
        cost = np.sum(particle**2) - 450
        lower_bound = -100
        upper_bound = 100
    elif function_choice == 3:
        # Griewank Function
        Product = np.cos(particle[0] / np.sqrt(1))
        for i in range(1, num_variables):
            Product = Product * np.cos(particle[i] / np.sqrt(i + 1))
        cost = 1+(1/4000) * np.sum(particle**2) - Product
        lower_bound = -600
        upper_bound = 600
    elif function_choice == 4:
        # Shifted Rosenbrock
        cost = 0
        for i in range(0, num_variables-1):
            cost = cost + 100 * (particle[i+1] - particle[i]**2)**2 + (1 - particle[i])**2
        cost = cost + 390
        lower_bound = -100
        upper_bound = 100
    elif function_choice == 5:
        # Shifted Rotated Ackley
        cos_array = np.zeros(particle.size)
        for j in range(0, particle.size):
            cos_array[j] = np.cos(2 * np.pi * particle[j])
        cost = -20 * np.exp(-.02 * np.sqrt(np.sum((particle**2)/num_variables))) \
               - np.exp(np.sum(cos_array)/num_variables) \
               + 20 + np.exp(1) + (-140)
        lower_bound = -32
        upper_bound = 32
    else:
        print('Cost Unavailable')

    return lower_bound, upper_bound, cost

