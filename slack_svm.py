
# Author : Jaydev Kshirsagar

# The program in this file is an implementation of a Soft Support-Vector-Machine
# It exposes the functionality as a Python Class that can get the SVM trained and also
# make it usable for testing.

import numpy as np

LEARN_RATE = 0.05
CONVERGENCE_THRESHOLD = 0.5
MAX_ITER = 2000

# Expects as input
# training_data : matrix (n x d), set of real numbers
# training_labels : vector (n x 1), {+1, -1}
# n : number of samples
# d : dimensionality of the data

class Slack_SVM:
    
    def __init__(self, learn_rate, slack_coeff, convg_thresh, training_data, training_labels):

        # Debug
        #print("\n Creating SVM Model with the params : " + str(learn_rate) + ", " + str(slack_coeff) + ", " + str(convg_thresh) + "\n")

        # convert the data points in the input training_data
        # to homogeneous coordinates
        n, d = training_data.shape

        self.num_samples = n
        self.dimensions = d + 1
        
        self.data_matrix = np.ones((self.num_samples, self.dimensions))
        self.data_matrix[:, :(self.dimensions - 1)] = training_data[:, :]
        self.data_matrix = np.multiply(self.data_matrix, np.matmul(training_labels, np.ones((1, self.dimensions))))

        self.slack_coeff = slack_coeff
        self.learn_rate = learn_rate
        self.convg_thresh = convg_thresh
        self.weight_vec = np.random.rand(self.dimensions, 1)

        # randomly assign signs to weight_vec components, just to increase randomness
        random_signs = np.array([[pow(-1, j) for j in range(0, self.dimensions)]])
        self.weight_vec = np.multiply(self.weight_vec, np.transpose(random_signs))


    def compute_next_iteration(self):

        # compute the gradient and update the weight_vec according to gradient_descent

        z = np.matmul(self.data_matrix, self.weight_vec)
        z = (2 / (1 + np.exp(-(z - 0.5)))) - 2

        gradient = np.matmul(np.transpose(z), self.data_matrix)
        gradient = (2 * np.transpose(self.weight_vec)) + (self.slack_coeff * gradient)

        self.weight_vec = self.weight_vec - (self.learn_rate * np.transpose(gradient))
       

    def train(self):

        num_iterations = 0
        while (num_iterations < MAX_ITER):

            num_iterations += 1

            current_weight_vec = self.weight_vec
            self.compute_next_iteration()

            change = self.weight_vec - current_weight_vec
            movement = np.sqrt(np.matmul(np.transpose(change), change))

            #print("\n Itr: " + str(num_iterations) + " movement : " + str(movement[0, 0]))

            if (movement < self.convg_thresh):
                break

        #print("\n Iterations before Convergence : " + str(num_iterations) + "\n")


    def test(self, test_input):

        X = np.ones((1, self.dimensions))
        X[:, :(self.dimensions - 1)] = np.transpose(test_input)

        if (np.matmul(X, self.weight_vec) >= 0):
            return +1
        else:
            return -1

