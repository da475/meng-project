import numpy as np

LEARN_RATE = 0.05
CONVERGENCE_THRESHOLD = 0.5

# Expects as input
# training_data : matrix (n x d), set of real numbers
# training_labels : vector (n x 1), {+1, -1}
# n : number of samples
# d : dimensionality of the data

class Slack_SVM:
    
    def __init__(self, slack_coeff, training_data, training_labels):

        # convert the data points in the input training_data
        # to homogeneous coordinates
        n, d = training_data.shape

        self.num_samples = n
        self.dimensions = d + 1
        
        self.data_matrix = np.ones((self.num_samples, self.dimensions))
        self.data_matrix[:, :(self.dimensions - 1)] = training_data[:, :]
        self.data_matrix = np.multiply(self.data_matrix, np.matmul(training_labels, np.ones((1, self.dimensions))))

        self.slack_coeff = slack_coeff
        self.learn_rate = LEARN_RATE
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
        while (1):

            num_iterations += 1

            current_weight_vec = self.weight_vec
            self.compute_next_iteration()

            change = self.weight_vec - current_weight_vec
            movement = np.sqrt(np.matmul(np.transpose(change), change))

            print("\n Itr: " + str(num_iterations) + " movement : " + str(movement[0, 0]))

            if (movement < CONVERGENCE_THRESHOLD):
                break

        print("\n Iterations before Convergence : " + str(num_iterations) + "\n")


    def test(self, test_input):

        X = np.ones((1, self.dimensions))
        X[:, :(self.dimensions - 1)] = np.transpose(test_input)

        if (np.matmul(X, self.weight_vec) >= 0):
            return +1
        else:
            return -1

