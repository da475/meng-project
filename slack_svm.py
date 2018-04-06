import numpy as np

LEARN_RATE = 0.05
CONVERGENCE_THRESHOLD = 0.01

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


    def compute_next_iteration(self):

        # compute the gradient and update the weight_vec according to gradient_descent

        z = np.matmul(self.data_matrix, self.weight_vec)
        z = (2 / (1 + np.exp(-(z - 0.5)))) - 2

        gradient = np.matmul(np.transpose(z), self.data_matrix)
        gradient = (2 * np.transpose(self.weight_vec)) + (self.slack_coeff * gradient)

        self.weight_vec = self.weight_vec - (self.learn_rate * np.transpose(gradient))
       

    def train(self):

        while (1):

            current_weight_vec = self.weight_vec
            self.compute_next_iteration()

            change = self.weight_vec - current_weight_vec
            movement = np.sqrt(np.matmul(np.transpose(change), change))

            if (movement < CONVERGENCE_THRESHOLD):
                break


    def test(self, test_input):

        X = np.ones((1, self.dimensions))
        X[:, :(self.dimensions - 1)] = test_input[:, :]

        if (np.matmul(X, self.weight_vec) >= 0):
            return +1
        else:
            return -1


if __name__ == "__main__":

    training_data = []
    training_labels = []

    svm = Slack_SVM(1, training_data, training_labels)

    svm.train()

    testing_data = []
    testing_labels = []

    n, d = testing_data.shape

    testing_error = 0

    for i in range(0, n):
        result = svm.test(testing_data[i, :])
        if (result != testing_labels[i]):
            testing_error += 1

    print("\n Testing Error : " + str(testing_error) + "/" + str(n) + "\n")
