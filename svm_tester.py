
# Author : Jaydev Kshirsagar

# The program in this file is a tester of the Sof Support-Vector-Machine Model
# It creates a synthetic Data-Set for testing the SVM and checks if the Model works well

import numpy as np

from slack_svm import Slack_SVM

if __name__ == "__main__":

    SAMPLE_CNT = 300
    DIM_CNT = 2
    sample_data = np.zeros((SAMPLE_CNT, DIM_CNT))
    sample_labels = np.zeros((SAMPLE_CNT, 1))

    # TODO_JK : Generalize the computations to multiple dimensions

    # Generate synthetic linearly separable dataset for testing.
    # Data Points in each class are Gaussian distributed

    data_point_count = 0

    while (data_point_count < SAMPLE_CNT):
    
        x1 = (np.random.rand() - 0.5) / 0.4
        x2 = (np.random.rand() - 0.5) / 0.3

        if (((x1 * x1) + (x2 * x2)) <= 1):

            data_point = np.array([10 * x1, 10 * x2])

            sample_data[data_point_count, :] = data_point + 20
            sample_labels[data_point_count] = 1

            data_point_count += 1

            sample_data[data_point_count, :] = data_point - 20
            sample_labels[data_point_count] = -1

            data_point_count += 1

    # split the samples into training and testing sets

    permutation = np.arange(SAMPLE_CNT)
    np.random.shuffle(permutation)

    training_data_size = SAMPLE_CNT * 4 / 5
    training_data = np.zeros((training_data_size, DIM_CNT))
    training_labels = np.zeros((training_data_size, 1))
    testing_data = np.zeros((SAMPLE_CNT - training_data_size, DIM_CNT))
    testing_labels = np.zeros((SAMPLE_CNT - training_data_size, 1))

    j = 0
    for i in range(0, training_data_size):

        training_data[j, :] = sample_data[permutation[i], :]
        training_labels[j] = sample_labels[permutation[i]]

        j += 1

    j = 0
    for i in range(training_data_size, SAMPLE_CNT):

        testing_data[j, :] = sample_data[permutation[i], :]
        testing_labels[j] = sample_labels[permutation[i]]

        j += 1

    # Create and Train the model

    svm = Slack_SVM(1, training_data, training_labels)

    svm.train()

    testing_error = 0

    for i in range(0, testing_data.shape[0]):

        result = svm.test(testing_data[i, :])
        
        if (result != testing_labels[i]):
            testing_error += 1

    print("\n Testing Error : " + str(testing_error) + "/" + str(j) + "\n")

