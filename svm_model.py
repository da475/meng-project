
import numpy as np

from slack_svm import Slack_SVM

class SVM_Model:

    def __init__(self, datafiles_path):

        # If the location isn't provided, it is assumed to be
        # the current working directory
        if datafiles_path == "":
            datafiles_path = "."

        datafile_fullpath = datafiles_path + "/processed_data_image.npy"
        labelfile_fullpath = datafiles_path + "/processed_data_label.npy"

        # load the image data and labels and pre-process them
        raw_image_data = np.load(datafile_fullpath)
        raw_labels = np.load(labelfile_fullpath)

        self.num_data_points = raw_image_data.shape[0]

        # The raw image data would be an array of 3D images
        # This needs some pre-processing and vectorization
        self.data = self.pre_process_image_data(raw_image_data)

        # similarly for the labels
        self.labels = self.pre_process_labels(raw_labels)


    def pre_process_image_data(self, raw_image_data):

        # TODO_JK:
        # Select the middle slice and vectorize the image
        # 


    def pre_process_labels(self, raw_labels):

        # TODO_JK:
        # SVM needs labels form the set {+1, -1}


    def evaluate(self, slack_coeff):

        # split the samples into training and testing sets

        permutation = np.arange(self.num_data_points)
        np.random.shuffle(permutation)

        num_data_points, data_dimensions = self.data.shape

        training_data_size = num_data_points * 4 / 5
        training_data = np.zeros((training_data_size, data_dimensions))
        training_labels = np.zeros((training_data_size, 1))
        testing_data = np.zeros((num_data_points - training_data_size, data_dimensions))
        testing_labels = np.zeros((num_data_points - training_data_size, 1))

        j = 0
        for i in range(0, training_data_size):

            training_data[j, :] = self.data[permutation[i], :]
            training_labels[j] = self.labels[permutation[i]]

            j += 1

        j = 0
        for i in range(training_data_size, num_data_points):

            testing_data[j, :] = self.data[permutation[i], :]
            testing_labels[j] = self.labels[permutation[i]]

            j += 1

        # Create and Train the model

        svm = Slack_SVM(slack_coeff, training_data, training_labels)

        svm.train()

        testing_error = 0

        for i in range(0, testing_data.shape[0]):

            result = svm.test(testing_data[i, :])
            
            if (result != testing_labels[i]):
                testing_error += 1

        print("\n Testing Error : " + str(testing_error) + "/" + str(j) + "\n")

        return testing_error

