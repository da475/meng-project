
import numpy as np
import os
import sys
import csv

from slack_svm import Slack_SVM

class SVM_Model:

    def __init__(self, datafiles_path):

        # If the location isn't provided, it is assumed to be
        # the current working directory
        if datafiles_path == "":
            datafiles_path = "."

        datafile_fullpath = datafiles_path + "/processed_for_svm_data.npy"
        labelfile_fullpath = datafiles_path + "/processed_for_svm_labels.npy"

        # load the image data and labels and pre-process them
        raw_image_data = np.load(datafile_fullpath)
        raw_labels = np.load(labelfile_fullpath)

        # The raw image data would be an array of 3D images
        # This needs some pre-processing and vectorization
        #self.data = self.pre_process_image_data(raw_image_data)
        self.data = raw_image_data

        # similarly for the labels
        self.labels = self.pre_process_labels(raw_labels)


    def pre_process_image_data(self, raw_image_data):

	num_data_points, image_depth, image_width, image_height = raw_image_data.shape

        # load the spiralling order from the csv file

        os.system('gcc -o spiralling ./spiralling.c')
        os.system('./spiralling ' + str(image_width) + ' > ./spiralling.csv')

        contents = []
        f = open('./spiralling.csv')
        h = csv.reader(f)

        for r in h:
            contents.append(int(r[0]))

        f.close()

        # Select the middle slice and vectorize the image

        processed_data = np.zeros((num_data_points, image_width * image_height))

        remapping = np.array(contents)

        for i in range(num_data_points):

            raw_image = raw_image_data[i, (image_depth / 2)]
            raw_image = np.clip(raw_image, 0, 255)
            raw_image = raw_image / 255

            flattened = raw_image.flatten()
            processed_data[i] = flattened[remapping]

        return processed_data


    def pre_process_labels(self, raw_labels):

        # SVM needs labels form the set {+1, -1}
        # Incoming Labels are in the set {0, 1}
        return (raw_labels * 2) - 1


    def evaluate(self, learn_rate, slack_coeff, convg_thresh):

        # split the samples into training and testing sets

        num_data_points, data_dimensions = self.data.shape

        permutation = np.arange(num_data_points)
        #np.random.shuffle(permutation)

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

        svm = Slack_SVM(learn_rate, slack_coeff, convg_thresh, training_data, training_labels)

        svm.train()

        testing_error = 0

        for i in range(0, testing_data.shape[0]):

            result = svm.test(testing_data[i, :])
            
            if (result != testing_labels[i]):
                testing_error += 1

        #print("\n Testing Error : " + str(testing_error) + "/" + str(j) + "\n")

        return (float(testing_error) / float(j))


if __name__ == "__main__":

    p1 = float(sys.argv[1])
    p2 = float(sys.argv[2])
    p3 = float(sys.argv[3])

    svm_model = SVM_Model('.')

    err = svm_model.evaluate(p1, p2, p3)

    print err

