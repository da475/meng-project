import os
import numpy as np
import pickle

class DATASET_MGR:
    def __init__(self, filename):
        self.data_file = open(filename, "rb")
        
        byte = self.data_file.read(4)    # first four bytes for magic number
        byte = self.data_file.read(4)    # four bytes for no of images

        # convert the data to a valid number
        data_size = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])

        # read image height
        byte = self.data_file.read(4)    # bytes for height pixels
        image_rows = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])
        
        # read image width
        byte = self.data_file.read(4)    # bytes for width pixels
        image_cols = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])
        
        # calcualte data dimension
        data_dim = image_rows * image_cols   # data dimension is total no of pixels in size
        self.data_array = np.zeros((data_size, data_dim))    # rows: no of images, cols: feature dimension



    # This is one-time needed function which will
    # - Load the dataset from the file
    # - Convert it into numpy 2-D array
    # - Dump it into a pickle file
    def load_train_images_data(self):
        byte = 20
        image_index = 0 # index for image in numpy array
        pixel_index = 0 # index for pixel in numpy array
        max_image_index, max_pixel_index = self.data_array.shape

        print (max_image_index)
        print (max_pixel_index)

        # start traversing the file
        for image_index in range(max_image_index):
            for pixel_index in range(max_pixel_index):
                byte = self.data_file.read(1)
                self.data_array[image_index, pixel_index] = (float)(byte[0])  # read pixel of 1 byte

        # dump the dataset into a pickle file
        f = open('dataset.pkl', 'wb')
        pickle.dump(self.data_array, f)
        f.close()

    def print_data(self):
        #print(self.data_array)
        f = open('dataset_labels.pkl', 'rb')
        data_array = pickle.load(f)
        f.close()
        print(data_array)

    
def set_numpy_array_for_train_labels():
        
    data_file = open('train_labels', "rb")
    byte = data_file.read(4)    # first four bytes magic number
    byte = data_file.read(4)    # four bytes for no of images

    # convert the data to a valid number
    no_of_images = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])

    data_array = np.zeros(no_of_images)    # rows: no of images, cols: feature dimension
 
    byte = 20
    image_index = 0 # index for image in numpy array
    max_image_index = data_array.shape

    print (max_image_index[0])

    # start traversing the file
    for image_index in range(max_image_index[0]):
        byte = data_file.read(1)
        data_array[image_index] = (float)(byte[0])  # read pixel of 1 byte

    # dump the dataset into a pickle file
    f = open('dataset_labels.pkl', 'wb')
    pickle.dump(data_array, f)
    f.close()



if __name__ == "__main__":
    #d = DATASET_MGR("train_labels")
    #d.load_and_dump_data()
    #d.print_data()

    set_numpy_array_for_train_labels()


