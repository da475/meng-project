import os
import numpy as np

class DATASET_MGR:
    def __init__(self, filename):
        self.data_file = open(filename, "rb")
        
        byte = self.data_file.read(4)    # first four bytes magic number
        byte = self.data_file.read(4)    # bytes for no of images

        # convert the data to a valid number
        data_size = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])

        # read image height
        byte = self.data_file.read(4)    # bytes for no of images
        image_rows = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])
        
        # read image width
        byte = self.data_file.read(4)    # bytes for no of images
        image_cols = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])
        
        # calcualte data dimension
        data_dim = image_rows * image_cols   # data dimension is total no of pixels in size
        self.data_array = np.zeros((data_size, data_dim))    # rows: no of images, cols: feature dimension

    def load_data(self):
        byte = 20
        image_index = 0 # index for image in numpy array
        pixel_index = 0 # index for pixel in numpy array
        max_image_index, max_pixel_index = self.data_array.shape()

        # start traversing the file
        for image_index in xrange(max_image_index):
            for pixel_index in xrange(max_pixel_index):
                self.data_array[image_index][pixel_index] = self.data_file.read(1)  # read pixel of 1 byte
                

    def print_data(self):
        print(self.data_array)

"""
with open("myfile", "rb") as f:
    byte = f.read(4)
    byte = f.read(4)
    number = int(byte[0]) << 24 | int(byte[1]) << 16 | int(byte[2]) << 8 | int(byte[3])
    print(number)
    #while byte != "":
    #    # Do stuff with byte.
    #    byte = f.read(1)
"""


if __name__ == "__main__":
    d = DATASET_MGR("myfile")
    d.load_data()
    d.print_data()
    
