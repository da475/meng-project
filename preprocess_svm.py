		
def pre_process_image_data(raw_image_data):
	num_data_points, image_depth, image_width, image_height = raw_image_data.shape

	# load the spiralling order from the csv file

	#os.system('gcc -o spiralling ./spiralling.c')
	#os.system('./spiralling ' + str(image_width) + ' > ./spiralling.csv')

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
		#raw_image = np.clip(raw_image, 0, 255)
		#raw_image = raw_image / 255

		flattened = raw_image.flatten()
		processed_data[i] = flattened[remapping]

	return processed_data

if __name__ == "__main__":

	datafile_fullpath = "./processed_data_image.npy"

	# load the image data and labels and pre-process them
	raw_image_data = np.load(datafile_fullpath)

	# The raw image data would be an array of 3D images
	# This needs some pre-processing and vectorization
	final_data = pre_process_image_data(raw_image_data)

	np.save("./processed_for_svm.npy", final_data)

