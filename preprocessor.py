# Author: Jonathan Zwiebel, Robert Ross, Samuel Lurye
# Version: 18 May 2018

import sys
import numpy as np 
from scipy.io import loadmat

# Takes in a formatted MATLAB file and returns numpy matrices with input and output information
def preprocess(filename):
	file_in = loadmat(filename)
	raw_data = file_in['R'][0]
	in_data = []
	out_data = []

	for i in range(len(raw_data)):
		out_data.append(raw_data[i][9]) # Already as np array
		in_data.append(np.concatenate((raw_data[i][10].toarray(), raw_data[i][11].toarray())))

	print(len(in_data))
	print(len(out_data))
	print(in_data[1].shape)
	print(out_data[1].shape)
	return in_data, out_data


def set_split(input_data):
	""" """
	train_set = []
	dev_set = []
	test_set = []
	test_set_percentage = .1
	dev_set_percentage = .1
	
	test_set_size = int(test_set_percentage * len(input_data))
	dev_set_size = int(dev_set_percentage * len(input_data))

	for _ in range(test_set_size):
		rand_int = np.random.randint(0, len(input_data))
		test_set.append(input_data[rand_int])
		del input_data[rand_int]
	
	for _ in range(dev_set_size):
		rand_int = np.random.randint(0, len(input_data))
		dev_set.append(input_data[rand_int])
		del input_data[rand_int]
	
	for _ in range(len(input_data)):
		rand_int = np.random.randint(0, len(input_data))
		train_set.append(input_data[rand_int])
		del input_data[rand_int]
	return train_set, dev_set, test_set