# Author: Jonathan Zwiebel, Robert Ross, Samuel Lurye
# Version: 18 May 2018

import numpy as np
import random
from scipy.io import loadmat
from random import shuffle

# Takes in a formatted MATLAB file and returns a list of numpy arrays with input and output information
def preprocess(filename, output_standardization_method, seq_length=None, seq_lookback_hard=None, seq_lookback_sample_range=None):
    file_in = loadmat(filename)
    raw_data = file_in['trials'][0]
    in_data = []
    out_data = []

    print("Done converting .mat into py")

    for m in range(len(raw_data)):
        if output_standardization_method == "position_relative":
            output_raw = raw_data[m][11]
        else:
            print("Bad output standardization method")
            return None

        out_data.append(output_raw)  # Already as np array
        in_data.append(raw_data[m][5].toarray())  # Converts a sparse array into a numpy array

    print("Done converting py into numpy arrays")

    in_data_proc = []
    out_data_proc = []

    # Seq length is the desired length of each sequence
    # Seq lookback distance is how far back the preprocessor can start its sampling
    if seq_length is not None:
        for m in range(len(out_data)):
            assert len(in_data) == len(out_data)
            start_lookback = seq_lookback_hard + random.randint(0, seq_lookback_sample_range)
            seq_start = raw_data[m][1][0][0] - start_lookback
            if not seq_start >= 0:
                print("Throwing for underflow subsample")
                continue
            print("m: " + str(m) + " | len(in_data): " + str(len(in_data)) + " | len(in_data[m]): " + str(len(in_data[m])))
            if not seq_start + seq_length - 1 < len(in_data[m][0]):
                print("Throwing for overflow subsample")
                continue

            if output_standardization_method == "position_relative":
                starts = out_data[m][:, seq_start].copy()
                for i in range(len(out_data[m][0])):
                    out_data[m][:, i] = out_data[m][:, i] - starts
            else:
                print("Bad outputput output standardization method")
                return None

            in_data_proc.append(in_data[m][:, seq_start:seq_start + seq_length])
            out_data_proc.append(out_data[m][:, seq_start:seq_start + seq_length])

            assert len(in_data_proc[-1][0]) == seq_length and len(out_data_proc[-1][0]) == seq_length

    print("Done down-sampling values")
    return in_data_proc, out_data_proc


# Splits the dataset (numpy array) into training, dev, and test datasets
def set_split(input_data, percentages):
    assert percentages["train"] + percentages["dev"] + percentages["test"] == 1

    train_set = []
    dev_set = []
    test_set = []

    train_set_size = int(percentages["train"] * len(input_data))
    dev_set_size = int(percentages["dev"] * len(input_data))
    test_set_size = len(input_data) - train_set_size - dev_set_size

    for _ in range(train_set_size):
        rand_int = np.random.randint(0, len(input_data))
        test_set.append(input_data[rand_int])
        del input_data[rand_int]

    for _ in range(dev_set_size):
        rand_int = np.random.randint(0, len(input_data))
        dev_set.append(input_data[rand_int])
        del input_data[rand_int]

    for _ in range(test_set_size):
        rand_int = np.random.randint(0, len(input_data))
        train_set.append(input_data[rand_int])
        del input_data[rand_int]

    assert len(input_data) == 0
    return train_set, dev_set, test_set

def two_value_shuffle(first, second):
    assert len(first) == len(second)
    indices = range(len(first))
    shuffle(indices)
    first_cp = first.copy
    second_cp = second.copy

    for i in range(len(first)):
        first_cp[i] = first[indices[i]]
        second_cp[i] = second[indices[i]]

    return first_cp, second_cp

class Dataset:
    def __init__(self, in_data, out_data):
        self.in_data = in_data
        self.out_data = out_data
        self.current_index = 0
        assert len(self.in_data) == len(self.out_data)

    def shuffle_dataset(self):
        np.random.shuffle(in_data)
        np.random.shuffle(out_data)

    def get_next_batch(self, batch_size):
        if (self.current_index + 1) * batch_size > len(self.in_data):
            self.reset_epoch()
            return None

    def reset_epoch(self):
        self.current_index = 0

def get_next_batch(train, test, batch_size, batch_index):
    return
