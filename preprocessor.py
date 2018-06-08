# Author: Jonathan Zwiebel, Robert Ross, Samuel Lurye
# Version: 18 May 2018

import numpy as np
import random
from scipy.io import loadmat
from random import shuffle
from time import sleep


def preprocess(files, output_standardization_method, seq_length=None, skip_length = None):
    in_data_full = []
    out_data_full = []
    for filename in files:
        file_in = loadmat(filename)
        raw_data = file_in['trials'][0]
        in_data = []
        out_data = []
        print("Done converting .mat into py")

        for m in range(len(raw_data)):
            if output_standardization_method == "position_relative":
                output_raw = raw_data[m][11]
            elif output_standardization_method == "velocity":
                output_raw = raw_data[m][7]
            elif output_standardization_method == "velocity_bin":
                output_raw = raw_data[m][9]
            else:
                print("Bad output standardization method")
                return None

            out_data.append(output_raw)  # Already as np array
            if output_standardization_method in ["position_relative", "velocity"]:
                in_data.append(raw_data[m][5].toarray())  # Converts a sparse array into a numpy array
            else:
                in_data.append(raw_data[m][6].toarray())

        print("Done converting py into numpy arrays")

        in_data_proc = []
        out_data_proc = []

        for m in range(len(out_data)):
            assert len(in_data) == len(out_data)
            assert (seq_length + skip_length) < len(out_data)
            if output_standardization_method == "position_relative":
                starts = out_data[m][:, 0].copy()
                for i in range(len(out_data[m][0])):
                    out_data[m][:, i] = out_data[m][:, i] - starts
            elif output_standardization_method == "velocity" or output_standardization_method == "velocity_bin":
                pass
            else:
                print("Bad output standarization method")
                return None

            in_data_proc.append(in_data[m][:, skip_length:seq_length+skip_length])
            out_data_proc.append(out_data[m][:, skip_length:seq_length+skip_length])
            in_data_proc[-1] = in_data_proc[-1].T
            out_data_proc[-1] = out_data_proc[-1].T

            assert len(in_data_proc[-1]) == seq_length and len(out_data_proc[-1]) == seq_length
        if len(in_data_full) == 0:
            in_data_full = in_data_proc
            out_data_full = out_data_proc
        else:
            np.concatenate((in_data_full,in_data_proc), axis = 0)
            np.concatenate((out_data_full,out_data_proc), axis = 0)
        #in_data_full.append(in_data_proc)
        #out_data_full.append(out_data_proc)

    print("Done down-sampling values")
    print(in_data_full[0].shape)
    print(out_data_full[0].shape)
    sleep(3)
    return np.asarray(in_data_full), np.asarray(out_data_full)


def two_value_shuffle(first, second):
    assert len(first) == len(second)
    indices = list(range(len(first)))
    shuffle(indices)
    first_cp = np.copy(first)
    second_cp = np.copy(second)

    for i in range(len(first)):
        first_cp[i] = first[indices[i]]
        second_cp[i] = second[indices[i]]

    return first_cp, second_cp

def set_split(X, Y, percentages):
    """
    Splits the X and Y into the train, test, and dev set.

    Arguments:
    X -- numpy array of shape (batch_size, sequence_length, layer_size)
    Y -- numpy array of shape (batch_size, sequence_length, output_size)
    percentages -- dictionary of the percentages of the train, dev, and test set

    Return:
    sets -- dictionary of tuples of the train, dev, and test set for X and Y
    """
    assert percentages["train"] + percentages["dev"] + percentages["test"] == 1
    assert len(X) == len(Y)

    sets = {}

    size_input = len(X)
    test_set_size = int(size_input * percentages["test"])
    dev_set_size = int(size_input * percentages["dev"])

    X, Y = two_value_shuffle(X, Y)

    test_set_X = X[0:test_set_size]
    dev_set_X = X[test_set_size:test_set_size + dev_set_size]
    train_set_X = X[test_set_size + dev_set_size:]

    test_set_Y = Y[0:test_set_size]
    dev_set_Y = Y[test_set_size:test_set_size + dev_set_size]
    train_set_Y = Y[test_set_size + dev_set_size:]

    sets["train"] = (train_set_X, train_set_Y)
    sets["dev"] = (dev_set_X, dev_set_Y)
    sets["test"] = (test_set_X, test_set_Y)

    return sets

class Dataset:
    def __init__(self, in_data, out_data):
        self.in_data = in_data
        self.out_data = out_data
        self.current_index = 0
        assert len(self.in_data) == len(self.out_data)

    def shuffle_dataset(self):
        self.in_data, self.out_data = two_value_shuffle(self.in_data, self.out_data)

    def get_next_batch(self, batch_size):
        batch_x = self.in_data[:batch_size]
        batch_y = self.out_data[:batch_size]
        return batch_x, batch_y

        # FORCE

        if (self.current_index + 1) * batch_size > len(self.in_data):
            self.reset_epoch()
            return None, None

        batch_x = self.in_data[batch_size * self.current_index: batch_size * (self.current_index + 1)]
        batch_y = self.out_data[batch_size * self.current_index: batch_size * (self.current_index + 1)]
        self.current_index = self.current_index + 1
        return batch_x, batch_y

    def reset_epoch(self):
        self.current_index = 0
        self.shuffle_dataset()


# Takes in a formatted MATLAB file and returns a list of numpy arrays with input and output information
def preprocess_old(filename, output_standardization_method, seq_length=None, seq_lookback_hard=None, seq_lookback_sample_range=None):
    file_in = loadmat(filename)
    raw_data = file_in['trials'][0]
    in_data = []
    out_data = []

    print("Done converting .mat into py")

    for m in range(len(raw_data)):
        if output_standardization_method == "position_relative":
            output_raw = raw_data[m][11]
        elif output_standardization_method == "velocity":
            output_raw = raw_data[m][9]
        else:
            print("Bad output standardization method")
            return None

        out_data.append(output_raw)  # Already as np array
        in_data.append(raw_data[m][6].toarray())  # Converts a sparse array into a numpy array
        # in_data.append(raw_data[m][5].toarray())

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
                continue
            if not seq_start + seq_length - 1 < len(in_data[m][0]):
                continue

            if output_standardization_method == "position_relative":
                starts = out_data[m][:, seq_start].copy()
                for i in range(len(out_data[m][0])):
                    out_data[m][:, i] = out_data[m][:, i] - starts
            elif output_standardization_method == "velocity":
                print("ok")
            else:
                print("Bad outputput output standardization method")
                return None

            in_data_proc.append(in_data[m][:, seq_start:seq_start + seq_length])
            out_data_proc.append(out_data[m][:, seq_start:seq_start + seq_length])
            in_data_proc[-1] = in_data_proc[-1].T
            out_data_proc[-1] = out_data_proc[-1].T

            assert len(in_data_proc[-1]) == seq_length and len(out_data_proc[-1]) == seq_length

    print("Done down-sampling values")
    return np.asarray(in_data_proc), np.asarray(out_data_proc)
