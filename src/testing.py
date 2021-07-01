import random
import numpy as np
import torch

# multivariate data preparation
from numpy import array
from numpy import hstack
import pandas as pd
src = torch.rand(32, 10, 512)

def split_sequences(sequences, startpoint, n_steps):
    X, y = list(), list()

    # find the end of this pattern
    end_ix = startpoint + n_steps
    # check if we are beyond the dataset
    if end_ix > len(sequences):
        print('critical error occured in "split_sequences"')
        return
    # gather input and output parts of the pattern
    X, y = sequences[startpoint:end_ix, :-1], sequences[end_ix - 1, -1]

    return array(X), array(y)


dataset = pd.read_csv('newFile.csv', header=0, nrows=200, index_col=0).to_numpy()
# define input sequence

for i in range(10):
    A, b = split_sequences(dataset, i, 10)

print('test')
#
# in_seq2 = array([x for x in range(5, 105, 10)])
# out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
# # convert to [rows, columns] structure
# in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# out_seq = out_seq.reshape((len(out_seq), 1))
# # horizontally stack columns
# dataset = hstack((in_seq1, in_seq2, out_seq))
