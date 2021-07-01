import random
import numpy as np
import torch

# multivariate data preparation
from numpy import array
from numpy import hstack
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# # define input sequence
# in_seq1 = array([x for x in range(0, 100, 10)])
# in_seq2 = array([x for x in range(5, 105, 10)])
# out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
# # convert to [rows, columns] structure
# in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# out_seq = out_seq.reshape((len(out_seq), 1))
# # horizontally stack columns
# dataset = hstack((in_seq1, in_seq2, out_seq))
dataset = pd.read_csv('newFile.csv', header=0, nrows=200, index_col=0).to_numpy()

class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20  # number of hidden states
        self.n_layers = 1  # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)


n_features = 14 # this is number of parallel inputs
n_timesteps = 100 # this is number of timesteps

# convert dataset into input/output
X, y = split_sequences(dataset, n_timesteps)
scaler = MinMaxScaler(feature_range=(-1, 1))
y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

print(X.shape, y.shape)

# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=0.00001)

train_episodes = 5000
batch_size = 20

mv_net.train()
loss_values = []
for t in range(train_episodes):
    running_loss = 0
    for b in range(0, len(X), batch_size):
        inpt = X[b:b + batch_size, :, :]
        target = y[b:b + batch_size]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        y_batch = torch.tensor(target, dtype=torch.float32)

        mv_net.init_hidden(x_batch.size(0))
        #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
        #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(x_batch)
        loss = criterion(output.view(-1), y_batch)
        running_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    loss_values.append(running_loss / batch_size)
    if(t%200 == 0):
        plt.plot(loss_values)
        plt.show()
    print('step : ', t, 'loss : ', loss.item())




