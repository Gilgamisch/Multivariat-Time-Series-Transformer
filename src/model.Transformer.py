import time
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot


nrows=500           # number of rowas taken from data set
sampels = 350
input_window = 100  # number of input steps
n_features = 14
output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size = 20
eval_batch_size = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



class TransformerModel(nn.Module):
    def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """
        super().__init__()
        # self.permute = Permute(2, 0, 1)
        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout,
                                                activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
        # self.transpose = Transpose(1, 0)
        # self.max = Max(1)
        self.outlinear = nn.Linear(d_model, c_out)

    def forward(self, x):
        # x = self.permute(x)  # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 0)  # seq_len x bs x d_model -> bs x seq_len x d_model
        x = x.max(1, keepdim=False)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        return x


def get_data():
    dataset = pd.read_csv('newFile.csv', header=0, nrows=nrows, index_col=0).to_numpy()
    train_data = dataset[:sampels]
    test_data = dataset[sampels:]

    return train_data, test_data


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


def train(train_data):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()
    X, y = split_sequences(train_data, input_window)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))
    for b in range(0, len(X)):   #b: batch
        inpt = X[b:b + batch_size, :, :]
        target = y[b:b + batch_size]

        x_batch = torch.tensor(inpt, dtype=torch.float32).permute(1,0,2).to(device)
        y_batch = torch.tensor(target, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(X) / batch_size / 5)
        if b % log_interval == 0 and b > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, b, len(X) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    X, y = split_sequences(data_source, input_window)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))
    prior = None
    with torch.no_grad():
        for b in range(0, len(X) - 1):
            inpt = X[b:b + batch_size, :, :]
            target = y[b:b + batch_size]
            x_batch = torch.tensor(inpt, dtype=torch.float32).permute(1, 0, 2).to(device)
            y_batch = torch.tensor(target, dtype=torch.float32).to(device)
            if prior != None:
                if list(x_batch.size()) != prior:
                    print(x_batch.size())
            output = eval_model(x_batch)
            prior = list(x_batch.size())
            total_loss += criterion(output, y_batch).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, y_batch[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    len(test_result)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.plot(test_result - truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-epoch%d.png' % epoch)
    pyplot.close()

    return total_loss / b


# predict the next n steps based on the input data
def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    X, y = split_sequences(data_source, input_window)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))
    inpt = X[:batch_size, :, :]
    target = y[:batch_size]
    x_batch = torch.tensor(inpt, dtype=torch.float32).permute(1, 0, 2).to(device)
    y_batch = torch.tensor(target, dtype=torch.float32).to(device)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(x_batch[-input_window:])
            future = torch.cat((y_batch[:-1].reshape(-1), output[-1]))

    future = future.cpu().view(-1)

    # I used this plot to visualize if the model pics up any long therm struccture within the data.
    pyplot.plot(future, color="red")
    pyplot.plot(future[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-future%d.png' % steps)
    pyplot.close()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    X, y = split_sequences(data_source, input_window)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))
    with torch.no_grad():
        for b in range(0, len(X) - 1, eval_batch_size):
            inpt = X[b:b + batch_size, :, :]
            target = y[b:b + batch_size]
            x_batch = torch.tensor(inpt, dtype=torch.float32).permute(1, 0, 2).to(device)
            y_batch = torch.tensor(target, dtype=torch.float32).to(device)
            output = eval_model(x_batch)
            total_loss += len(x_batch[0]) * criterion(output, y_batch).cpu().item()
    return total_loss / len(data_source)



if __name__ == "__main__":
    lr = 0.0005
    epochs = 100  # The number of epochs

    train_data, val_data = get_data()

    model = TransformerModel(14, 1)
    criterion = nn.MSELoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        if epoch % 10 == 0:
            val_loss = plot_and_loss(model, val_data, epoch)
            predict_future(model, val_data, 200)
        else:
            val_loss = evaluate(model, val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time),
                                                                                                      val_loss,
                                                                                                      math.exp(
                                                                                                          val_loss)))
        print('-' * 89)

        # if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    best_model = model

        scheduler.step()

    # src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
    # out = model(src)
    #
    # print(out)
    # print(out.shape)

