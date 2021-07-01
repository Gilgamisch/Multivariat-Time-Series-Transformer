from tsai.all import *

import torch


bs = 32
c_in = 9  # aka channels, features, variables, dimensions
c_out = 2
seq_len = 60

xb = torch.randn(bs, c_in, seq_len)

# standardize by channel by_var based on the training set
xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)

# Settings
max_seq_len = 120
d_model = 128
n_heads = 16
d_k = d_v = None # if None --> d_model // n_heads
d_ff = 256
res_dropout = 0.1
act = "gelu"
n_layers = 3
fc_dropout = 0.1
kwargs = {}
# kwargs = dict(kernel_size=5, padding=2)

model = TST(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
            d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout, act=act, n_layers=n_layers,
            fc_dropout=fc_dropout, **kwargs)
test_eq(model(xb).shape, [bs, c_out])
print(f'model parameters: {count_parameters(model)}')