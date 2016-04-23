from blocks import initialization
from blocks.bricks import Linear, NDimensionalSoftmax, Tanh
from blocks.bricks.cost import SquaredError
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.bricks.lookup import LookupTable
from theano import tensor

# Initialization (used by all layers)
def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()

def squared_error(y_hat, y):
    return tensor.sqr(y - y_hat)

# Different make-layer functions
# ------------------------------

# Softmax
def softmax_layer(h, y, frame_length, hidden_size):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size,
                              output_dim=frame_length)
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(h)
    linear_output.name = 'linear_output'
    softmax = NDimensionalSoftmax()
    y_hat = softmax.apply(linear_output, extra_ndim=1)
    y_hat.name = 'y_hat'
    cost = softmax.categorical_cross_entropy(
        y, linear_output, extra_ndim=1).mean()
    cost.name = 'cost'
    return y_hat, cost

# MSE loss
def MSEloss_layer(h, y, frame_length, hidden_size):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size,
                              output_dim=frame_length)
    initialize([hidden_to_output])
    y_hat = hidden_to_output.apply(h)
    y_hat.name = 'y_hat'
    cost = squared_error(y_hat, y).mean()
    cost.name = 'cost'
    #import ipdb; ipdb.set_trace()
    return y_hat, cost


# Vanilla RNN
def rnn_layer(dim, h, n):
    linear = Linear(input_dim=dim, output_dim=dim, name='linear' + str(n))
    rnn = SimpleRecurrent(dim=dim, activation=Tanh(), name='rnn' + str(n))
    initialize([linear, rnn])
    return rnn.apply(linear.apply(h))

# GRU
def gru_layer(dim, h, n):
    fork = Fork(output_names=['linear' + str(n), 'gates' + str(n)],
                name='fork' + str(n), input_dim=dim, output_dims=[dim, dim * 2])
    gru = GatedRecurrent(dim=dim, name='gru' + str(n))
    initialize([fork, gru])
    linear, gates = fork.apply(h)
    return gru.apply(linear, gates)

# LSTM
def lstm_layer(dim, h, n):
    linear = Linear(input_dim=dim, output_dim=dim * 4, name='linear' + str(n))
    lstm = LSTM(dim=dim, name='lstm' + str(n))
    initialize([linear, lstm])
    return lstm.apply(linear.apply(h))

# Linear input layer
def input_layer(dim, hidden_size, x):
    input_to_hidden = Linear(name='input_to_hidden', input_dim=dim,
                             output_dim=hidden_size)
    initialize([input_to_hidden])
    return input_to_hidden.apply(x)

#----------------------------------------------------

# Put the layers together
def nn_fprop(x, y, frame_length, hidden_size, num_layers, model):
    h = input_layer(frame_length, hidden_size, x)
    cells = []
    for i in range(num_layers):
        if model == 'rnn':
            h = rnn_layer(hidden_size, h, i)
        if model == 'gru':
            h = gru_layer(hidden_size, h, i)
        if model == 'lstm':
            h, c = lstm_layer(hidden_size, h, i)
            cells.append(c)
    return MSEloss_layer(h, y, frame_length, hidden_size) + (cells, )

#softmax_layer(h, y, frame_length, hidden_size) + (cells, ) #REMOVE SOFTMAX
