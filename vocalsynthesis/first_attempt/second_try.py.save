
"""
My first attempt at getting results on the vocal synthesis project. 
Input data is 3h of ooohhhhahhhhhmmmmmm meditation music
I do mean subtraction and input normalization, and then generate
training examples from the data using a sliding window. Then train
an LSTM to generate audio. Better description here:
 https://wordpress.com/post/teganmaharaj.wordpress.com/510
Apologies the gods of having more than one file.
"""

import sys
import random
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as tensor

from blocks.model import Model
from blocks.bricks import Linear, Tanh, Logistic #@list: changed from Sigmoid
from blocks.bricks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from fuel.transformers import Transformer
from blocks.algorithms import (GradientDescent, Scale, StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import LSTM
from blocks.graph import ComputationGraph


# =========================================
# SETTING UP DATA
# =========================================

# ==============
# things that should be argparsed maybe
# ==============
datafile = 'song.wav'
train_test_val_split = [80,10,10]
data_start = 2000
data_end = 20000 # set small for CPU testing; should be -150000 for reals
atom_length = 1000 # number of samples which are predicted at each step
atoms_in_example = 25 # number of atoms in a sequence
input_target_overlap = 0
example_shift = 1000
minibatch_size = 100
minibatch_shuffle = False
truncated_BPTT_length = 100 # is this the same as example length? Is using variable length sequences the same as variable length tbptt?
secs_to_generate = 60


# =============
# load data and cut it up
# =============
print '\nLoading data...\n'

data = wave.read(datafile)
sec = data[0]
minute = sec*60
raw = data[1]
track = raw[data_start:data_end]
num_frames = len(track)
length_in_seconds = num_frames/sec
length_in_minutes = length_in_seconds/60
length_in_hours = length_in_minutes/60

print datafile
print 'sample rate '+str(sec) + ' Hz, '+ str(num_frames) + ' samples long'
print str(length_in_seconds) +' seconds (' + str(length_in_minutes) + ' minutes)'

print '\n\nPreprocessing...\n'

test_start = int(num_frames*(train_test_val_split[0]/100.0))
val_start = int(num_frames*(train_test_val_split[1]/100.0)+ test_start)

train_data = np.asarray(raw[0:test_start], dtype="float32")
test_data = np.asarray(raw[test_start:val_start], dtype="float32")
val_data = np.asarray(raw[val_start: ], dtype="float32")

print 'cut up in ' + ':'.join([str(x) for x in train_test_val_split]) +' train:test:val split'


# =============
# calculate train/test/val statistics
# =============
train_mean = np.mean(train_data)
train_min = np.min(train_data)
train_max = np.max(train_data)

test_mean = np.mean(test_data)
test_min = np.min(test_data)
test_max = np.max(test_data)
if test_min > train_min:
    test_min = train_min
    print "test min greater than train min; using train min to normalize"
if test_max < train_max:
    test_max = train_max
    print "test max less than train max; using train max to normalize"

val_mean = np.mean(val_data)
val_min = np.min(val_data)
val_max = np.max(val_data)
val_range = val_max-val_min
if val_min > train_min:
    val_min = train_min
    print "val min greater than train min; using train min to normalize"
if val_max < train_max:
    val_max = train_max
    print "val max less than train max; using train max to normalize"


# =============
# rescale function that should be somewhere else probably
# =============
def rescale(unscaled_x, min_allowed, max_allowed, data_min, data_max):
# mostly stolen from http://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
    return (max_allowed-min_allowed)*(unscaled_x-data_min)/(data_max-data_min) + min_allowed


# =============
# normalize input data
# =============
processed_train_data = rescale(train_data-train_mean, -1,1, train_min, train_max)
processed_test_data = rescale(test_data-train_mean, -1,1, train_min, train_max)
processed_val_data = rescale(val_data-train_mean, -1,1, train_min, train_max)

train_test_val = [processed_train_data, processed_test_data, processed_val_data]
print "subtracted training mean and normalized data to [-1,1]"


# =============
# make examples by taking overlapping slices of data
# need S x T x B x F
# (num_batches x timesteps x batch_size x features)
# i.e. (num_batches x atoms_in_example x batch size x atom_length)
# I do num_batches
# =============
print '\n\nSlicing into examples...\n'
example_length = atoms_in_example*atom_length
print 'atom length: '+str(atom_length)+' samples'
print 'example length: '+str(atoms_in_example)+' atoms ('+example_length+' samples)'
print 'shift from example_i->example_i+1: ' +str(example_shift)+' samples'
print 'input-target overlap: '+str(input_target_overlap)+' samples'
print 'minibatch size: ' +str(minibatch_size)+' examples'
dataset_name = datafile.split('.')[0]+'_' + str(atom_length)+'_'+str(example_length)+'_'+str(example_shift)+'_'+str(input_target_overlap)+'_'+str(minibatch_size)
full_dataset = []
for what_dataset,dataset in enumerate(train_test_val):
    if what_dataset == 0:
        wha = 'train'
    elif what_dataset == 1:
        wha = 'test'
    elif what_dataset == 2:
        wha = 'val'
    else
        print 'well something went wrong'
        break;
    print 'slicing '+ wha
    inputs = []
    targets = []
    current_input = []
    current_target = []
    required_length = offset + target_length
    processed_so_far = required_length
    input_start = 0

    while target_end <= len(dataset):

        #input_start shifted by example_shift at the end of the loop
        input_end = input_start + example_length
        target_start = input_end - input_target_overlap
        target_end = target_start + example_length

        inputs.append( np.asarray(dataset[input_start : input_end] )
        targets.append( np.asarray(dataset[target_start : target_end] )


    full_dataset.append(examples)
    print 'examples generated'

    
    print '(note: '+ len(dataset) - current_length +' frames disregareded at end due to defined input target lengths)'
    input_start += example_shift
    
with open(dataset_name+'.pkl', "wb") as f:
    pickle.dump(full_dataset, f, pickle.HIGHEST_PROTOCOL )
    print 'pickle dumped to '+dataset_name+'.pkl'



# =========================================
# SET UP MODEL
# =========================================

# =============
# layers and outputs
# =============
x_to_h = Linear(name='x_to_h', input_dim=x_dim, output_dim=4 * h_dim)
x_transform = x_to_h.apply(x)
lstm = LSTM(activation=Tanh(), dim=h_dim, name="lstm")
h, c = lstm.apply(x_transform)
h_to_o = Linear(name='h_to_o', input_dim=h_dim, output_dim=x_dim)
y_hat = h_to_o.apply(h)
tanh = Tanh()
y_hat = tanh.apply(y_hat)
y_hat.name = 'y_hat'


# =============
# for generation
# =============
h_initial = tensor.tensor3('h_initial', dtype=floatX)
c_initial = tensor.tensor3('c_initial', dtype=floatX)
h_testing, c_testing = lstm.apply(x_transform, h_initial,c_initial, iterate=False)
y_hat_testing = h_to_o.apply(h_testing)
y_hat_testing = sigm.apply(y_hat_testing)
y_hat_testing.name = 'y_hat_testing'

cost = SquaredError().apply(y, y_hat)
cost.name = 'SquaredError'



# =========================================
# ACTUAL DO STUFF
# =========================================

# =============
# initialize
# =============
for brick in (lstm, x_to_h, h_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()


# =============
# build training
# =============
print 'Bulding training process...'
algorithm = GradientDescent(cost=cost,
                            params=ComputationGraph(cost).parameters,
                            step_rule=CompositeRule([StepClipping(10.0),
                                                     Scale(4)]))
monitor_cost = TrainingDataMonitoring([cost],
                                      prefix="train",
                                      after_epoch=True)


# =============
# reshape input and make into fuel datastream
# =============
# S x T x B x F
inputs = full_dataset[0][1]
outputs = full_dataset[0][
outputs[:, 0:-1, :, :] = inputs[:, 1:, :, :]
print 'Bulding DataStream ...'
dataset = IterableDataset({'x': inputs,
                           'y': outputs})
stream = DataStream(dataset)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[monitor_cost,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()],
                     model=model)


# =============
# mainloop for reals yo
# =============
print 'Starting training ...'
main_loop.run()

generate1 = theano.function([x], [y_hat, h, c])
generate2 = theano.function([x, h_initial, c_initial],
                            [y_hat_testing, h_testing, c_testing])
initial_seq = inputs[0, :20, 0:1, :]
current_output, current_hidden, current_cell = generate1(initial_seq)
current_output = current_output[-1:]
current_hidden = current_hidden[-1:]
current_cell = current_cell[-1:]
generated_seq = initial_seq[:, 0]
next_input = current_output
prev_state = current_hidden
prev_cell = current_cell
for i in range(200):
    current_output, current_hidden, current_cell = generate2(next_input,
                                                             prev_state,
                                                             prev_cell)
    next_input = current_output
    prev_state = current_hidden
    prev_cell = current_cell
    generated_seq = numpy.vstack((generated_seq, current_output[:, 0]))
print generated_seq.shape
save_as_gif(generated_seq.reshape(generated_seq.shape[0],
                                  numpy.sqrt(generated_seq.shape[1]),
                                  numpy.sqrt(generated_seq.shape[1])))





