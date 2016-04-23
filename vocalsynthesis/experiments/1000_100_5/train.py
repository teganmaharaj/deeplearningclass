import theano
import numpy
from theano import tensor

from blocks.model import Model
from blocks.graph import ComputationGraph, apply_dropout
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing, saveload
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation

from blocks_extras.extensions.plot import Plot

from utils import get_stream, track_best, MainLoop, get_seed, make_wav
from model import nn_fprop
from config import config

# Load config parameters
locals().update(config)

# DATA
train_stream = get_stream(hdf5_file, 'train', batch_size)
dev_stream = get_stream(hdf5_file, 'dev', batch_size)


# MODEL
x = tensor.tensor3('inputs', dtype='float64')
y = tensor.tensor3('targets', dtype='float64')
import h5py
ff = h5py.File('song.hdf5', 'r')
x.tag.test_value = ff['inputs'][:3]
y.tag.test_value = ff['targets'][:3]
y_hat, cost, cells = nn_fprop(x, y, frame_length, hidden_size, num_layers, model)

# PREDICTIONS
predict_fn = theano.function([x], y_hat)
# test_data is a batch of examples
# test_predictions = predict_fn(test_data)
# train_predictions = predict_fn(train_data)
# make_wav(test_predictions)


# COST
cg = ComputationGraph(cost)

if dropout > 0:
    # Apply dropout only to the non-recurrent inputs (Zaremba et al. 2015)
    inputs = VariableFilter(theano_name_regex=r'.*apply_input.*')(cg.variables)
    cg = apply_dropout(cg, inputs, dropout)
    cost = cg.outputs[0]

# Learning algorithm
step_rules = [RMSProp(learning_rate=learning_rate, decay_rate=decay_rate),
              StepClipping(step_clipping)]
algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=CompositeRule(step_rules))

# Extensions
gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
step_norm = aggregation.mean(algorithm.total_step_norm)
monitored_vars = [cost, gradient_norm, step_norm]

dev_monitor = DataStreamMonitoring(variables=[cost], after_epoch=True,
                                   before_first_epoch=True, data_stream=dev_stream, prefix="dev")
train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True,
                                       before_first_epoch=True, prefix='train')

plotter = Plot('RNN char-level prediction',
	       channels=[[train_monitor.record_name(cost)],
                  [dev_monitor.record_name(cost)]],
	       server_url="http://bart4.iro.umontreal.ca:5006",
	       after_batch=True)

#to start fresh instead of starting from checkpoint, do saveload.Load(load_path)
#to start from checkpoint, do saveload.Load(last_path, load_log=True)
extensions = [dev_monitor, train_monitor, Timing(), Printing(after_batch=True),
              FinishAfter(after_n_epochs=num_epochs),
              saveload.Load(load_path), plotter,
              saveload.Checkpoint(last_path, save_separately=['log']),
              ] + track_best('dev_cost', save_path)

if learning_rate_decay not in (0, 1):
    extensions.append(SharedVariableModifier(step_rules[0].learning_rate,
                                             lambda n, lr: numpy.cast[theano.config.floatX](learning_rate_decay * lr), after_epoch=True, after_batch=False))

print 'number of parameters in the model: ' + str(tensor.sum([p.size for p in cg.parameters]).eval())
# Finally build the main loop and train the model
main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     model=Model(cost), extensions=extensions)
main_loop.run()

# Generate
print "generating audio..."
seed = get_seed(hdf5_file, [400])
generated_seq = []
prev_input = seed
for x in range(0, len_to_generate):
    prediction = predict_fn(prev_input)
    generated_seq.append(prediction.flatten()) #NEED TO OVERLAP/AVG
    #generated_seq.append((prev_input[1:]+prediction[:-1])/2)
    #generated_seq.append(prediction[-1])
    prev_input = prediction
make_wav(output_filename, numpy.asarray(generated_seq).flatten())