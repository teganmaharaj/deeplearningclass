import theano
import numpy
import pickle
from theano import tensor

from blocks.model import Model
from blocks.graph import ComputationGraph, apply_dropout
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing, saveload
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from blocks.serialization import load

from blocks_extras.extensions.plot import Plot

from utils import get_stream, track_best, MainLoop, get_seed, make_wav, rescale
from model import nn_fprop
from config import config
# Load config parameters
locals().update(config)

# Set up model and prediction function
x = tensor.tensor3('inputs', dtype='float64')
y = tensor.tensor3('targets', dtype='float64')

model = 'bs'
with open ('gru_best.pkl', 'r') as picklefile:
    model = load(picklefile)
y_hat, cost, cells = nn_fprop(x, y, frame_length, hidden_size, num_layers, model)
predict_fn = theano.function([x], y_hat)

# Generate
print "generating audio..."
seed = get_seed(hdf5_file, [seed_index])
sec = 16000
samples_to_generate = sec*secs_to_generate
num_frames_to_generate = samples_to_generate/frame_length + seq_length #don't include seed
predictions = []
prev_input = seed
for i in range(num_frames_to_generate):
    prediction = predict_fn(prev_input)
    predictions.append(prediction)
    pred_min = numpy.min(predictions)
    pred_max = numpy.max(predictions)
    prev_input = rescale(prediction, pred_max, pred_min) 
actually_generated = numpy.asarray(predictions)[seq_length:,:,:,:] #cut off seed
last_frames = actually_generated[:,:,-1,:]
make_wav(output_filename, actually_generated.flatten())
print str(secs_to_generate)+'seconds of audio generated'
print output_filename