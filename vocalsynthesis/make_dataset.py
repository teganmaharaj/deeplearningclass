import numpy
import codecs
import h5py
import yaml
import scipy.io.wavfile as wave
from fuel.datasets import H5PYDataset
from config import config
from utils import rescale

# Load config parameters
locals().update(config)
numpy.random.seed(0)

# Load data
print '\nLoading data...\n'
data = wave.read(song_file)
sample_rate = data[0]
data = numpy.asarray(data[1])
num_samples = len(data)
example_length = frame_length*seq_length

# Normalize?
if normalize == True:
    print '\nNormalizing...\n'
    data_mean = numpy.mean(data)
    data_min = numpy.min(data)
    data_max = numpy.max(data)
    data = rescale(data-data_mean, -1, 1, data_min, data_max)
    print "subtracted training mean and normalized data to [-1,1]"
else:
    print "using unnormalized data"

# Make sure data will be the right length for reshaping
if len(data) % example_length < frame_length:
    data = data[:-example_length+frame_length]
else:
    data = data[: -(len(data)%example_length-frame_length)]
if not len(data)%example_length == frame_length:
    print "problem! you suck at math."

# Reshape
num_examples = len(data) // example_length
chars = data.reshape(len(data)/frame_length, frame_length) #list(set(data))
vocab_size = len(chars)
input_array = data[:-frame_length].reshape(num_examples,seq_length,frame_length) 
target_array = data[frame_length:].reshape(num_examples,seq_length,frame_length)
print input_array.shape
print target_array.shape
#assert False
# Nake H5PY file
f = h5py.File(hdf5_file, mode='w')
print "making file..."
inputs = f.create_dataset('inputs', input_array.shape, dtype='float64')
targets = f.create_dataset('targets', target_array.shape, dtype='float64')
#targets.attrs['inputs'] = yaml.dump(char_to_ix)
#targets.attrs['targets'] = yaml.dump(ix_to_char)
inputs[...] = input_array
targets[...] = target_array
inputs.dims[0].label = 'batch'
inputs.dims[1].label = 'sequence'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'sequence'

num_train_examples = train_samples // example_length

split_dict = {
    'train': {'inputs': (0, num_train_examples), 'targets': (0, num_train_examples)},
    'dev': {'inputs': (num_train_examples, num_examples), 'targets': (num_train_examples, num_examples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
print "file should be made"