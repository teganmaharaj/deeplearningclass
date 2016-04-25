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
print '\nLoading data...'
data = wave.read(song_file)
sample_rate = data[0]
data = numpy.asarray(data[1])
num_samples = len(data)
example_length = frame_length*seq_length

# Normalize?
if normalize == True:
    print '\nNormalizing -1 to 1...'
    data_mean = numpy.mean(data)
    data_min = numpy.min(data)
    data_max = numpy.max(data)
    data = rescale(data-data_mean, -1, 1)
    print 'subtracted training mean and normalized data to [-1,1]'
elif normalize == 'stdev':
    data_stdev = numpy.std(data)
    data = data/data_stdev
    print 'divided by standard deviation'
else:
    print 'using unnormalized data'

# Make sure data will be the right length for reshaping
print '\nGetting examples...'
num_examples = 0
data_to_use = []
if example_shift == False:
    num_examples = len(data)-frame_length // example_length
    data_to_use = data[:num_examples*example_length + frame_length]
else:
    shift = 0
    while len(data[shift:]) >= example_length + frame_length:
        num_ex_this_pass = len(data[shift:]) // example_length
        data_to_use.extend( data[shift:shift+(num_ex_this_pass*example_length)] )
        num_examples += num_ex_this_pass
        shift += example_shift
    data_to_use.extend( data[shift:shift+frame_length] )

# Reshape
print '\nReshaping...'
data_to_use = numpy.asarray(data_to_use)
input_array = data_to_use[:-frame_length].reshape(num_examples,seq_length,frame_length) 
target_array = data_to_use[frame_length:].reshape(num_examples,seq_length,frame_length)
print input_array.shape
print target_array.shape

# Make H5PY file
print '\nMaking Fuel-formatted HDF5 file...'
f = h5py.File(hdf5_file, mode='w')
inputs = f.create_dataset('inputs', input_array.shape, dtype='float64')
targets = f.create_dataset('targets', target_array.shape, dtype='float64')
inputs[...] = input_array
targets[...] = target_array
inputs.dims[0].label = 'batch'
inputs.dims[1].label = 'sequence'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'sequence'

# Split in test:train
print 'doing train:test split (at '+str(train_samples)+')'
num_train_examples = train_samples // example_length
split_dict = {
    'train': {'inputs': (0, num_train_examples), 'targets': (0, num_train_examples)},
    'dev': {'inputs': (num_train_examples, num_examples), 'targets': (num_train_examples, num_examples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
print "file should be made"