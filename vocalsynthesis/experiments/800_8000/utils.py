import sys
import h5py
import yaml
import numpy
from fuel.datasets import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Mapping
from blocks.extensions import saveload, predicates
from blocks.extensions.training import TrackTheBest
from blocks import main_loop
from fuel.utils import do_not_pickle_attributes
import scipy.io.wavfile as wave

#Define this class to skip serialization of extensions
@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)

    def load(self):
        self.extensions = []

def rescale(unscaled_data, min_allowed, max_allowed):
    data_min = numpy.min(unscaled_data)
    data_max = numpy.max(unscaled_data)
    return (max_allowed-min_allowed)*(unscaled_data-data_min)/(data_max-data_min) + min_allowed

def transpose_stream(data):
    return (data[0].T, data[1].T)


def track_best(channel, save_path):
    tracker = TrackTheBest(channel, choose_best=min)
    checkpoint = saveload.Checkpoint(
        save_path, after_training=False, use_cpickle=True)
    checkpoint.add_condition(["after_epoch"],
                             predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
    return [tracker, checkpoint]


def get_stream(hdf5_file, which_set, batch_size=None):
    dataset = H5PYDataset(
        hdf5_file, which_sets=(which_set,), load_in_memory=True)
    if batch_size == None:
        batch_size = dataset.num_examples
    stream = DataStream(dataset=dataset, iteration_scheme=ShuffledScheme(
        examples=dataset.num_examples, batch_size=batch_size))
    # Required because Recurrent bricks receive as input [sequence, batch,
    # features]
    return stream #Mapping(stream, transpose_stream)


def get_seed(file_name, seed_index):
    infile = h5py.File(file_name, 'r')
    input_array = infile['inputs'][:]
    return input_array[seed_index]

def make_wav(output_filename, generated_seq, sample_rate=16000,data_min=-3000, data_max=3000): # data_min=-8936, data_max=9124, norm_min=-1.0, norm_max=1.0):
    generated_seq = rescale(generated_seq, data_min, data_max)
    generated_seq = generated_seq.astype('int16')
    wave.write(output_filename, sample_rate, generated_seq)
