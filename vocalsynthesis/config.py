config = {}

# mostly for make_dataset.py
config['song_file'] = '../../song.wav'  # raw input file
config['normalize'] = 'stdev'
config['normalized_min'] = -1
config['normalized_max'] = 1
config['frame_length'] = 40000 # number of samples per step (i.e. per 'char')
config['seq_length'] = 40  # number of chars in the sequence
config['example_shift'] = False #config['frame_length'] # number of samples to shift by for overlapping examples
config['train_samples'] = 160000000 # train-test split point (from leaderboard)
config['file_string'] = '{0}_{1}_{2}'.format(config['frame_length'], config['seq_length'], config['example_shift'])
config['hdf5_file'] = config['file_string']+'.hdf5'  # hdf5 file to save to (Fuel format)

# mostly for model.py
config['model'] = 'gru'  # 'rnn', 'gru' or 'lstm'
config['hidden_size'] = 256  # number of hidden units per layer
config['num_layers'] = 2

# mostly for train.py
config['batch_size'] = 128  # number of examples taken per each update
config['num_epochs'] = 120  # number of full passes through the training data
config['learning_rate'] = 0.002
config['learning_rate_decay'] = 0.97 # set to 0 to not decay learning rate
config['decay_rate'] = 0.95  # decay rate for rmsprop
config['step_clipping'] = 1.0  # clip norm of gradients at this value
config['dropout'] = 0
config['plot_name'] = config['file_string'] +'_'+ config['model']
config['save_path'] = config['plot_name'] + '_best.pkl' # path to best model file
config['load_path'] = config['plot_name'] + '_saved.pkl' # start from a saved model file
config['last_path'] = config['plot_name'] + '_last.pkl' # path to save the model of the last iteration

# mostly for generate.py
config['seed_filename'] = config['hdf5_file']
config['output_filename'] = config['plot_name'] + '.wav'
config['secs_to_generate'] = 30