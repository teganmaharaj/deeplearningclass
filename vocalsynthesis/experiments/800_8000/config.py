config = {}

# mostly for make_dataset.py
config['song_file'] = '../song.wav'  # raw input file
config['normalize'] = True
config['normalized_min'] = -1
config['normalized_max'] = 1
config['frame_length'] = 800 # number of samples per step (i.e. per 'char')
config['seq_length'] = 8000  # number of chars in the sequence
config['example_shift'] = config['frame_length'] # number of samples to shift by for overlapping examples
config['train_samples'] = 160000000 # train-test split point (from leaderboard)
config['hdf5_file'] = '800_8000.hdf5'  # hdf5 file to save to (Fuel format)

# mostly for model.py
config['model'] = 'gru'  # 'rnn', 'gru' or 'lstm'
config['hidden_size'] = 199  # number of hidden units per layer
config['num_layers'] = 2

# mostly for train.py
config['batch_size'] = 128  # number of examples taken per each update
config['num_epochs'] = 30  # number of full passes through the training data
config['learning_rate'] = 0.002
config['learning_rate_decay'] = 0.97 # set to 0 to not decay learning rate
config['decay_rate'] = 0.95  # decay rate for rmsprop
config['step_clipping'] = 1.0  # clip norm of gradients at this value
config['dropout'] = 0
config['plot_name'] = 'gru_800_8000'
config['save_path'] = '{0}_best.pkl'.format(config['model'])  # path to best model file
config['load_path'] = '{0}_saved.pkl'.format(config['model'])  # start from a saved model file
config['last_path'] = '{0}_last.pkl'.format(config['model'])  # path to save the model of the last iteration
config['pickled_model'] = 'gru_best.pkl'
config['train_and_generate'] = True

# mostly for generate.py
config['seed_filename'] = config['hdf5_file']
config['output_filename'] = '800_8000.wav'
config['secs_to_generate'] = 30