config = {}
config['song_file'] = 'song.wav'  # raw input file
config['normalize'] = True
config['normalized_min'] = -1
config['normalized_max'] = 1
config['frame_length'] = 4000 # number of samples per step (i.e. per 'char')
config['seq_length'] = 50  # number of chars in the sequence
config['example_shift'] = 1000 # number of frames to shift by to start next sequence 
config['train_samples'] = 160000000 # (from leaderboard)
config['batch_size'] = 128  # number of examples taken per each update
config['num_epochs'] = 30  # number of full passes through the training data

config['model'] = 'gru'  # 'rnn', 'gru' or 'lstm'
config['hidden_size'] = 199  # number of hidden units per layer
config['num_layers'] = 2
config['learning_rate'] = 0.002
config['learning_rate_decay'] = 0.97 # set to 0 to not decay learning rate
config['decay_rate'] = 0.95  # decay rate for rmsprop
config['step_clipping'] = 1.0  # clip norm of gradients at this value
config['dropout'] = 0

config['seed_filename'] = 'song.wav'
config['output_filename'] = 'generated_audio.wav'
config['len_to_generate'] = 5

config['hdf5_file'] = 'song.hdf5'  # hdf5 file with Fuel format
config['save_path'] = '{0}_best.pkl'.format(config['model'])  # path to best model file
config['load_path'] = '{0}_saved.pkl'.format(config['model'])  # start from a saved model file
config['last_path'] = '{0}_last.pkl'.format(config['model'])  # path to save the model of the last iteration
