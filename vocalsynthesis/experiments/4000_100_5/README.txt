Current workflow:
run make_dataset.py
run train.py

the hyperparams are pretty much all defined in the config.py
the model is in model.py

it should probably be a separate file, but for now generation of audio is just done at the end of train.py
