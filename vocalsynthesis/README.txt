Current workflow:

------
1. Start bokeh server for plots
   (remember to set server address in config)
------
bokeh-server --ip 0.0.0.0

------
2. Make a directory in /experiments and copy necessary files
3. Set parameters and hyperparameters in config.py
4. Run the following:
------

make_dataset.py
train.py
generate.py

