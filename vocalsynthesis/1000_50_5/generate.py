def generate(X_test, model, l_out, out_fn, length, data_avg, data_range):
    """ Given a trained model, generates output audio using the start
    of the test set as a seed.
    """
    set_all_param_values(l_out, model)
    seed = X_test[0:1]
    generated_seq = []
    prev_input = seed
    for x in range(0, length):
        next_input = out_fn(prev_input)
        generated_seq.append(next_input.flatten()[0:8000])
        prev_input = next_input
    generated_seq = np.array(generated_seq).flatten() * data_range
    generated_seq = generated_seq.astype('int16')
    return generated_seq







def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes