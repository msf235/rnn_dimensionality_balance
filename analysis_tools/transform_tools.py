from numpy import np

# %% Helper functions
def undo_onehot(x):
    # returns the array X where instead of the last dimension being a onehot representation,
    # the class index is returned

    sequence = np.where(x != 0)[-1]
    x_seq = np.reshape(sequence, x.shape[:-1])
    return x_seq


def input_transform(X, mode='temporal'):
    if mode == 'static':
        return X[:, 0]
    if mode == 'temporal':
        return X
