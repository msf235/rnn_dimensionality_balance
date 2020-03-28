import pickle as pkl
import os
import numpy as np
from joblib import Memory
import copy

class Memoize:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            classification_instance = args[0]
        except IndexError:
            classification_instance = kwargs['classification_instance']

        out_dir = classification_instance.params['output_dir']
        save_dir = out_dir + '/' + self.f.__name__ + '.pkl'
        duplicate = os.path.isfile(save_dir)
        if not duplicate:
            retvals = self.f(*args, **kwargs)
            with open(save_dir, 'wb') as fid:
                pkl.dump(retvals, fid, protocol=4)
        else:
            with open(out_dir + '/' + self.f.__name__ + '.pkl', 'rb') as fid:
                # spectra = pkl.load(fid)['spectra']
                retvals = pkl.load(fid)
        # Warning: You may wish to do a deepcopy here if returning objects
        print()
        return retvals

class Memoize2:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        class_arg = True
        try:
            classification_instance = args[0]
        except IndexError:
            classification_instance = kwargs['classification_instance']
            class_arg = False
        try:
            params_table = classification_instance.params_table.__dict__
        except AttributeError:
            params_table = classification_instance.params_table
        kwargs_new = {}
        for key in kwargs:
            if key != 'classification_instance':
                kwargs_new[key] = kwargs[key]
        if class_arg:
            args_new = args[1:]
            kwargs_new.update(params_table)
        else:
            args_new = args
            del kwargs_new['classification_instance']
            kwargs_new.update(params_table)

        out_dir = classification_instance.params['output_dir']
        memory = Memory(cachedir=out_dir, verbose=1)

        # memory.clear()

        # @memory.cache
        def f_restrict(*arg_restrict, **kwargs_restrict):
            return self.f(*args, **kwargs)

        f_restrict.__name__ = self.f.__name__
        f_restrict = memory.cache(f_restrict)

        return f_restrict(*args_new, **kwargs_new)

class Memoize_numpy:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            classification_instance = args[0]
        except IndexError:
            classification_instance = kwargs['classification_instance']

        out_dir = classification_instance.params['output_dir']
        save_dir = out_dir + '/' + self.f.__name__ + '.npz'
        duplicate = os.path.isfile(save_dir)
        single_output = False
        if not duplicate:
            retvals = self.f(*args, **kwargs)
            if not isinstance(retvals, tuple):
                with open(save_dir, 'wb') as fid:
                    np.savez(fid, retvals)
            else:
                with open(save_dir, 'wb') as fid:
                    np.savez(fid, *retvals)
        else:
            retvals = []
            with open(out_dir + '/' + self.f.__name__ + '.npz', 'rb') as fid:
                # spectra = pkl.load(fid)['spectra']
                # retvals = pkl.load(fid)
                retval_load = np.load(fid)

                for i1 in range(len(retval_load.files)):
                    key = 'arr_' + str(i1)
                    retvals.append(retval_load[key])
            retvals = tuple(retvals)
        # Warning: You may wish to do a deepcopy here if returning objects
        if isinstance(retvals, tuple) and len(retvals) == 1:
            return retvals[0]
        else:
            return retvals

def format_dir(dir_string):
    return dir_string.parent
    # dstr = str(dir_string)
    # temp = dir_string.split('/')
    # if temp[-1] == '':
    #     dir_string = '/'.join(temp)
    # else:
    #     dir_string = '/'.join(temp) + '/'
    # return dir_string
