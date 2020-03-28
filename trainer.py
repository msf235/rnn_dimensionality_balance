import copy
import inspect
from typing import *
import math
import numpy as np
import torch
from torch.utils.data import Dataset
# import model_output_manager as mom
from network_analysis import model_output_manager as mom
import network_analysis as na
from network_analysis import model_trainer
from network_analysis import models
import classification_task

TABLE_PATH = 'output/output_table.csv'


def get_max_eigval(A):
    ew, __ = torch.eig(A)
    mags = torch.sqrt(ew[:, 0] ** 2 + ew[:, 1] ** 2)
    idx_sort = torch.flip(torch.argsort(mags), dims=[0])
    ew = ew[idx_sort]
    return ew


# def gen_input(X_clusters, samples_per_epoch, num_train, loss_points, batch_size, n_lag, n_hold, n_out, X_dim, Y_classes,
#               clust_sig, input_style, model_seed, freeze_input, nonlinearity, squeeze=False):
#     if input_style == 'hypercube' and nonlinearity == 'tanh':
#         print("Input style is: hypercube")
#         cluster_method = 6
#     elif input_style == 'hypercube' and nonlinearity == 'relu':
#         print("Input style is: hypercube")
#         cluster_method = 7
#     else:
#         cluster_method = 3
#
#     X, Y, Z, sequence, centers, cluster_class_label = classification_task.classification(samples_per_epoch, n_lag, X_dim, Y_classes,
#                                                                                          X_clusters, n_hold, n_out,
#                                                                                          noise_sigma=clust_sig,
#                                                                                          return_verbose=True,
#                                                                                          cluster_method=cluster_method,
#                                                                                          avg_magn=1, cluster_seed=2 * model_seed,
#                                                                                          assignment_and_noise_seed=3 * model_seed)
#
#
#
#     # visualize_input(class_datasets['train'])
#     # visualize_input(class_datasets['val'])
#
#     dataloaders = {
#         x: torch.utils.data.DataLoader(class_datasets[x], batch_size=batch_size, shuffle=False, num_workers=0)
#         for x in ['train', 'val']}
#
#     return dataloaders, sequence, centers, cluster_class_label, where_noise, class_datasets


def nested_tuple_to_str(inp_tuple):
    temp = ''
    for x in inp_tuple:
        for y in x:
            temp = temp + str(y) + '&'
        temp = temp[:-1]
        temp = temp + '_'
    inp_tuple = temp[:-1]
    return inp_tuple


def train(N: int, X_clusters: int, samples_per_epoch: int, n_lag: int, n_hold: int, n_out: int, X_dim: int,
          Y_classes: int, clust_sig: float = 0.1, model_seed: int = 2, hid_nonlin: str = 'tanh', stop_epoch: int = 20,
          learning_rate: float = 0.005, patience_before_stopping: int = 10, batch_size: int = 10, loss: str = 'cce',
          optimizer: str = 'rmsprop', perc_val: float = 0.2, scheduler: str = 'plateau', learning_patience: int = 5,
          scheduler_factor: float = 0.5, Win: str = 'orthog', Wrec_rand_proportion: float = 1,
          net_architecture: str = 'vanilla_rnn', input_scale: float = 1., g_radius: float = 20., dt: float = 0.01,
          param_regularization: Optional[str] = None, param_regularization_weight: Optional[float] = None,
          activity_regularization: Optional[str] = None, activity_regularization_weight: Optional[float] = None,
          freeze_input: bool = False, input_style: str = 'hypercube', epochs_per_save: int = 1, rerun: bool = False):
    """
    Parameters
    ----------
    N : int
        Number of units in the "hidden" layer, i.e. number of neurons making up the recurrent layer.
    X_clusters : int
        Number of clusters.
    samples_per_epoch : int
        Number of input points to draw per epoch.
    n_lag : int
        Number of timesteps from stimulus onset to end of loss evaluation.
    n_hold : int
        Number of timesteps for which the input is presented.
    n_out : int
        Number of timesteps for which loss is evaluated.
    X_dim : int
        Dimension of the ambient space in which clusters are generated.
    Y_classes : int
        Number of class labels.
    clust_sig : float
        Standard deviation of each cluster.
    model_seed : int
        Seed for generating input and model weights.
    hid_nonlin : str
        Activation function for the hidden units, or if using a sompolinsky style recurrent network the nonlinear
        transfer function.
    stop_epoch : int
        The number of epochs to train for.
    learning_rate : float
        Learning rate for optimizer.
    patience_before_stopping : int
        Number of consecutive epochs to wait for which there is no improvement to the (cumulative average) validation
        loss before ending training.
    batch_size : int
        Size of each training data minibatch.
    loss : str
        The loss function to use. Options are "mse" for mean squared error and "cce" for categorical cross entropy.
    optimizer : str
        The optimizer to use. Options are "sgd" for standard stochastic gradient descent and "rmsprop" for RMSProp.
    perc_val : float
        The percent of input data to use for validation.
    scheduler : str
        The strategy used to adjust the learning rate through training. Options are None for constant learning rate
        through training, "plateau" for reducing the learning rate by a multiplicative factor after a plateau of a
        certain number of epochs, and "steplr" for reducing the learning rate by a multiplicative factor. In both
        cases, the number of epochs is specified by scheduler_patience and the multiplicative factor by
        scheduler_factor.
    learning_patience : int
        If using plateau scheduler, this is the number of epochs over which to measure that a plateau has been
        reached. If using steplr scheduler, this is the number of epochs after which to reduce the learning rate.
    scheduler_factor : float
        The multiplicative factor by which to reduce the learning rate.
    Win : str
        Type of input weights to use. Can be 'diagonal_first_two' for feeding inputs to only the first two neurons
         in the network or 'orthogonal' for a (truncated) orthogonal/unitary matrix.
    Wrec_rand_proportion : float
        The proportion of Wrec that should initially be random. Wrec will be initialized as a convex combination of a
        random matrix and an orthogonal matrix, weighted by Wrec_rand_proportion.
    net_architecture : str
        The type of network architecture to use. Options are "vanilla_rnn" for a vanilla RNN, "sompolinsky" for a
        Sompolinsky style RNN, and "feedforward" for a feedforward network.
    input_scale : float
        Global scaling of the inputs.
    g_radius : float
        Magnitude of the largest eigenvalue of the random part of the recurrent weight matrix. This holds exactly
        (i.e. the random matrix is rescaled so that this is satisfied exactly), not just on average.
    dt : float
        Size of the timestep to use for the discretization of the dynamics if using an RNN.
    param_regularization : Optional[str]
        WARNING: Not implemented yet. Type of regularization to apply to the parameters. Options are 'l2' and 'l1'.
    param_regularization_weight : float
        WARNING: Not implemented yet. Weighting of regularization to apply to the parameters.
    activity_regularization : Optional[str]
        WARNING: Not implemented yet. Type of regularization to apply to the hidden unit activations.
        Options are 'l2' and 'l1'.
    activity_regularization_weight : float
        WARNING: Not implemented yet. Weighting of regularization to apply to the hidden unit activations.
    input_style: str
        Input style. Currently 'hypercube' is the only valid option.
    freeze_input: bool
        Whether or not to present the same input every epoch. If False, new input samples are drawn every epoch
    rerun: bool
        Whether or not to run the simulation again even if a matching run is found on disk. True means run the
        simulation again.

    Returns
    -------
        torch.nn.Module
            The trained network model.
        dict
            A collection of all the (meta) parameters used to specify the run. This is basically a dictionary of the
            input arguments to this function.
        str
            The directory where the output for the run is stored, including model parameters over training.
    """

    if param_regularization in (None, 'None'.casefold(), 'NA'.casefold()):
        param_regularization = 'na'
    if param_regularization_weight in (None, 'None'.casefold(), 'NA'.casefold()):
        param_regularization_weight = 'na'
    if activity_regularization in (None, 'None'.casefold(), 'NA'.casefold()):
        activity_regularization = 'na'
    if activity_regularization_weight in (None, 'None'.casefold(), 'NA'.casefold()):
        activity_regularization_weight = 'na'
    loc = locals()
    args = inspect.getfullargspec(train)[0]
    arg_dict = {arg: loc[arg] for arg in args}
    del arg_dict['rerun']

    torch.manual_seed(model_seed)  # Set random seed.

    def ident(x):
        return x

    ## Take care of parameter options
    if hid_nonlin == 'linear'.casefold():
        nonlin = ident
    elif hid_nonlin == 'tanh'.casefold():
        nonlin = torch.tanh
    elif hid_nonlin == 'relu'.casefold():
        nonlin = torch.relu
    else:
        raise ValueError('Unrecognized option for hid_nonlin')

    if param_regularization == 'l1':
        def param_regularization_f(p):
            # mean is over all elements of p, even if p is a matrix
            return param_regularization_weight * torch.mean(torch.abs(p))
    elif param_regularization == 'l2':
        def param_regularization_f(p):
            # mean is over all elements of p, even if p is a matrix
            return param_regularization_weight * torch.mean(p ** 2)
    elif param_regularization == 'na':
        param_regularization_f = None
    else:
        raise ValueError("param_regularization option not recognized")
    if activity_regularization == 'l1':
        def activity_regularization_f(x):
            # mean is over all elements of p, even if p is a matrix
            return activity_regularization_weight * torch.mean(torch.abs(x))
    elif activity_regularization == 'l2':
        def activity_regularization_f(x):
            # mean is over all elements of p, even if p is a matrix
            return activity_regularization_weight * torch.mean(x ** 2)
    elif activity_regularization == 'na':
        activity_regularization_f = None
    else:
        raise ValueError("activity_regularization option not recognized.")

    if net_architecture != 'feedforward'.casefold():
        loss_points = torch.arange(n_lag - n_out, n_lag + n_hold - 1)
    else:
        loss_points = torch.tensor([0], dtype=int)
    num_train = int(round((1 - perc_val) * samples_per_epoch))

    m = loss_points.shape[0]
    # if loss in ('categorical_crossentropy'.casefold(), 'cce'.casefold()):
    #     criterion_CEL = torch.nn.CrossEntropyLoss()
    #     if net_architecture == 'feedforward':  # This implies that the output does not have a time dimension
    #         def criterion(output, label):
    #             return criterion_CEL(output, label.long())
    #     else:  # The output does have a time dimension
    #         def criterion(output, label):
    #             cum_loss = 0
    #             for i0 in loss_points:
    #                 cum_loss += criterion_CEL(output[:, i0], label[:, i0].long())
    #             return cum_loss / m
    # elif loss in ('mean_square_error'.casefold(), 'mean_squared_error'.casefold(), 'mse'.casefold()):
    #     criterion_mse = torch.nn.MSELoss()
    #
    #     def criterion_single_timepoint(output, label):  # The output does not have a time dimension
    #         label_onehot = torch.zeros(label.shape[0], Y_classes)
    #         for i0 in range(Y_classes):
    #             label_onehot[label == i0, i0] = 1
    #         return criterion_mse(output, .7 * label_onehot)
    #
    #     if net_architecture == 'feedforward':  # This implies that the output does not have a time dimension
    #         criterion = criterion_single_timepoint
    #     else:  # The output does have a time dimension
    #         def criterion(output, label):
    #             cum_loss = 0
    #             for i0 in loss_points:
    #                 cum_loss += criterion_single_timepoint(output[:, i0], label[:, i0])
    #             cum_loss = cum_loss / m
    #             return cum_loss
    # else:
    #     raise AttributeError("loss option not recognized.")

    if loss in ('categorical_crossentropy'.casefold(), 'cce'.casefold()):
        criterion_CEL = torch.nn.CrossEntropyLoss()

        # if net_architecture == 'feedforward':  # This implies that the output does not have a time dimension
        #     def criterion(output, label):
        #         return criterion_CEL(output, label.long())
        # else:  # The output does have a time dimension
        def criterion(output, label):
            cum_loss = 0
            for i0 in loss_points:
                cum_loss += criterion_CEL(output[:, i0], label[:, i0].long())
            return cum_loss / m
    elif loss in ('mean_square_error'.casefold(), 'mean_squared_error'.casefold(), 'mse'.casefold()):
        criterion_mse = torch.nn.MSELoss()

        def criterion_single_timepoint(output, label):  # The output does not have a time dimension
            label_onehot = torch.zeros(label.shape[0], Y_classes)
            for i0 in range(Y_classes):
                label_onehot[label == i0, i0] = 1
            return criterion_mse(output, .7 * label_onehot)

        # if net_architecture == 'feedforward':  # This implies that the output does not have a time dimension
        #     criterion = criterion_single_timepoint
        # else:  # The output does have a time dimension
        def criterion(output, label):
            cum_loss = 0
            for i0 in loss_points:
                cum_loss += criterion_single_timepoint(output[:, i0], label[:, i0])
            cum_loss = cum_loss / m
            return cum_loss
    else:
        raise AttributeError("loss option not recognized.")

    if Win == 'identity'.casefold():
        Win_instance = input_scale * torch.eye(X_dim, N)
    elif Win in ('orth'.casefold(), 'orthogonal'.casefold(), 'orthog'.casefold()):
        temp = torch.empty(X_dim, N)
        temp = torch.nn.init.orthogonal_(temp)
        temp = temp / torch.mean(torch.abs(temp))
        temp = input_scale * temp / np.sqrt(X_dim)
        Win_instance = temp
    else:
        raise AttributeError("Win option not recognized.")

    Wout_instance = torch.randn(N, Y_classes) * (.3 / math.sqrt(N))

    brec = torch.zeros(N)
    bout = torch.zeros(Y_classes)
    J = torch.randn(N, N) / math.sqrt(N)
    top_ew = get_max_eigval(J)[0]
    top_ew_mag = torch.sqrt(top_ew[0] ** 2 + top_ew[1] ** 2)
    J_scaled = g_radius * (J / top_ew_mag)
    Q = torch.nn.init.orthogonal_(torch.empty(N, N))
    Q_scaled = g_radius * Q
    if net_architecture in ('somp'.casefold(), 'sompolinsky'.casefold()):
        Wrec = Wrec_rand_proportion * J_scaled + (1 - Wrec_rand_proportion) * Q
        model = models.SompolinskyRNN(Win_instance, Wrec, Wout_instance, brec, bout, nonlin, dt=dt,
                                      output_over_recurrent_time=True)

    elif net_architecture == 'vanilla_rnn'.casefold():
        Wrec = (1 - dt) * torch.eye(N, N) + dt * g_radius * (J / top_ew_mag)
        model = models.RNN(Win_instance, Wrec, Wout_instance, brec, bout, nonlin, output_over_recurrent_time=True)

    elif net_architecture == 'feedforward'.casefold():
        # layer_weights: List[Tensor], biases: List[Tensor], nonlinearities: List[Callable]
        Wrec = (1 - dt) * torch.eye(N, N) + dt * g_radius * (J / top_ew_mag)
        # Wrec = g_radius * (J / top_ew_mag)
        layer_weights = [Win_instance]
        biases = [torch.zeros(N)]
        nonlinearities = [nonlin]
        for i0 in range(n_lag + n_hold - 2):
            layer_weights.append(Wrec.clone())
            biases.append(torch.zeros(N))
            nonlinearities.append(nonlin)
        layer_weights.append(Wout_instance)
        biases.append(torch.zeros(Y_classes))
        nonlinearities.append(ident)

        model = models.FeedForward(layer_weights, biases, nonlinearities)
    else:
        raise AttributeError('Option for net_architecture not recognized.')

    if optimizer.casefold() == 'sgd'.casefold():  # case-insensitive equality check
        optimizer_instance = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=learning_rate)
    elif optimizer.casefold() == 'rmsprop'.casefold():
        # noinspection PyUnresolvedReferences
        optimizer_instance = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=learning_rate)
    else:
        raise AttributeError('optimizer option not recognized.')
    if scheduler == 'plateau'.casefold():
        learning_scheduler_torch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_instance,
                                                                              factor=scheduler_factor,
                                                                              patience=learning_patience,
                                                                              threshold=1e-7,
                                                                              threshold_mode='abs', min_lr=0,
                                                                              verbose=True)
    elif scheduler == 'steplr'.casefold():
        learning_scheduler_torch = torch.optim.lr_scheduler.StepLR(optimizer_instance, step_size=learning_patience,
                                                                   gamma=scheduler_factor)
    else:
        raise AttributeError('scheduler option not recognized.')
    learning_scheduler_instance = model_trainer.LearningScheduler(learning_scheduler_torch)

    # todo: fix n_lag
    out = classification_task.delayed_mixed_gaussian(samples_per_epoch, perc_val, X_dim, Y_classes, X_clusters,
                                                     n_hold, n_lag, clust_sig, 2 * model_seed + 1,
                                                     3 * model_seed + 13, cluster_method=5, avg_magn=1,
                                                     freeze_input=freeze_input)
    datasets, centers, cluster_class_label = out

    # datasets = {x: dataloaders[x].dataset for x in ('train', 'val')}
    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=False, num_workers=0)
        for x in ('train', 'val')}

    batches_per_epoch = {x: int(len(datasets[x]) / dataloaders[x].batch_size) for x in ('train', 'val')}
    stats_trackers = {x: na.model_trainer.DefaultStatsTracker(batches_per_epoch[x], x, accuracy=False) for x in
                      ('train', 'val')}

    def save_model_criterion(stat_dict):
        return stat_dict['epoch_end']

    start_epoch = 0
    # run_dirs = mom.get_dirs_for_run(arg_dict, TABLE_PATH, compare_exclude=['num_epochs'])
    if rerun:
        run_id, run_dir = mom.make_dir_for_run(arg_dict, TABLE_PATH)
    else:
        load_dir, check_num = na.model_loader_utils.smart_load(arg_dict, TABLE_PATH, stop_epoch,
                                                               model, optimizer_instance,
                                                               learning_scheduler_instance.scheduler,
                                                               ['stop_epoch'])

        if check_num is not None and check_num == stop_epoch:
            run_id = int(str(load_dir.parts[-1])[4:])
            print("Parameters match existing previous run. Loading previous run.")
            # outputs, params, run_id, run_dir = mom.load_data([], arg_dict, TABLE_PATH)
            outputs, params_loaded = mom.load_from_id(run_id, TABLE_PATH)

            params = dict(dataloaders=dataloaders, datasets=datasets,
                          stats_trackers=stats_trackers, learning_scheduler_instance=learning_scheduler_instance,
                          optimizer_instance=optimizer_instance)
            params.update(arg_dict)

            return model, params, outputs, load_dir
        elif check_num is not None and check_num < stop_epoch:  # This won't load if check_num = 0, but that's okay
            print("Parameters match existing previous run but more epochs needed. Loading last trained epoch and "
                  "training with more epochs")
            run_id = int(str(load_dir.parts[-1])[4:])
            start_epoch = check_num
            run_dir = load_dir
        else:  # (load_dir is None)
            run_id, run_dir = mom.make_dir_for_run(arg_dict, TABLE_PATH)

    returned_models, history_and_machinery = na.model_trainer.train_model(model, dataloaders, criterion,
                                                                          optimizer_instance,
                                                                          learning_scheduler_instance, start_epoch,
                                                                          stop_epoch, run_dir,
                                                                          stats_trackers=stats_trackers,
                                                                          save_model_criterion=save_model_criterion)
    model = returned_models[0]
    stats_history = history_and_machinery['stats_history']
    stats_trackers = history_and_machinery['stats_trackers']

    learning_scheduler_instance = history_and_machinery['learning_scheduler']
    optimizer_instance = history_and_machinery['optimizer']

    params = dict(dataloaders=dataloaders, datasets=datasets,
                  stats_trackers=stats_trackers, learning_scheduler_instance=learning_scheduler_instance,
                  optimizer_instance=optimizer_instance)
    params.update(arg_dict)
    outputs = dict(stats_history=stats_history, stats_trackers=stats_trackers)

    mom.write_output(outputs, params, arg_dict, run_dir, overwrite=True)
    return model, params, outputs, run_dir


if __name__ == '__main__':
    train(200, 60, 100, 10, 1, 1, 200, 2, stop_epoch=60, batch_size=20, learning_rate=.001, rerun=False)
    train(200, 60, 100, 10, 1, 1, 200, 2, stop_epoch=100, batch_size=20, learning_rate=.001, rerun=False)
    # train(10, 3, 100, 10, (('theta', 'x'), ('x',)), 20)
