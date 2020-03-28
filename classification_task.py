import numpy as np
import scipy
import torch
from torch.utils.data import Dataset

norm = np.linalg.norm


def onehot(x):
    # if the array x is filled with integers then it makes each of this integer a class returning the onehot
    # representation in the last dimension of x
    x_unique = np.unique(x)
    y = np.zeros((x.shape[0], x_unique.shape[0]))
    for x_el, idx in enumerate(x_unique): y[np.where(x == x_el), int(idx)] = 1
    return y


def onehot2(x, N):
    # if the array x is filled with integers then it makes each of this integer a class returning the onehot
    # representation in the last dimension of x
    x_unique = np.arange(N)
    y = np.zeros((x.shape[0], x_unique.shape[0]))
    for x_el, idx in enumerate(x_unique): y[np.where(x == x_el), int(idx)] = 1
    return y


def undo_onehot(x):
    # returns the array X where instead of the last dimension being a onehot representation, the class index is returned
    sequence = np.where(x != 0)[-1]
    x_seq = np.reshape(sequence, x.shape[:-1])
    return x_seq

class InpData(Dataset):
    """"""

    def __init__(self, X, Y):
        # self.X = torch.from_numpy(X).float()
        self.X = X
        self.Y = Y
        # self.Y = torch.from_numpy(Y[:, -num_loss_pnts:]).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class Gaussian_Spheres(Dataset):

    def __init__(self, centers, center_labels, final_time, max_samples=None, noise_sig=0.1, nonzero_time_points=None,
                 squeeze=False):
        self.centers = torch.from_numpy(centers).float()
        self.center_labels = torch.from_numpy(center_labels).float()
        self.max_samples = max_samples
        self.num_class_labels = len(np.unique(center_labels))
        self.final_time = final_time

        # self.centers_shape = centers.shape
        # self.Y = torch.from_numpy(np.argmax(Y, axis=-1)).long()
        # self.Y = torch.from_numpy(np.argmax(center_labels, axis=-1)).float()
        self.nonzero_time_points = nonzero_time_points
        self.noise_sig = noise_sig
        self.squeeze = squeeze
        self.cluster_identity = None
        # self.noise = torch.autograd.Variable(torch.empty(X.shape[-1]))

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):

        X = self.centers.clone()
        Y = self.center_labels.clone()

        idx_singleton = False
        if isinstance(idx, slice):
            num_draws = len(range(*idx.indices(self.__len__())))
        elif hasattr(idx, '__len__'):
            num_draws = len(idx)
        else:
            num_draws = 1
            idx_singleton = True

        # This is a uniform draw from the clusters. There is no guarantee of draws being spread out
        #   "as much as possible".
        # sequence = np.random.choice(self.centers.shape[0], num_draws, True)

        # This is a draw that tries to hit clusters as evenly as possible.
        # m = int(round(num_draws / self.centers.shape[0]))
        m = np.mod(num_draws, self.centers.shape[0])
        sequence = np.mod(np.arange(num_draws), self.centers.shape[0])
        if m > 0:
            leftover = np.random.choice(self.centers.shape[0], m, False)
            sequence[-m:] = leftover
        np.random.shuffle(sequence)

        X = X[sequence]
        Y = Y[sequence]

        noise = self.noise_sig * torch.randn(num_draws, X.shape[-1])
        X = X + noise
        if not self.squeeze:
            X = X[:, None]
            X = X.repeat(1, self.final_time, 1)
            Y = Y[:, None]
            Y = Y.repeat(1, self.final_time)
            mask = np.ones(X.shape[1], np.bool)
            mask[self.nonzero_time_points] = 0
            mask = np.nonzero(mask)[0]
            X[:, mask, :] = 0.

        self.cluster_identity = sequence

        if idx_singleton:
            return X[0], Y[0]
        else:
            return X, Y


def draw_centers_binary(num_clusters, dim):
    """Each cluster has coordinates with entries drawn from +/-1"""

    X = np.random.choice([-1, 1], (num_clusters, dim))
    X_unique, ind = np.unique(X, axis=0, return_index=True)
    num_unique = X_unique.shape[0]
    mask = np.ones(num_clusters, np.bool)
    mask[ind] = 0
    # other_data = data[mask]
    counter = 0
    while num_unique < num_clusters:
        temp = np.random.choice([-1, 1], (num_clusters - num_unique, dim))
        X[mask] = temp
        X_unique, ind = np.unique(X, axis=0, return_index=True)
        num_unique = X_unique.shape[0]
        mask = np.ones(num_clusters, np.bool)
        mask[ind] = 0
        counter = counter + 1
        if counter > 10000:
            raise TimeoutError("Did not find cluster arrangement. Try increasing X_dim.")
    return X / np.sqrt(X.shape[-1])


def draw_centers_ball(num_clusters, dim, avg_magn, min_sep):
    """Way of getting clusters that don't overlap (uniformly distributed within radius of origin, with minimum
    separation distance)"""
    from mython.gen_pnts import uniform_ball
    exp_d_b0 = 0.58440832 * dim ** (-0.33270217) - 0.04467088
    bounding_radius = avg_magn / exp_d_b0  # radius of ball to get expected magnitude of points of avg_magn

    # For 99.9999% falling within
    # if not min_sep_defined:
    #     min_sep = noise_sigma * (0.59943537 * (dim - 1)**0.57832581 + 4.891638480163717)

    X = []
    p = bounding_radius * uniform_ball(dim, 1)[0]
    X.append(p)
    counter = 0
    for i1 in range(num_clusters - 1):
        min_sep_p = min_sep - 1
        while min_sep_p < min_sep:
            p = bounding_radius * uniform_ball(dim, 1)[0]
            min_sep_p = 10000 * bounding_radius  # Just a very large number...
            for x in X:
                sep = norm(np.array(x) - p)
                min_sep_p = min(min_sep_p, sep)
            counter = counter + 1
            # print(sep)
        X.append(p)
    X = np.array(X)
    print("minimum cluster separation allowed: " + str(min_sep))
    from scipy.spatial.distance import pdist
    print("minimum cluster separation generated: " + str(np.min(pdist(X))))
    # X_norm = np.max(norm(X, axis=1))
    # X_norm = np.max(X)
    # X = X / X_norm
    return np.array(X)


def draw_centers_hypercube(num_clusters, dim, min_sep):
    """Uniformly distributed on hypercube."""
    # Way of getting clusters that don't overlap (uniformly distributed within radius of origin, with minimum
    #   separation distance)
    # For 99.9999% falling within
    # if not min_sep_defined:
    #     min_sep = noise_sigma * (0.59943537 * (dim - 1)**0.57832581 + 4.891638480163717)
    print("minimum separation: ", min_sep)

    X = []
    # p = 4*(np.random.rand() - 0.5)
    p = 4 * (np.random.rand(dim) - 0.5)
    X.append(p)
    counter = 0
    for i1 in range(num_clusters - 1):
        min_sep_p = min_sep - 1
        while min_sep_p < min_sep:
            p = 4 * (np.random.rand(dim) - 0.5)
            min_sep_p = 100000  # Just a very large number...
            for x in X:
                sep = norm(np.array(x) - p)
                min_sep_p = min(min_sep_p, sep)
            counter = counter + 1
            print("separation for this draw: ", sep)
        X.append(p)
    X = np.array(X)
    print("minimum cluster separation allowed: " + str(min_sep))
    from scipy.spatial.distance import pdist
    print("minimum cluster separation generated: " + str(np.min(pdist(X))))
    # X_norm = np.max(norm(X, axis=1))
    # X_norm = np.max(X)
    # X = X / X_norm
    return np.array(X)


def delayed_mixed_gaussian(num_trials, perc_val, X_dim, Y_classes, X_clusters, n_hold, final_time_point,
                           noise_sigma=0, cluster_seed=None, assignment_and_noise_seed=None,
                           cluster_method=3, avg_magn=0.3, min_sep=None, freeze_input=False):
    """This is a classification task where a stimulus is presented to the network in (either sustained or only on the
    first timestep) for n_lag timesteps. Each stimulus is sampled from X_clusters labelled into Y_classes. The number
    of classes can be smaller but not bigger than X_clusters. The center of each cluster is generated randomly while
    its variance is controlled by noise_sigma. # The final shape of X and Y are (num_trials, final_time_point,
    X_dim/Y_classes).

    Args:
        num_trials (int): Number of input points to draw.
        final_time_point (int): Number of timesteps from stimulus onset to end of loss evaluation.
        X_dim (int): Dimension of the ambient space in which clusters are generated
        Y_classes (int): Number of class labels
        X_clusters (int): Number of clusters
        n_hold (int): Number of timesteps for which the input is presented
        final_time_point (int): Final timestep, and the number of timesteps from stimulus onset to end of loss
            evaluation.
        noise_sigma (float): Standard deviation of each cluster
        cluster_seed (int): rng seed for cluster center locations
        assignment_and_noise_seed (int): rng seed for assignment of class labels to clusters
        cluster_method (int): Method to use for generating cluster centers. They are
            1. Each center is a vector whose components are drawn from a standard normal distribution.
            2. Each center is a vector whose components are drawn uniformly from +/-1, with no overlapping centers.
            3. Each center is a vector that is drawn uniformly in a ball with radius min_sep * num_clusters such that
                each centers is at a distance of at least min_sep from every other center.
            4. Each center has coordinates drawn from [1,0]
            5. Each center is drawn uniformly from the hypersphere centered at the origin with side length 4.
        min_sep (float): Minimal separation of centers if using cluster_method 3.

    Returns:
        dict[str, torch.dataset]: Dictionary with keys 'train' and 'val' for training and validation datasets,
            respectively
        torch.Tensor: Locations for the centers of the clusters
        torch.Tensor: Class labels for the clusters

    """
    # loc = locals()
    # import inspect
    # args = inspect.getfullargspec(classification)[0]
    # arg_dict = {arg: loc[arg] for arg in args}
    # print(arg_dict)

    norm = np.linalg.norm

    min_sep_defined = True

    if min_sep is None:
        min_sep_defined = False
        min_sep = noise_sigma * 10

    # Really hacky way of getting clusters that don't overlap
    if not min_sep_defined:
        min_sep = noise_sigma * (0.59943537 * (X_dim - 1) ** 0.57832581 + 4.891638480163717)
    np.random.seed(cluster_seed)
    if cluster_method == 1:
        centers = np.random.randn(X_clusters, X_dim)
    elif cluster_method == 2:
        centers = draw_centers_binary(X_clusters, X_dim)
    elif cluster_method == 3:
        centers = draw_centers_ball(X_clusters, X_dim, avg_magn, min_sep)
    elif cluster_method == 4:
        centers = np.random.choice([0, 1], (X_clusters, X_dim))
    elif cluster_method == 5:
        centers = draw_centers_hypercube(X_clusters, X_dim, min_sep)
    else:
        raise AttributeError("cluster_method option not recognized.")

    # print(np.mean(np.abs(X)))
    num_train = int(round((1 - perc_val) * num_trials))
    num_val = num_trials - num_train

    cluster_class_label = np.mod(np.arange(X_clusters), Y_classes).astype(int)
    # Y = onehot(Y)
    # Y = Y[:, np.newaxis, :]
    # Y = Y.repeat(final_time_point, axis=1)
    # nonzero_time_points = torch.zeros(final_time_point, dtype=int)
    # nonzero_time_points[:n_hold] = 1
    nonzero_time_points = torch.arange(n_hold)

    torch.manual_seed(assignment_and_noise_seed)
    if freeze_input:
        dataset = Gaussian_Spheres(centers, cluster_class_label, final_time_point, max_samples=num_trials,
                                   noise_sig=noise_sigma, nonzero_time_points=nonzero_time_points)
        X, Y = dataset[:]
        # X = 10*X
        class_datasets = {'train': InpData(X[:num_train], Y[:num_train]),
                          'val': InpData(X[num_train:], Y[num_train:])}
        # dataset_frozen = InpData(X, Y)
    else:
        class_datasets = {
            'train': Gaussian_Spheres(centers, cluster_class_label, final_time_point, max_samples=num_train,
                                      noise_sig=noise_sigma, nonzero_time_points=nonzero_time_points),
            'val': Gaussian_Spheres(centers, cluster_class_label, final_time_point, max_samples=num_val,
                                    noise_sig=noise_sigma, nonzero_time_points=nonzero_time_points)}

    return class_datasets, centers, cluster_class_label

    # if return_verbose:
    #     return centers, Y, Z, sequence, centers, cluster_class_label
    # else:
    #     return centers, Y, Z


def variance_mag_calcs():
    # %% Find relationship between the expected magnitude of the entries of X drawn uniformly from a ball
    # and the dimension of the ball. This magnitude should be around 1 to make sure it's in the "right spot" of
    # the neural network nonlinearity.

    from scipy.optimize import curve_fit
    dmax = 300
    ds = np.arange(1, dmax + 1)
    num_samps = 10000
    exps = np.zeros(len(ds))
    for i1, d in enumerate(ds):
        B = uniform_ball(d, num_samps)
        exps[i1] = np.mean(np.abs(B))

    # def func(x, a, b, c):
    #     return a * np.exp(-b * x) + c
    def func(x, a, b, c):
        return a * x ** (-b) + c

    # def func(x, a, b, c):
    #     return a * b**(x) + c

    popt, pcov = curve_fit(func, ds, exps)
    # optimal parameters are a = 0.58440832, b = 0.33270217, c = -0.04467088
    # popt, pcov = curve_fit(func, ds, exps, p0=popt)

    fig, ax = plt.subplots()
    ax.plot(ds, exps, '-x')
    ax.plot(ds, func(ds, *popt), 'r-x')
    # ax.semilogy(ds, exps - 0.5 + 1/np.e, '-x')
    # ax.semilogy(ds, np.exp(-ds), '-x')
    # ax.plot(ds, 0.60264459930942094/np.sqrt(ds), 'r-x')
    fig.show()
    print()


def distance_prob_calcs():
    # %% Find relationship between the ratio of the distance between two balls to the variance of the balls and
    # the dimensionality of the point cloud made up of the two balls. Want this to be ~1 (so the position of the
    # centers dominates the dimensionality rather than the spread of points around the balls).

    sig = 1
    dists = np.arange(1, 401)
    dims = np.arange(2, 301)
    # dims = [200]
    eff_dims = np.zeros((len(dims), len(dists)))
    for i1, dim in enumerate(dims):
        B1_0 = np.random.randn(20000, dim)
        B2_0 = np.random.randn(20000, dim)
        v = np.ones(dim)
        v = v / np.linalg.norm(v)
        for i2, dist in enumerate(dists):
            B1 = B1_0 - 0.5 * dist * v
            B2 = B2_0 + 0.5 * dist * v
            X = np.concatenate((B1, B2), axis=0)
            eff_dims[i1, i2] = dim_tools.get_effdim(X)

    np.save("cluster_distance_data", eff_dims)

    print()
    fig, ax = plt.subplots()
    for i1, dim in enumerate(dims):
        ax.plot(dists, eff_dims[i1], color=plt.cm.jet(i1 / (len(dims) - 1)))
    ax.set_xlabel("distance")
    ax.set_ylabel("effective dimension")
    fig.show()
    fig, ax = plt.subplots()
    for i1, dist in enumerate(dists):
        ax.plot(dims, eff_dims[:, i1], color=plt.cm.jet(i1 / (len(dists) - 1)))
    ax.set_xlabel("dimension")
    ax.set_ylabel("effective dimension")
    fig.show()
    # fig, ax = plt.subplots()
    # for i1, dist in enumerate(dists[20:]):
    #     i1 = i1 + 20
    #     ax.plot(dims, eff_dims[:, i1], color=plt.cm.jet(i1 / (len(dists) - 1)))
    # ax.set_xlabel("dimension")
    # ax.set_ylabel("effective dimension")
    # fig.show()

    slopes = (eff_dims[-1, :] - eff_dims[0, :]) / (dims[-1] - dims[0])

    def func(x, a, b, c):
        return -a / (1 + b * np.exp(-c * x)) + 1 + a / (1 + b * np.exp(-0.5 * c))

    popt, pcov = curve_fit(func, dists, slopes)  # Optimal values are a=1.01412623, b=33.52570228, c=0.44890286

    fig, ax = plt.subplots()
    # ax.plot(dists, slopes)  # This is the function we'll be fitting.
    dists = np.linspace(0.5, 100)
    ax.plot(dists, func(dists, *popt))
    fig.show()

    dim = 100
    D = 1.1
    a, b, c = 1.01412623, 33.52570228, 0.44890286
    A = 1 + a / (1 + b * np.exp(-0.5 * c))
    min_sep = -(1 / c) * np.log((a / (A - D / dim) - 1) / b)

    B1_0 = np.random.randn(20000, dim)
    B2_0 = np.random.randn(20000, dim)

    v = np.ones(dim)
    v = v / np.linalg.norm(v)

    B1 = B1_0 - 0.5 * min_sep * v
    B2 = B2_0 + 0.5 * min_sep * v
    X = np.concatenate((B1, B2), axis=0)
    print(dim_tools.get_effdim(X))
