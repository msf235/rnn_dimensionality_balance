# File for utility functions that take a model as the first parameter
from matplotlib import pyplot as plt
from analysis_tools import plot_tools, dim_tools
import numpy as np
import torch

def extend_input(X, T, T_before=None):
    s = X.shape
    if T_before is None or T_before < 1:
        Xn = torch.zeros((s[0], T, s[2]))
        Xn[:, :s[1]] = X
    else:
        # Xn = np.zeros((s[0], T+T_before, s[2]))
        Xn = torch.zeros((s[0], T+T_before, s[2]))
        Xn[:, T_before:T_before+s[1]] = X

    return Xn

def get_centers_of_clusters(cluster_labels, hid):
    """
    First dimension of hid should be trials.

    Args:
        hid ():

    Returns:

    """
    # X_clusters = self.params.X_clusters
    X_clusters = np.max(cluster_labels)
    num_trials = hid.shape[0]
    if hid.ndim < 3:
        hid_centroids = np.zeros((X_clusters, hid.shape[1]))
    else:
        hid_centroids = np.zeros((X_clusters, hid.shape[1], hid.shape[2]))

    for i2 in range(X_clusters):
        cluster_bool = cluster_labels == i2
        hid_cluster = hid[cluster_bool[:num_trials]]
        hid_centroids[i2] = np.mean(hid_cluster, axis=0)

    return hid_centroids


def visualize_input(dataset, num_points=None, cmap=None):
    """

    Args:
        dataset (theano dataset): Should return input data as first argument and targets as second.

    Returns:

    """
    if num_points is None:
        num_points = len(dataset)

    X, Y = dataset[:num_points]
    X = X[:, 0].numpy()
    X_pcs = dim_tools.get_pcs_stefan(X, [0,1])
    Y = Y[:, -1].numpy()
    # Y = np.argmax(Y[:, -1].numpy(), axis=-1)
    # Y = np.argmax(Y, axis=-1)
    if cmap is None:
        cs = plot_tools.get_color(Y)
    else:
        cs = plot_tools.get_color(Y, cmap=cmap)

    fig, ax = plt.subplots()
    # ax.scatter(X_pcs[:,0], X_pcs[:,1], c=cs, s=10)
    ax.scatter(X[:,0], X[:,1], c=cs, s=10)
    fig.show()

# def visualize_model(model, inp_data, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)


# def get_activations(model, X):
#     # Todo: make this function deal with pytorch tensors and numpy arrays nicely
#     if not hasattr(model, 'get_activations'):
#         raise AttributeError("model needs to have a get_activations method.")
#
#     hid, pred = model.get_activations(X)
#     return hid, pred
