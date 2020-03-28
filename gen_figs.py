import copy
import sys
import pickle as pkl
from scipy import stats as st
import matplotlib.ticker
import torch
from matplotlib import animation, ticker
import subprocess
# import warnings
# warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
import numpy as np
import scipy as sc
import os
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

import trainer
import utils
import network_analysis as na

from analysis_tools import plot_tools, dim_tools

plt.interactive(False)
import matplotlib.ticker as tck
from joblib import Parallel, delayed, Memory
# import seaborn as sns
# import pandas as pd
from mpl_toolkits.mplot3d.art3d import juggle_axes

memory = Memory(location='../joblib_cache', verbose=2)


def point_replace(a_string):
    a_string = str(a_string)
    return a_string.replace(".", "p")


# def std(x, **kwargs): return np.std(x, ddof=1)

def median_and_bound(samples, perc_bound, dist_type='gamma', loc=0, shift=0, reflect=False, show_fit=False):
    def do_reflect(x, center):
        return -1 * (x - center) + center

    if dist_type == 'gamma':
        # gam = st.gamma
        # a1 = 2
        # scale = 200
        # xx = np.linspace(0,1500)
        # yy = gam.pdf(xx, a1, loc=0, scale=scale)
        # median_true = st.gamma.median(a1, loc=0, scale=scale)
        #
        # samples = gam.rvs(a1, loc=0, scale=scale, size=40)
        if np.sum(samples[0] == samples) == len(samples):
            median = samples[0]
            interval = [samples[0], samples[0]]
            return median, interval

        if reflect:
            # reflect_point = loc + shift
            samples_reflected = do_reflect(samples, loc)
            shape_ps, loc_fit, scale = st.gamma.fit(samples_reflected, floc=loc + shift)
            median_reflected = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval_reflected = np.array(st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))
            median = do_reflect(median_reflected, loc)
            interval = do_reflect(interval_reflected, loc)
        else:
            shape_ps, loc, scale = st.gamma.fit(samples, floc=loc + shift)
            median = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval = np.array(st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))

        if np.isnan(np.sum(median)) or np.isnan(np.sum(interval)):
            print

        if show_fit:
            fig, ax = plt.subplots()
            ax.hist(samples, density=True)
            xx = np.linspace(np.min(samples), np.max(samples))
            if reflect:
                yy_hat = st.gamma.pdf(do_reflect(xx, loc), shape_ps, loc=loc, scale=scale)
                ax.plot(xx, yy_hat)
                ax.axvline(x=median)
                ax.axvline(x=interval[0], linestyle='--')
                ax.axvline(x=interval[1], linestyle='--')
                plt.show()
                print
            else:
                yy_hat = st.gamma.pdf(xx, shape_ps, loc=loc, scale=scale)
                ax.plot(xx, yy_hat)
                ax.axvline(x=median)
                ax.axvline(x=interval[0], linestyle='--')
                ax.axvline(x=interval[1], linestyle='--')
                plt.show()
            print

            # plt.plot(xx, yy)
            # plt.plot(xx, yy_hat, '--')
            # plt.axvline(x=median)
            # plt.axvline(x=median_true, color='red')
            # plt.axvline(x=interval[0], linestyle='--')
            # plt.axvline(x=interval[1], linestyle='--')
            # plt.show()
            # plt.close()

    return median, interval


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=None, useLocale=None):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText, useLocale=useLocale)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


plt.rcParams['font.size'] = 6
plt.rcParams['font.size'] = 6
plt.rcParams['lines.markersize'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams['axes.titlesize'] = 8

class_style = 'color'
# hid_col = np.array([49, 160, 180]) / 255
# hid_col_edge = np.array([29, 140, 160]) / 255

# cols1 = np.array([49, 80, 160])
# cols1 = np.array([49, 80, 160])
# hid_col = cols1 / 255
# hid_col_edge = (cols1 - 20) / 255

# inp_col = np.array([90, 186, 49]) / 255
# inp_col_edge = np.array([166, 70, 29]) / 255

# cols1 = np.array([49, 180, 160])
# inp_col = cols1 / 255
# inp_col_edge = (cols1 - 20) / 255  # Todo: remove?

# cmap = mcolors.ListedColormap([(.9, .5, 0), (0.04, 0.4, 0.7)])
# cmap2 = mcolors.ListedColormap([(1, .6, 0), (.14, .5, .8)])
# cols1 = np.array([.9, .5, 0])
# cols2 = np.array([0.04, 0.4, 0.7])
# cols1 = np.array([.80, .49, .00])
# cols2 = np.array([0.15, 0.15, 0.15])
# cols1 = np.array([.5, .5, .9])
# cols2 = np.array([0.04, 0.7, 0.54])
cols11 = np.array([90, 100, 170]) / 255
cols12 = np.array([37, 50, 120]) / 255
cols21 = np.array([250, 171, 62]) / 255
cols22 = np.array([156, 110, 35]) / 255
cmap_activation_pnts = mcolors.ListedColormap([cols11, cols21])
cmap_activation_pnts_edge = mcolors.ListedColormap([cols12, cols22])

cmap_line_cycle = plt.get_cmap("tab10")
# cmap_line_cycle = plt.get_cmap("Paired")

# chaos_indicator = 'dashes'
chaos_indicator = 'color'

if chaos_indicator == 'color':
    chaos_marker = '-'
    # chaos_colors = [cmap_line_cycle(1), cmap_line_cycle(2)]
    # chaos_colors = [cmap_line_cycle(0), cmap_line_cycle(1)]
    # chaos_color_0 = np.array([0, 120, 120]) / 255
    # chaos_color_1 = np.array([173, 28, 30]) / 255
    chaos_color_0 = np.array([0, 160, 160]) / 255
    chaos_color_1 = np.array([233, 38, 41]) / 255
    # chaos_colors = [, cmap_line_cycle(1)]
    chaos_colors = [chaos_color_0, chaos_color_1]
else:
    chaos_marker = '--'
    chaos_colors = [cmap_line_cycle(0), cmap_line_cycle(0)]

# cmap = mcolors.ListedColormap([(0.4, 0, 0), (0.55, 0.55, 0.55)])
# cmap2 = mcolors.ListedColormap([(0.8, 0, 0), (0, 0, 0)])


rasterized = False
dpi = 800

ext = 'pdf'
# ext = 'png'
create_svg = False

Win = 'orthog'
folder = 'figs_sompolinsky/' + ext + '/Win_{}/'.format(Win)
# if rasterized and ext == 'pdf':
#     folder = 'pdf_rasterized/'
# elif ext == 'pdf':
#     folder = 'pdf/'
# elif ext == 'png':
#     folder = 'png/'
# elif ext == 'svg':
#     folder = 'svg/'

os.makedirs('figs/', exist_ok=True)
os.makedirs(folder, exist_ok=True)
os.makedirs(folder + 'snaps/', exist_ok=True)
os.makedirs(folder + 'snaps/no_border', exist_ok=True)
os.makedirs(folder + 'snaps/border', exist_ok=True)
os.makedirs(folder + 'snaps_3d/no_border', exist_ok=True)
os.makedirs(folder + 'snaps_3d/border', exist_ok=True)

folder_svg = 'figs/svg/Win_{}/'.format(Win)
os.makedirs(folder_svg, exist_ok=True)
os.makedirs(folder_svg + 'snaps/', exist_ok=True)
os.makedirs(folder_svg + 'snaps/no_border', exist_ok=True)
os.makedirs(folder_svg + 'snaps/border', exist_ok=True)
os.makedirs(folder_svg + 'snaps_3d/no_border', exist_ok=True)
os.makedirs(folder_svg + 'snaps_3d/border', exist_ok=True)

# # figsize = (2, 1.6)
# figsize = (1.5, 1.2)
# # figsize_small = (1, 0.8)
# figsize_small = (1.15, .92)
# # figsize = figsize_small
# # figsize_smaller = (.55, 0a.5)
# figsize_smaller = (.8, 0.6)
# figsize_smallest = (.3, 0.2)
# # figsize_smallest = (.4, 0.25)
# ax_pos = (0, 0, 1, 1)


# figsize = (2, 1.6)
figsize = (1.5, 1.2)
figsize_small = (1, 0.8)
# figsize_long = (1.2, 1.2/1.5)
# figsize_smaller = (.55, 0.5)
figsize_smaller = (.8, 0.6)
figsize_smallest = (.3, 0.2)
# figsize_long = (.8, 0.4)
figsize_long = (1, 0.5)
# figsize_smallest = (.4, 0.25)
ax_pos = (0, 0, 1, 1)


def make_fig(figsize=figsize, ax_pos=ax_pos):
    """

    Args:
        figsize ():
        ax_pos ():

    Returns:

    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_pos)
    return fig, ax


def out_fig(fig, name, train_params, subfolder='', show=False, save=True, axis_type=0, name_order=0, data=None):
    """

    Args:
        fig ():
        name ():
        g ():
        show ():
        save ():
        axis_type (int): 0: leave axes alone, 1: Borders but no ticks, 2: Axes off

    Returns:

    """
    # fig.tight_layout()
    # fig.axes[0].ticklabel_format(style='sci',scilimits=(-2,2),axis='both')
    # fig.tight_layout()
    g = train_params['g_radius']
    nonlinearity = train_params['hid_nonlin']
    loss = train_params['loss']
    X_dim = train_params['X_dim']
    ax = fig.axes[0]
    ax.set_rasterized(rasterized)
    if axis_type == 1:
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
    elif axis_type == 2:
        ax.axis('off')
    if name_order == 0:
        fig_name = point_replace(
            folder + subfolder + '{}_g_{}_Xdim_{}_{}_{}'.format(name, g, X_dim, nonlinearity, loss))
        fig_name_svg = point_replace(
            folder_svg + subfolder + '{}_g_{}_Xdim_{}'.format(name, g, X_dim, nonlinearity, loss))
    else:
        fig_name = point_replace(folder + subfolder + 'g_{}_Xdim_{}_{}'.format(g, X_dim, name, nonlinearity, loss))
        fig_name_svg = point_replace(
            folder_svg + subfolder + 'g_{}_Xdim_{}_{}'.format(g, X_dim, name, nonlinearity, loss))
    if save:
        fig_file = fig_name + '.' + ext
        fig_file_svg = fig_name_svg + '.svg'
        fig.savefig(fig_file, dpi=dpi, transparent=True, bbox_inches='tight')
        if create_svg:
            try:
                subprocess.run(["pdf2svg", fig_file, fig_file_svg])
            except FileNotFoundError:
                print("No svg created")

    if show:
        fig.tight_layout()
        fig.show()

    if data is not None:
        with open(folder + subfolder + 'g_{}_Xdim_{}_{}_data'.format(g, X_dim, name), 'wb') as fid:
            pkl.dump(data, fid, protocol=4)


def shade_plot(ax, x_pnts, middle, intervals, color, label, alpha=0.2, **kwargs):
    lines, = ax.plot(x_pnts, middle, color=color, label=label, **kwargs)
    ax.fill_between(x_pnts,
                    intervals[:, 0],
                    intervals[:, 1],
                    color=color,
                    alpha=alpha, lw=0)
    return lines


def chaos_before_after_training_shadeplots(fig_name, g_vals, x_pnts, center_pnts, intervals):
    # axes_labels = ['g', 'epochs', 'time']

    train_test_lines = ['--', '-']
    train_label = ['before training', 'after training']
    g_label = ['edge of chaos', 'strongly chaotic']
    fig, ax = make_fig(figsize_small)
    for i0 in range(center_pnts.shape[0]):
        for i1 in range(center_pnts.shape[1]):
            shade_plot(ax, x_pnts, center_pnts[i0, i1], intervals[i0, i1], chaos_colors[i0], '', 0.2,
                       linestyle=train_test_lines[i1])

    return fig, ax


def activity_visualization(train_params):
    # a = classification_dep.ClassificationAnalysis(architecture='noisy_recurrent')
    # init_params = dict(architecture='recurrent')
    g_str = '_' + str(train_params['g_radius'])
    X_dim = train_params['X_dim']

    num_pnts_dim_red = 800
    # num_pnts_dim_red = 40000
    num_plot = 600

    accv = []
    val_accv = []
    lossvs = []
    val_lossvs = []

    train_params_loc = copy.deepcopy(train_params)

    model, params, outputs, run_dir = trainer.train(**train_params_loc)

    class_datasets = params['datasets']
    class_datasets['train'].max_samples = num_pnts_dim_red
    torch.manual_seed(train_params_loc['model_seed'])
    X, Y = class_datasets['train'][:]
    # X = X.numpy()
    # Y = Y.numpy()

    y = Y[:, -1]

    stop_epoch = params['stop_epoch']

    X = utils.extend_input(X, 31)
    hid_bef_aft = na.model_loader_utils.get_activity(model, run_dir, X, [0, -1], slice(None),
                                                     return_as_Tensor=True)
    hid_0 = hid_bef_aft[0].detach().transpose(0, 1)
    hid = hid_bef_aft[-1].detach().transpose(0, 1)

    X = X.numpy()
    hid_0 = hid_0.numpy()
    hid = hid.numpy()
    y = y.numpy()

    # out = hid_bef_aft = na.model_loader_utils.get_activity(model, run_dir, X, [0, -1], [-1], return_as_Tensor=True)
    # out_load = na.model_loader_utils.activity_loader(model, run_dir, X, 1)
    # out = out_load[-1].transpose(0,1)
    # o = out[:, 10, 0]

    # w_loader = na.model_loader_utils.weight_loader(run_dir)
    # w = w_loader[-1]
    # wo = w['out_layer.weight'].detach().numpy().astype(float)
    # wo = w['Wout']
    # bo = w['bout']

    # h = hid[:, 10]
    # o2 = h @ wo + bo
    # np.allclose(o2, o)

    # from scipy.io import savemat
    # savemat('inputs_Xdim_{}.mat'.format(X_dim), dict(X=X, labels=y, hid_0=hid_0, hid_end=hid,
    #                                                  model_state_dict=model.state_dict()))
    # h = hid_load[:3][:, :5]
    # print(h.shape)

    # %% Visualize
    coloring = plot_tools.get_color(y, cmap_activation_pnts)[:num_plot]
    edge_coloring = plot_tools.get_color(y, cmap_activation_pnts_edge)[:num_plot]
    traj_size = (.6, .35)

    # Input

    if X_dim > 2:
        X_pcs = dim_tools.get_pcs_stefan(X[:num_pnts_dim_red, 0], [0, 1, 2])
    else:
        X_pcs = dim_tools.get_pcs_stefan(X[:num_pnts_dim_red, 0], [0, 1])

    fig, ax = make_fig()
    if class_style == 'shape':
        X_pcs_plot = X_pcs[:num_plot]
        X_0 = X_pcs_plot[y[:num_plot] == 0]
        X_1 = X_pcs_plot[y[:num_plot] == 1]
        ax.scatter(X_0[:, 0], X_0[:, 1], c=coloring, edgecolors=edge_coloring, s=10, linewidths=.4, marker='o')
        ax.scatter(X_1[:, 0], X_1[:, 1], c=coloring, edgecolors=edge_coloring, s=10, linewidths=.4, marker='^')
    else:
        ax.scatter(X_pcs[:num_plot, 0], X_pcs[:num_plot, 1], c=coloring,
                   edgecolors=edge_coloring, s=10, linewidths=.4)

    # out_params = train_params_loc
    out_fig(fig, "input_pcs", train_params_loc, axis_type=1)

    # fig, ax = make_fig()
    if X_dim > 2:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        if class_style == 'shape':
            X_pcs_plot = X_pcs[:num_plot]
            X_0 = X_pcs_plot[y[:num_plot] == 0]
            X_1 = X_pcs_plot[y[:num_plot] == 1]
            ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], c=inp_col, edgecolors=inp_col_edge, s=10, linewidths=.4,
                       marker='o')
            ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], c=inp_col, edgecolors=inp_col_edge, s=10, linewidths=.4,
                       marker='^')
        else:
            ax.scatter(X_pcs[:num_plot, 0], X_pcs[:num_plot, 1], X_pcs[:num_plot, 2], c=coloring,
                       edgecolors=edge_coloring, s=10, linewidths=.4)
        ax.grid(False)
        out_fig(fig, "input_pcs_3d", train_params_loc, axis_type=0)

        fig, ax = make_fig()
        ax.scatter(X[:num_plot, 0, 0], X[:num_plot, 0, 1], c=coloring,
                   edgecolors=edge_coloring, s=10, linewidths=.4)

        out_fig(fig, "input_first2", train_params_loc, axis_type=1)
        #
        #
        # fig, ax = make_fig(figsize=traj_size)
        # fig, ax = make_fig(figsize=figsize_small)
        fig, ax = make_fig(figsize=figsize_long)
        ax.plot(hid_0[0, :, :10], linewidth=0.7)
        # ax.plot(hid_0[0, :, :], linewidth=0.7)
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([-0.02, 30.02])
        ax.set_xlabel(r"$t$", labelpad=-5)
        ax.set_ylabel(r"$h$", labelpad=-2)
        ax.xaxis.set_major_locator(plt.MultipleLocator(30))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        ax.yaxis.set_major_locator(plt.FixedLocator([-1, 1]))
        ax.yaxis.set_minor_locator(plt.FixedLocator([-1, 0, 1]))
        out_fig(fig, "hid_before_train", train_params_loc)
        #
        # plt.style.use('ggplot')
        # fig, ax = make_fig(figsize=traj_size)
        # fig, ax = make_fig(figsize=figsize_small)
        fig, ax = make_fig(figsize=figsize_long)
        # plt.rc('axes', linewidth=2.3)
        ax.plot(hid[0, :, :10], linewidth=0.7)
        # ax.plot(hid[0, :, :], linewidth=0.7)
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([-0.02, 30.02])
        ax.set_xlabel(r"$t$", labelpad=-5)
        ax.set_ylabel(r"$h$", labelpad=-2)
        ax.xaxis.set_major_locator(plt.MultipleLocator(30))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        ax.yaxis.set_major_locator(plt.FixedLocator([-1, 1]))
        ax.yaxis.set_minor_locator(plt.FixedLocator([-1, 0, 1]))
        # ax.tick_params(axis='both', which='major', width=1, length=3)
        # ax.tick_params(axis='both', which='minor', width=.7, length=2)
        out_fig(fig, "hid_after_train", train_params_loc)

        # fig, ax = make_fig(figsize=traj_size)
        # ax.plot(hid_0[0, :, :10])
        # # ax.plot(hid_0[0, :, :])
        # ax.set_ylim([-1.1, 1.1])
        # ax.set_xlim([-0.02, 30.02])
        # ax.set_xlabel(r"$t$", labelpad=-5)
        # ax.set_ylabel(r"$h$", labelpad=-2)
        # ax.xaxis.set_major_locator(plt.MultipleLocator(30))
        # ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        # out_fig(fig, "hid_before_train", g, X_dim)
        #
        # # plt.style.use('ggplot')
        # fig, ax = make_fig(figsize=traj_size)
        # # plt.rc('axes', linewidth=2.3)
        # ax.plot(hid[0, :, :10])
        # # ax.plot(hid[0, :, :])
        # ax.set_ylim([-1.1, 1.1])
        # ax.set_xlim([-0.02, 30.02])
        # ax.set_xlabel(r"$t$", labelpad=-5)
        # ax.set_ylabel(r"$h$", labelpad=-2)
        # # ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        # ax.xaxis.set_major_locator(plt.MultipleLocator(30))
        # # ax.xaxis.set_minor_locator(plt.MultipleLocator(2.5))
        # # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        # ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        # # ax.tick_params(axis='both', which='major', width=1, length=3)
        # # ax.tick_params(axis='both', which='minor', width=.7, length=2)
        # out_fig(fig, "hid_after_train", g, X_dim)

        fig, ax = make_fig(figsize=traj_size)
        # ax.plot(hid_0[0, :, :10])
        ax.hist(hid_0[:, 0, :].flatten())
        out_fig(fig, "hid_t0_hist", train_params_loc)

        # pcs = dim_tools.get_pcs_stefan(hid, [0, 1])
        pcs = np.zeros(hid.shape[:-1] + (3,))
        n_points = X.shape[1]
        p_track = 0
        y0 = Y[:, 0] == 0
        y1 = Y[:, 0] == 1
        norm = np.linalg.norm
        for i1 in range(n_points):
            pc = dim_tools.get_pcs_stefan(hid[:, i1], [0, 1, 2])
            if i1 > 0:
                # pc_old = pc.copy()
                pc_flip_x = pc.copy()
                pc_flip_x[:, 0] = -pc_flip_x[:, 0]
                pc_flip_y = pc.copy()
                pc_flip_y[:, 1] = -pc_flip_y[:, 1]
                pc_flip_both = pc.copy()
                pc_flip_both[:, 0] = -pc_flip_both[:, 0]
                pc_flip_both[:, 1] = -pc_flip_both[:, 1]

                difference0 = norm(p_track - pc)
                difference1 = norm(p_track - pc_flip_x)
                difference2 = norm(p_track - pc_flip_y)
                difference3 = norm(p_track - pc_flip_both)

                amin = np.argmin([difference0, difference1, difference2, difference3])

                if amin == 1:
                    pc[:, 0] = -pc[:, 0]
                elif amin == 2:
                    pc[:, 1] = -pc[:, 1]
                elif amin == 3:
                    pc[:, 0] = -pc[:, 0]
                    pc[:, 1] = -pc[:, 1]
            p_track = pc.copy()
            pcs[:, i1] = pc
        pcs = pcs[:num_plot]
        plt.close('all')

        # if gs[0] == 20:
        #     pcs[:,:,0] = -pcs[:,:,0]
        # p_track = pcs[0]

        # fig, ax = make_fig()  # This causes an error with the movie
        # fig, ax = plt.subplots()

        # y = y[:num_plot]

        def take_snap(i0, scat, dim=2, border=False):
            # hid_pcs_plot = pcs[y==1, i0, :dim]
            hid_pcs_plot = pcs[:, i0, :dim]
            # hid_pcs_plot = hid_pcs_plot[:, :2] - np.mean(hid_pcs_plot[:, :2], axis=0)
            xm = np.min(hid_pcs_plot[:, 0])
            xM = np.max(hid_pcs_plot[:, 0])
            ym = np.min(hid_pcs_plot[:, 1])
            yM = np.max(hid_pcs_plot[:, 1])
            xc = (xm + xM) / 2
            yc = (ym + yM) / 2
            hid_pcs_plot[:, 0] = hid_pcs_plot[:, 0] - xc
            hid_pcs_plot[:, 1] = hid_pcs_plot[:, 1] - yc
            if class_style == 'shape':
                # scat[0].set_offsets(hid_pcs_plot[y==0])
                # scat[1].set_offsets(hid_pcs_plot[y==1])
                scat[0].set_offsets(hid_pcs_plot)
            else:
                if dim == 3:
                    scat._offsets3d = juggle_axes(*hid_pcs_plot[:, :dim].T, 'z')
                else:
                    scat.set_offsets(hid_pcs_plot)
            # if dim == 3:
            #     scat[0]._offsets3d = juggle_axes(*hid_pcs_plot[:, :dim].T, 'z')
            # else:
            #     scat.set_offsets(hid_pcs_plot)

            xm = np.min(hid_pcs_plot[:, 0])
            xM = np.max(hid_pcs_plot[:, 0])
            ym = np.min(hid_pcs_plot[:, 1])
            yM = np.max(hid_pcs_plot[:, 1])
            max_extent = max(xM - xm, yM - ym)
            max_extent_arg = xM - xm > yM - ym
            if dim == 2:
                x_factor = .4
                if max_extent_arg:
                    ax.set_xlim([xm - x_factor * max_extent, xM + x_factor * max_extent])
                    ax.set_ylim([xm - .1 * max_extent, xM + .1 * max_extent])
                else:
                    ax.set_xlim([ym - x_factor * max_extent, yM + x_factor * max_extent])
                    ax.set_ylim([ym - .1 * max_extent, yM + .1 * max_extent])
            else:
                if max_extent_arg:
                    ax.set_xlim([xm - .1 * max_extent, xM + .1 * max_extent])
                    ax.set_ylim([xm - .1 * max_extent, xM + .1 * max_extent])
                    ax.set_zlim([xm - .1 * max_extent, xM + .1 * max_extent])
                else:
                    ax.set_xlim([ym - .1 * max_extent, yM + .1 * max_extent])
                    ax.set_ylim([ym - .1 * max_extent, yM + .1 * max_extent])
                    ax.set_zlim([ym - .1 * max_extent, yM + .1 * max_extent])
                # ax.set_xlim([-10, 10])
                # ax.set_ylim([-10, 10])
                # ax.set_zlim([-10, 10])

            if dim == 3:
                if border:
                    out_fig(fig, "snapshot_{}".format(i0), train_params_loc, subfolder="snaps_3d/border/", axis_type=0,
                            name_order=1)
                else:
                    out_fig(fig, "snapshot_{}".format(i0), train_params_loc, subfolder="snaps_3d/no_border/",
                            axis_type=2,
                            name_order=1)
                # out_fig(fig, "snapshot_{}".format(i0), g, X_dim, folder=folder + "snaps/no_border/", axis_type=0,
                #         name_order=1)
            else:
                if border:
                    out_fig(fig, "snapshot_{}".format(i0), train_params_loc, subfolder="snaps/border/", axis_type=0,
                            name_order=1)
                else:
                    out_fig(fig, "snapshot_{}".format(i0), train_params_loc, subfolder="snaps/no_border/", axis_type=2,
                            name_order=1)
                # out_fig(fig, "snapshot_{}".format(i0), g, X_dim, folder=folder + "snaps/no_border/", axis_type=0,
                #         name_order=1)
            return scat,

        dim = 2
        hid_pcs_plot = pcs[:num_plot, 0]
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            # ax.xaxis._axinfo['juggled'] = (0, 1, 0)
        else:
            fig, ax = make_fig()
        ax.grid(False)

        scat = ax.scatter(*hid_pcs_plot[:num_plot, :dim].T, c=coloring,
                          edgecolors=edge_coloring, s=10, linewidths=.65)

        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])
        # snap_idx = np.array([0, 1, 2, 3])
        for i0 in snap_idx:
            take_snap(i0, scat, dim=dim, border=False)
        # plt.close('all')
        # anim = animation.FuncAnimation(fig, take_snap, interval=200, frames=n_points - 1, blit=True)
        # anim = animation.FuncAnimation(fig, take_snap, frames=8, interval=200)
        # anim.save('/Users/matt/something.mp4')
        # plt.show()
        # anim.save("figs/pdf/snaps/video_{}.mp4".format(g))
        # fname = point_replace("/Users/matt/video_{}".format(g))
        # fname = point_replace("video_{}".format(g))
        # anim.save(fname + ".mp4")
        # anim.save("something.gif")
        # plt.show()
        print()


def dim_over_training(seeds, batch_sizes, num_epochs, train_params, epoch_final=None, rerun=None,
                      plot=True):
    # key = batch_sizes.keys()[0]
    train_params_loc = copy.deepcopy(train_params)
    samples_per_epoch = train_params_loc['samples_per_epoch']
    X_dim = train_params_loc['X_dim']
    # num_epochs = train_params_loc['stop_epoch']
    # epochs_per_save = [1, 199]
    num_updates = int(round(num_epochs[0] / batch_sizes[0]))
    dims = np.zeros((len(seeds), len(batch_sizes), num_updates))
    # ep = [0]
    # ep = ep + list(range(99,1999,200))
    epochs = []
    # epochs = [np.arange(20), ep]
    epochs_per_save = []
    for i1 in range(len(batch_sizes)):
        filter_step = int(round(num_epochs[i1] / num_updates))
        epochs.append(np.arange(num_epochs[i1]+1, step=filter_step))
        epochs_per_save.append(filter_step)
    epochs_per_save[1] = epochs_per_save[1]
    if rerun is not None:
        train_params_loc['rerun'] = rerun
    # train_params_loc['num_epochs'] = num_epochs
    train_params_loc['freeze_input'] = True

    num_pnts_dim_red = 200
    dims_all = []
    for i0, seed in enumerate(seeds):
        train_params_loc['model_seed'] = seed
        dims_all.append([])
        for i1, batch_size in enumerate(batch_sizes):

            train_params_loc['batch_size'] = batch_size
            train_params_loc['stop_epoch'] = num_epochs[i1]
            train_params_loc['epochs_per_save'] = epochs_per_save[i1]
            model, params, outputs, run_dir = trainer.train(**train_params_loc)

            class_datasets = params['datasets']
            # val_loss = params['history']['losses']['val']
            # val_losses[i0, i1] = val_loss
            # val_acc = params['history']['accuracies']['val']
            # val_accs[i0, i1] = val_acc

            train_samples_per_epoch = len(class_datasets['train'])
            class_datasets['train'].max_samples = num_pnts_dim_red
            torch.manual_seed(params['model_seed'])
            X = class_datasets['train'][:][0]
            if params['net_architecture'] == 'feedforward'.casefold():
                X = X[:,0]
            # num_epochs = params['history']['final_epoch']

            dims = np.zeros(len(epochs[i1]))
            for i2, epoch in enumerate(epochs[i1]):
                model1 = na.model_loader_utils.load_model_from_epoch_and_dir(model, run_dir, epoch)
                hid = model1.get_post_activations(torch.tensor(X, dtype=torch.float32))[:-1]
                hid = torch.stack(hid[:-1])
                dims[i2] = dim_tools.get_effdim(hid[-1])
            filter_step = int(round(num_epochs[i1] / num_updates))
            # dims = dims[::filter_step]
            hid = na.model_loader_utils.get_activity(model, run_dir, X, -1, -2, return_as_Tensor=True)
            dims_all[-1].append(dims)

    D_eff_responses = np.array(dims_all)
    T_ls = X.shape[1]
    medians = np.zeros((len(epochs[0]), len(batch_sizes)))
    intervals = np.zeros((len(epochs[0]), len(batch_sizes), 2))
    for i0 in range(len(epochs[0])):
        for i1 in range(len(batch_sizes)):
            # for i2 in range(T_ls):
            medians[i0, i1], intervals[i0, i1] = median_and_bound(D_eff_responses[:, i1, i0],
                                                                  perc_bound=0.75, loc=1,
                                                                  show_fit=False)
    fig, ax = make_fig(figsize_small)
    x_pnts = np.arange(len(epochs[0]))
    batch_ind = 0
    shade_plot(ax, x_pnts, medians[:, batch_ind], intervals[:, batch_ind], color=chaos_colors[
        batch_ind], label="batch_1")
    batch_ind = 1
    shade_plot(ax, x_pnts, medians[:, batch_ind], intervals[:, batch_ind], color=chaos_colors[
        batch_ind], label="batch_200")
    # train_params_loc['g_radius'] = 20
    out_fig(fig, "new_dim_through_training", train_params_loc,
            data={'D_eff_responses': D_eff_responses, 'medians': medians,
                  'intervals': intervals, 'epochs': epochs,
                  'figsize_small': figsize_small, 'chaos_colors': chaos_colors,
                  'num_epochs': num_epochs})


def dimcurves(seeds, gs, X_dim, num_epochs, samples_per_epoch, colors=chaos_colors, dim_curve_style='through_training',
              comparison='through_training'):
    """
    Figures that require instantiation of a single model to generate.
    Args:
        g ():

    Returns:

    """
    # Todo: look at margins before and after training
    # stp()
    # %% Get an example of plot_dim_vs_epoch_and_time compression down to two classes
    if not hasattr(gs, '__len__'):
        gs = [gs]
    g_str = ''
    for g in gs:
        g_str = g_str +'_'+str(g)
    g_str = g_str[1:]

    # D_eff_responses = []
    train_params_loc = copy.deepcopy(train_params)
    train_params_loc['X_dim'] = X_dim
    train_params_loc['num_epochs'] = num_epochs
    train_params_loc['samples_per_epoch'] = samples_per_epoch

    if dim_curve_style == 'before_after':
        D_eff_responses_short = np.zeros((len(seeds), len(gs), 2, 10))
        D_eff_responses = np.zeros((len(seeds), len(gs), 2, T+1))
    else:
        D_eff_responses_short = np.zeros((len(seeds), len(gs), num_epochs, 10))
        D_eff_responses = np.zeros((len(seeds), len(gs), num_epochs, T+1))
    for i0, seed in enumerate(seeds):
        for i1, g in enumerate(gs):

            train_params_loc['g_radius'] = g
            train_params_loc['model_seed'] = seed

            num_pnts_dim_red = 300

            model, params, run_dir = trainer.train(**train_params_loc)

            class_datasets = params['class_datasets']
            num_train_samples = len(class_datasets['train'])
            class_datasets['train'].max_samples = num_pnts_dim_red
            torch.manual_seed(params['model_seed'])
            X = class_datasets['train'][:][0].numpy()
            X = utils.extend_input(X, T+1)
            num_epochs = params['history']['final_epoch']

            epochs = np.arange(0, num_epochs+1, 10)

            Xt = X[:, :T+1]
            # hid_load = model_training.activity_loader(model, run_dir, torch.tensor(Xt, dtype=torch.float32), 0)
            # epochs = np.arange(len(hid_load))
            if dim_curve_style == 'before_after':
                hid = []
                model1 = model_training.load_model_from_epoch_and_dir(model, run_dir, 0)[0]
                hid1 = model1.get_post_activation(torch.tensor(Xt, dtype=torch.float32))[:-1]
                hid1 = torch.stack(hid1)
                model2 = model_training.load_model_from_epoch_and_dir(model, run_dir, -1)[0]
                hid2 = model2.get_post_activation(torch.tensor(Xt, dtype=torch.float32))[:-1]
                hid2 = torch.stack(hid2)
                hid = torch.stack((hid1, hid2))
            # if dim_curve_style == 'before_after':
            #     hid = hid_load[[0, -1]]
            # else:
            #     hid = hid_load[epochs]
            # D_eff_response = measure.dim_vs_epoch_and_time(hid, run_dir)
            D_eff_responses[i0, i1, :, :] = measure.dim_vs_epoch_and_time(hid.numpy(), run_dir)

            stop_epoch = epochs[-1]
            n_max = hid.shape[2]
            # num_trials = hid.shape[1]
            cmap = plt.cm.viridis

            if dim_curve_style == 'before_after':
                epoch_ticks_idx = np.array([0, stop_epoch])
            else:
                epoch_ticks_idx = np.arange(0, num_epochs+1, 60)
                # epoch_ticks_idx = np.array(epochs)
                # epoch_ticks_idx = np.arange(0, stop_epoch + 1, int(round((stop_epoch + 4) / 8)))
            epoch_ticks = num_train_samples*epoch_ticks_idx

            # D_eff_response_ts = []
            # for i2, t in enumerate([11, T+1]):
            #     D_eff_input = dim_tools.get_effdim(Xt[:, 0, :])
            #     D_eff_response_t = D_eff_response[:, :t]
            #     D_eff_response_ts.append(D_eff_response_t)
            # # D_eff_responses.append(D_eff_response_ts)
            #
            # D_eff_responses_short[i0, i1, :, :] = D_eff_response_ts[0]
            # D_eff_responses_long[i0, i1, :, :] = D_eff_response_ts[1]

    T_ls = T+1
    medians = np.zeros((len(gs), 2, T_ls))
    intervals = np.zeros((len(gs), 2, T_ls, 2))
    for i0 in range(len(gs)):
        for i1 in range(2):
            for i2 in range(T_ls):  # before and after training
                medians[i0, i1, i2], intervals[i0, i1, i2] = measure.median_and_bound(D_eff_responses[:, i0, i1, i2],
                                                                                      perc_bound=0.75, loc=1,
                                                                                      show_fit=False)
    mean_Ds = np.mean(D_eff_responses, axis=0)
    std_Ds = np.std(D_eff_responses, axis=0)
    individual_realizations = False
    # for t in [11, T+1]:
    for t in [T+1]:
        if dim_curve_style == 'before_after':
            # fig, ax = make_fig()
            # for i0, g in enumerate(gs):
            #     ax.plot(D_eff_responses[i0][1][0][:t], color=chaos_colors[i0], label=r"$g={}$".format(g))
            # ax.set_ylabel(r"$D_{PR}$")
            # ax.set_title("Dimensionality before training")
            # ax.set_xlim([0, t-1])
            # # ax.set_ylim([0, 41])
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            # ax.set_xlabel(r'$t$')
            # if X_dim == N:
            #     ax.set_yticks(np.array([0, 10, 20, 30, 40]))
            # out_fig(fig, "dimcurves_{}_before".format(t), '', X_dim)
            # fig, ax = make_fig()
            # # for i0, g in enumerate(gs):
            # #     ax.plot(D_eff_responses[i0][1][-1], color=cmap_cycle(i0), label=r"$g={}$".format(g))
            # ax.plot(D_eff_responses[0][1][-1][:t], color=chaos_colors[0], label=r"$g={}$".format(gs[0]))
            # ax.plot(D_eff_responses[1][1][-1][:t], color=chaos_colors[1], label=r"$g={}$".format(gs[1]))
            # ax.set_ylabel(r"$D_{PR}$")
            # ax.set_title("Dimensionality after training")
            # ax.set_xlim([0, t - 1])
            # # ax.set_ylim([0, 41])
            # ax.set_xlabel(r'$t$')
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # if X_dim == N:
            #     ax.set_yticks(np.array([0, 10, 20, 30, 40]))
            # out_fig(fig, "dimcurves_{}_after".format(t), '', X_dim)
            fig, ax = make_fig(figsize_small)
            # for i0, g in enumerate(gs):
            # ax.plot(D_eff_responses[i0][1][0], '--', color=cmap_cycle(i0), label=r"$g={}$".format(g))
            # ax.plot(D_eff_responses[i0][1][-1], color=cmap_cycle(i0), label=r"$g={}$".format(g))
            x_pnts = np.arange(len(mean_Ds[0, 1, :t]))
            # ax.plot(x_pnts, mean_Ds[0, 0, :t], '--', color=colors[0], label=r"before training")
            chaos_ind = 0
            train_ind = 0
            shade_plot(ax, x_pnts, medians[chaos_ind, train_ind], intervals[chaos_ind, train_ind], color=colors[
                chaos_ind], label="before training", linestyle='--')
            if individual_realizations:
                for i1 in range(len(seeds)):
                    ax.plot(x_pnts, D_eff_responses[i1, 0, 0, :t], '--', color=colors[0], alpha=0.2,
                            label=r"before training")
                    ax.plot(x_pnts, D_eff_responses[i1, 1, 0, :t], color=colors[0], alpha=0.2,
                            label=r"after training")
                    if len(gs) > 1:
                        ax.plot(x_pnts, D_eff_responses[i1, 0, 1, :t], '--', color=colors[1], alpha=0.2,
                                label=r"before training")
                        ax.plot(x_pnts, D_eff_responses[i1, 1, 1, :t], color=colors[1], alpha=0.2,
                                label=r"after training")

            chaos_ind = 0
            train_ind = 1
            shade_plot(ax, x_pnts, medians[chaos_ind, train_ind], intervals[chaos_ind, train_ind], color=colors[
                chaos_ind], label="after training")

            if len(gs) > 1:
                chaos_ind = 1
                train_ind = 0
                shade_plot(ax, x_pnts, medians[chaos_ind, train_ind], intervals[chaos_ind, train_ind], color=colors[
                    chaos_ind], label="before training", linestyle='--')
                chaos_ind = 1
                train_ind = 1
                shade_plot(ax, x_pnts, medians[chaos_ind, train_ind], intervals[chaos_ind, train_ind], color=colors[
                    chaos_ind], label="after training")
            # ax.set_ylabel(r"$D_{PR}$")
            ax.set_ylabel("ED")
            ax.set_xlim([0, t-1])
            ax.set_ylim([0, None])
            if X_dim == N:
                ax.set_yticks(np.array([0, 10, 20, 30, 40]))
                # ax.legend(loc='upper right')
            elif X_dim == 2:
                ax.set_yticks(np.array([0, 5, 10, 15]))
                # ax.legend(loc='upper left')
            # ax.set_ylim([0, 41])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            # leg = ax.get_legend()
            # leg.legendHandles[0].set_color('black')
            # leg.legendHandles[1].set_color('black')
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax.set_xlabel(r'$t$')
            # ax.set_title("Dimensionality through time")
            # ax.set_title("Dimensionality")
            out_fig(fig, "dimcurves_{}_both".format(t), g_str, X_dim)

    plt.close('all')


if __name__ == '__main__':
    train_params = dict(N=200,
                        stop_epoch=20,
                        samples_per_epoch=1250,
                        X_clusters=60,
                        X_dim=200,
                        Y_classes=2,
                        n_lag=11,
                        # n_lag=4,
                        g_radius=1,
                        clust_sig=.02,
                        input_scale=10.0,
                        perc_val=0.2,
                        input_style='hypercube',
                        model_seed=1,
                        n_hold=1,
                        n_out=1,
                        # loss='mse',
                        loss='cce',
                        optimizer='rmsprop',
                        dt=.01,
                        # optimizer='sgd',
                        # learning_rate=.001,
                        # learning_rate=.001,
                        # learning_rate=.008,
                        # learning_rate=.0001,
                        learning_rate=.05,
                        batch_size=10,
                        freeze_input=False,
                        # net_architecture='vanilla_rnn',
                        net_architecture='sompolinsky',
                        # net_architecture='feedforward',
                        Win=Win,  # todo: refactor code so this can be defined here.
                        Wrec_rand_proportion = .2,
                        patience_before_stopping=6000,
                        # hid_nonlin='linear',
                        hid_nonlin='tanh',
                        epochs_per_save=1,
                        rerun=False)

    # train_params['samples_per_epoch'] = 800
    batch_size = int(round(train_params['samples_per_epoch']*(1-train_params['perc_val'])))
    # dim_over_training([0], [1, batch_size], [6, 6 * train_params['samples_per_epoch']], train_params)
    # dim_over_training([0, 1], [1, batch_size], [40, 40], train_params)
    # dim_over_training([0,1], [10, 200], [20, 20*train_params['samples_per_epoch']], train_params)
    activity_visualization(train_params)
