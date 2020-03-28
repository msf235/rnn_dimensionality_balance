import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import axes3d, Axes3D


def plot(X, coloring=None, mark_ends=True, ax=None, coloring_ends=None, line_kwargs={}, end_kwargs={}):
    """
    Plot many curves together, coloring each curve according to parameter 'coloring'.
    Args:
        X ((num_curves, T, dim)-array): Input data to be plotted. num_curves is the number of curves that will be
        plotted. Can also be a list of arrays, where the list has length num_curves
        coloring ((num_curves, 4)-array, pyplot.cmap): Either
            rgba color values for each of the curves OR
            a pyplot.cmap object

    Returns:

    """
    # plt.style.use(rd + 'analyze_tasks/ggplot_thick.mplstyle')

    if coloring_ends is None:
        coloring_ends = coloring
    X = np.asarray(X)
    dim = X.shape[-1]

    # colors, rinds = np.unique(coloring, axis=0, return_inverse=True)
    if ax is None:
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif dim == 2:
            fig, ax = plt.subplots()
        else:
            print("Cutting dimensions of X down to 2.")
            fig, ax = plt.subplots()
            dim = 2
    else:
        fig = None
    if X.ndim > 2:
        for i1 in range(X.shape[0]):
            if isinstance(coloring, matplotlib.colors.LinearSegmentedColormap):
                color = coloring(i1 / max(X.shape[0] - 1, 1))
            elif isinstance(coloring, str) or len(coloring) == 3 or len(coloring) == 4:
                color = coloring
            else:
                color = coloring[i1]
            if isinstance(coloring_ends, matplotlib.colors.LinearSegmentedColormap):
                color_ends = coloring_ends(i1 / max(X.shape[0] - 1, 1))
            elif isinstance(coloring, str) or len(coloring) == 3 or len(coloring) == 4:
                color_ends = coloring_ends
            else:
                color_ends = coloring_ends[i1]
            ax.plot(*X[i1, :, :dim].T, color=color, zorder=1, **line_kwargs)
            if mark_ends:
                # if 'color' not in end_kwargs:
                #     end_kwargs['color'] = color
                # ax.scatter(*X[i1, 0, :dim].T, marker='o', s=30, zorder=4, color=color_ends, **end_kwargs)
                ax.scatter(*X[i1, 0, :dim].T, marker='o', s=30, zorder=4, facecolors=color, edgecolors=color_ends,
                           **end_kwargs)
                ax.scatter(*X[i1, -1, :dim].T, marker='s', s=30, facecolors='none', zorder=4, color=color_ends,
                           **end_kwargs)
    else:
        if isinstance(coloring, matplotlib.colors.LinearSegmentedColormap):
            color = coloring(0)
        else:
            color = np.squeeze(coloring)
        ax.plot(*X[:, :2], color=color, zorder=1, **line_kwargs)
        if mark_ends:
            ax.scatter(*X[0, :2].T, marker='.', zorder=4, color=color)
            ax.scatter(*X[-1, :2].T, zorder=4, color=color)

    return fig, ax

def plot_snapshots(X, coloring, ax_rect=(0, 0, 1, 1), **kwargs):
    n_points = X.shape[1]
    dim = X.shape[-1]
    figs = []
    plt.ioff()

    for i1 in range(n_points):
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif dim == 2:
            fig = plt.figure()
            ax = fig.add_axes(ax_rect)
        else:
            print("Cutting dimensions of X down to 2.")
            fig = plt.figure()
            ax = fig.add_axes(ax_rect)
            dim = 2
        ax.scatter(*X[:, i1, :dim].T, color=coloring, **kwargs)
        if i1 == 0 and dim == 2:
            fig_inset = plt.figure()
            ax_main = fig_inset.add_axes(ax_rect)
            ax_inset = fig_inset.add_axes([.6, .6, .3, .3])
            ax_inset.spines['right'].set_visible(True)
            ax_inset.spines['top'].set_visible(True)
            ax_main.scatter(*X[:, i1, :dim].T, color=coloring, **kwargs)
            ax_inset.scatter(*X[:, i1, :dim].T, color=coloring, **kwargs)
            ax_inset.set_xlim([-1.5, 1])
            ax_inset.set_ylim([-1.5, 1.2])
            figs.append(fig_inset)

        figs.append(fig)

    return figs

class InitializePlots:
    def __init__(self, plot_style='subplots', figsize=None, rows=2, columns=3, return_names=False):
        self.figs = []
        self.axs = []
        self.names = []
        self.figsize = figsize
        self.plot_style = plot_style
        self.fig_cnt = 1
        self.rows = rows
        self.columns = columns
        if plot_style == 'subplots':
            if self.figsize is None:
                self.fig = plt.figure()
            else:
                self.fig = plt.figure(figsize=self.figsize)

            self.figs.append(self.fig)

    def get_plots(self, projection='2d', name=None):
        if self.plot_style == 'subplots':
            if isinstance(projection, str) and projection == '3d':
                ax = self.fig.add_subplot(self.rows, self.columns, self.fig_cnt, projection='3d')
            else:
                ax = self.fig.add_subplot(self.rows, self.columns, self.fig_cnt)
            self.fig_cnt = self.fig_cnt + 1
            self.axs.append(ax)
            self.names.append(name)
            return ax, self.fig

        elif self.plot_style == 'individual':
            if self.figsize is None:
                fig = plt.figure()
            else:
                fig = plt.figure(figsize=self.figsize)
            if isinstance(projection, str) and projection == '3d':
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
            self.figs.append(fig)
            self.axs.append(ax)
            self.names.append(name)
            return ax, fig
        else:
            raise ValueError("self.plot_style not recognized.")

def plot_vectors(V_heads, ax, reflect, V_tails=None, *plt_args, **plt_kwargs):
    """

    Args:
        V (): array where rows are the vectors to be plotted
        reflect (bool): whether or not to plot the reflection of the vectors in V
        ax ():

    Returns:

    """
    V_heads = np.array(V_heads)
    if len(V_heads.shape) == 1:
        V_heads = V_heads[None, :]
    # V = np.array([[1, 1], [-2, 2], [4, -7]])
    origin = np.zeros(2)
    is_scalar = not hasattr(V_tails, '__len__')
    if V_tails is None or (is_scalar and V_tails == 0):
        V_tails = 0 * V_heads
    V_tails = np.array(V_tails)
    if len(V_tails.shape) == 1:
        V_tails = V_tails[None, :]

    for i0 in range(V_heads.shape[0]):
        v_head = V_heads[i0]
        v_tail = V_tails[i0]
        if reflect:
            M = np.vstack((-v_head + v_tail, v_head + v_tail))
        else:
            M = np.vstack((origin + v_tail, v_head + v_tail))
        lines = ax.plot(M[:, 0], M[:, 1], *plt_args, **plt_kwargs)

    return lines

    # fig, ax = plt.subplots()
    # # ax.quiver(*origin, V[:, 0], V[:, 1], '--', color=['r', 'b', 'g'], units='width', scale=scale, *plt_args,
    # #           **plt_kwargs)
    # ax.quiver(*origin, V[:, 0], V[:, 1], linestyle='dashed', color=['r', 'b', 'g'], units='width', scale=scale)
    # plt.quiver
    # if reflect:
    #     ax.quiver(*origin, -V[:, 0], -V[:, 1], color=['r', 'b', 'g'])

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    if isinstance(seq[0], str):
        c = mcolors.ColorConverter().to_rgb
        seq = [c(color_str) for color_str in seq]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def get_color(hidden, cmap=plt.cm.plasma):
    mag = np.max(hidden) - np.min(hidden)
    # hid_norm = (hidden - np.min(hidden)) / (mag - 1)
    hid_norm = (hidden - np.min(hidden)) / mag

    return cmap(hid_norm)

def set_size(w, h, left_boundary=None, top_boundary=None, right_boundary=None, bottom_boundary=None, ax=None):
    """ w, h: width, height in inches """
    if ax is None: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)

def set_axes_buffer_inches(left=None, bottom=None, right=None, top=None, ax=None, modify_figsize=False):
    """Repositions axes to have the specified buffers. The figure should not change dimensions. If None, the original
    buffer is maintained."""
    if ax is None: ax = plt.gca()

    original_pos = ax.get_position().bounds  # in figure coordinates
    l = ax.figure.subplotpars.left
    b = ax.figure.subplotpars.bottom
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top

    fig_size = ax.figure.get_size_inches()

    if left is None:
        left_perc = original_pos[0]
        # left_perc = l
    else:
        left_perc = left / fig_size[0]
    if bottom is None:
        bottom_perc = b
    else:
        bottom_perc = original_pos[1] / fig_size[1]
        # bottom_perc = bottom / fig_size[1]
    if right is None:
        right_pos_perc_0 = original_pos[0] + original_pos[2]
        # right_pos_perc_0 =
        width_perc = right_pos_perc_0 - left_perc
    else:
        right_perc = right / fig_size[0]
        right_pos_perc = 1 - right_perc
        width_perc = right_pos_perc - left_perc
    if top is None:
        top_pos_perc_0 = original_pos[1] + original_pos[3]
        height_perc = top_pos_perc_0 - bottom_perc
    else:
        top_perc = top / fig_size[0]
        top_pos_perc = 1 - top_perc
        height_perc = top_pos_perc - bottom_perc

    # left_perc = left / fig_size[0]

    # ax.set_position([a, 0.1, new_width_perc, .8])
    # if modify_figsize:
    #     l = ax.figure.subplotpars.left
    #     r = ax.figure.subplotpars.right
    #     t = ax.figure.subplotpars.top
    #     b = ax.figure.subplotpars.bottom
    #     orig_right

    ax.set_position([left_perc, bottom_perc, width_perc, height_perc])

    # ax.figure.subplots_adjust(left=new_pos_left)
    # ax.figure.subplots_adjust(left=.4)
    # ax.set_position(left=new_pos_left)

    print

# def get_color_labels(self, pyplot_cmap):
#     # cmap = plt.cm.plasma
#     Y = np.argmax(self.params.Y[:, -1], axis=1)
#     Y_classes = self.params.Y_classes
#     cs = np.zeros((len(Y), 4))
#     for i1 in range(Y_classes):
#         cs[Y == i1] = pyplot_cmap(i1 / Y_classes)
#     return cs

if __name__ == '__main__':
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(range(5))
    # set_axes_buffer_inches(left=.4)
    print(ax.get_position().bounds)
    el = .5
    # rl = .5
    rl = 0.02
    fig_size = ax.figure.get_size_inches()
    a = el / fig_size[0]
    rl_perc = rl / fig_size[0]
    new_width_perc = 1 - rl_perc - a
    # ax.set_position([a, 0.1, new_width_perc, .8])
    # ax.figure.tight_layout(rect=[0, 0, 1, 1])

    # fig.subplots_adjust(left=.2, right=.4)
    # fig.tight_layout(rect=[.6, .2, .8, .8])
    # fig.savefig("test1.png")
    fig.savefig("test1.png")
    set_axes_buffer_inches(ax=ax)
    fig.savefig("test2.png")
    # set_axes_buffer_inches(left=2, ax=ax)
    set_axes_buffer_inches(left=2, right=2, top=1, bottom=1, ax=ax)
    fig.savefig("test3.png")
    # plt.show()
    print

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # f = np.random.random(100)
    # g = np.random.random(100)
    # fig = plt.figure()
    # fig.suptitle('Long Suptitle', fontsize=24)
    # plt.subplot(121)
    # plt.plot(f)
    # plt.title('Very Long Title 1', fontsize=20)
    # plt.subplot(122)
    # plt.plot(g)
    # plt.title('Very Long Title 2', fontsize=20)
    # plt.subplots_adjust(top=0.65)
    # plt.show()

def pos_to_extent(pos):
    extent = [0, 0, 0, 0]
    extent[0] = pos[0]
    extent[1] = pos[1]
    extent[2] = pos[2] - pos[0]
    extent[3] = pos[3] - pos[1]
    return extent

def extent_to_pos(ext):
    pos = [0, 0, 0, 0]
    pos[0] = ext[0]
    pos[1] = ext[1]
    pos[2] = ext[2] + ext[0]
    pos[3] = ext[3] + ext[1]
    return pos

def add_cbar(pnts, pos_cbar=(.93, 0, .98, 1), canv_r_pos=.9, fig=None, **kwargs):
    """

    Args:
        ax ():
        pos (list): (x0, y0, x1, y1)
        pos_canv: (x1, y1)

    Returns:

    """
    if fig is None: fig = plt.gcf()
    ax = fig.axes[0]
    ext_old = ax.get_position().bounds
    pos_old = extent_to_pos(ext_old)
    pos_canv_full = [0, 0, 0, 0]
    pos_canv_full[0] = pos_old[0]
    pos_canv_full[1] = pos_old[1]
    pos_canv_full[2] = canv_r_pos
    pos_canv_full[3] = pos_old[3]
    ext_cbar = pos_to_extent(pos_cbar)
    ext_canv = pos_to_extent(pos_canv_full)
    # fig = ax.figure

    scale_x = ext_canv[2] / ext_old[2]
    scale_y = ext_canv[3] / ext_old[3]
    figsize = fig.get_size_inches()
    figsize_new = [0, 0]
    figsize_new[0] = figsize[0] / scale_x
    figsize_new[1] = figsize[1] / scale_y
    fig.set_size_inches(figsize_new)

    ax.set_position(ext_canv)
    cax = fig.add_axes(ext_cbar)
    cbar = fig.colorbar(pnts, cax=cax, **kwargs)
    return cbar

def append_cbar(pnts, pos_cbar=(1.03, 0, 1.08, 1), fig=None, **kwargs):
    """

    Args:
        ax ():
        pos (list): (x0, y0, x1, y1)
        pos_canv: (x1, y1)

    Returns:

    """
    if fig is None: fig = fig.gcf()
    ext_cbar = pos_to_extent(pos_cbar)
    cax = fig.add_axes(ext_cbar)
    cbar = fig.colorbar(pnts, cax=cax, **kwargs)
    return cbar




