import numpy as np
from matplotlib import pyplot as plt
from analysis_tools import plot_tools as pt


x = np.linspace(0, 10)
y = x

fig = plt.figure(figsize=(6,5))
ax = fig.add_axes([0, 0, 1, 1])
# pt.set_size(6,5)
ax.plot(x,y)
fig.savefig("test1.png")
fig.show()

fig = plt.figure(figsize=(6,5))
ext = [0, 0, 1, 1]
ax = fig.add_axes(ext)

# cax = fig.add_axes([.88, 0, .05, 1])
# pt.set_size(6,5)
M = np.random.randn(20,5)
pnts = ax.imshow(M, aspect='auto')
# fig.savefig("test2_a.png")
fig.savefig("test2_a.pdf", bbox_inches="tight", transparent=True)
# fig.savefig("test2_a.png", transparent=True)
# pt.add_cbar(pnts, ax=ax, canv_r_pos=ext[2]+ext[0])
pt.add_cbar(pnts, fig=fig)
ax.set_title("Test2")
ax.set_xlabel("xlabel")
ax.set_ylabel("ylabel")
fig.savefig("test2_b.pdf", bbox_inches="tight", transparent=True)
# fig.savefig("test2_b.png", transparent=True)
fig.show()

fig = plt.figure(figsize=(6,5))
ext = [0, 0, 1, 1]
ax = fig.add_axes(ext)
pnts = ax.imshow(M, aspect='auto')
fig.colorbar(pnts, use_gridspec=False)
fig.savefig("test2_c.png", bbox_inches="tight", transparent=True)

fig = plt.figure(figsize=(6,5))
ext = [0, 0, 1, 1]
ax = fig.add_axes(ext)
pnts = ax.imshow(M, aspect='auto')
pt.append_cbar(pnts, fig=fig)
# fig.colorbar(pnts, use_gridspec=False)
fig.savefig("test2_d.png", bbox_inches="tight", transparent=True)


fig = plt.figure(figsize=(6/.9,5))
ax = fig.add_axes([.1, 0, .9, 1])
# pt.set_size(6,5)
ax.plot(x,y)
fig.savefig("test3.png")
# fig.show()





# %%
#
# import matplotlib.pyplot as plt
# from urllib.request import urlopen
#
# # load the image
# img = plt.imread(urlopen('http://upload.wikimedia.org/wikipedia/en/thumb/5/56/Matplotlib_logo.svg/500px-Matplotlib_logo.svg.png'))
#
# # get the dimensions
# ypixels, xpixels, bands = img.shape
#
# # get the size in inches
# dpi = 72.
# xinch = xpixels / dpi
# yinch = ypixels / dpi
#
# # plot and save in the same size as the original
# fig = plt.figure(figsize=(xinch,yinch))
#
# ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[],yticks=[])
# ax.imshow(img, interpolation='none')
#
# plt.savefig('mpl_logo.pdf', dpi=dpi, transparent=True)
#
#
# fig = plt.figure(figsize=(xinch,yinch/.8))
#
# ax = plt.axes([0., 0., 1., .8], frameon=False, xticks=[],yticks=[])
# ax.imshow(img, interpolation='none')
# ax.set_title('Matplotlib is fun!', size=16, weight='bold')
#
# plt.savefig('mpl_logo_title.pdf', dpi=dpi)