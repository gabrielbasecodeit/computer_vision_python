import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from skimage import color
from skimage import io

img=color.rgb2grey(io.imread(r'C:\Users\Gabriel\Documents\Projects\Basecodeit\norweign_carpets\images\positives\training\\2.gif'))

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, img ,rstride=10, cstride=10, cmap=plt.gray(), linewidth=0)

# show it
plt.show()