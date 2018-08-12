from scipy import ndimage, misc
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import numpy as np

fig = plt.figure()

plt.gray()  # show the filtered result in grayscale

ax1 = fig.add_subplot(221)  # left side
ax2 = fig.add_subplot(222)  # right side
ax3 = fig.add_subplot(223) 

img=color.rgb2grey(io.imread(r'C:\Users\Gabriel\Documents\Projects\Basecodeit\machine_learning_playground\images\\pipo.jpeg'))

kernel = np.array([[1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1]])

# First a 1-D  Gaussian
t = np.linspace(-10, 10, 30)
bump = np.exp(-0.1*t**2)
bump /= np.trapz(bump) # normalize the integral to 1

# make a 2-D kernel out of it
gaussianKernel = bump[:, np.newaxis] * bump[np.newaxis, :]

result = ndimage.convolve(img, kernel)
result2 = ndimage.convolve(img, gaussianKernel)

ax1.imshow(img)
ax2.imshow(result)
ax3.imshow(result2)

plt.show()