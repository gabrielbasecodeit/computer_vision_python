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
ax4 = fig.add_subplot(224) 

img_noisy=color.rgb2grey(io.imread(r'C:\Users\Gabriel\Documents\Projects\Basecodeit\machine_learning_playground\images\\coins_noisy.gif'))

kernel = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])

result = ndimage.convolve(img_noisy, kernel)
result2= ndimage.median_filter(img_noisy, size=(3,3))
result3 = ndimage.median_filter(result, size=(3,3))

ax1.imshow(img_noisy)
ax2.imshow(result)
ax3.imshow(result2)
ax4.imshow(result3)

plt.show()