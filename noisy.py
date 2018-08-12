import numpy as np
from skimage import color
from skimage import io
import matplotlib.pyplot as plt

def noisy(image):
    row,col = image.shape #
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

img=color.rgb2grey(io.imread(r'C:\Users\Gabriel\Documents\Projects\Basecodeit\machine_learning_playground\images\\pipo.jpeg'))

fig=plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(img,cmap="gray")

ax2 = fig.add_subplot(122)
ax2.imshow(noisy(img),cmap="gray")

plt.show()