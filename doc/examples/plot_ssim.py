'''
===========================
Structural similarity index
===========================

When comparing images, the mean squared error (MSE)--while simple to
implement--is not highly indicative of perceived similarity.  Structural
similarity aims to address this shortcoming by taking texture into account
[1]_, [2]_.

The example shows two modifications of the input image, each with the same MSE,
but with very different mean structural similarity indices.

.. [1] Zhou Wang; Bovik, A.C.; ,"Mean squared error: Love it or leave it? A new
       look at Signal Fidelity Measures," Signal Processing Magazine, IEEE,
       vol. 26, no. 1, pp. 98-117, Jan. 2009.

.. [2] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality
       assessment: From error visibility to structural similarity," IEEE
       Transactions on Image Processing, vol. 13, no. 4, pp. 600-612,
       Apr. 2004.

'''

from skimage import data, img_as_float
from skimage.measure import structural_similarity as ssim
from skimage.filter import median_filter

import numpy as np

img = img_as_float(data.camera())
img = img[::2, ::2] # shrink image for speed
rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1

def mse(x, y):
    return np.linalg.norm(x - y)

img_noise = img + noise
img_const = img + abs(noise)
img_median = median_filter(img, 5)

import matplotlib.pyplot as plt

f, axes = plt.subplots(2, 2)
axes = axes.ravel()

label = 'MSE: %2.f, SSIM: %.2f'

for ax, img_test in zip(axes, (img, img_noise, img_const, img_median)):
    mse_value = mse(img, img_test)
    ssim_value = ssim(img, img_test, dynamic_range='image2')
    ax.imshow(img_test, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.set_xlabel(label % (mse_value, ssim_value))

axes[0].set_title('Original image')
axes[1].set_title('Image with noise')
axes[2].set_title('Image plus constant')
axes[3].set_title('Median filtered')

# Remove spines and tick labels but not axis labels
for ax in axes:
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    for spine in ax.spines.itervalues():
        spine.set_visible(False)

plt.tight_layout()
plt.show()

