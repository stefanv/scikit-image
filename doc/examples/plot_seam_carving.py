"""
=================================
Seam Carving using Shortest Paths
=================================

Seam carving is a content-sensitive image rescaling method.  Usually,
when an image is rescaled, objects get squashed.  With seam carving,
we find paths through the image gradient that appear to convey less
information according to some error metric, and then remove those.

Imagine, for example, a scene with two people, some distance apart.
Typically, we cannot do a simple scaling--the bodies will get
squashed.  Instead, we can move them closer together, ignoring some of
the background.

The technique is described in more detail on `Wikipedia
<http://en.wikipedia.org/wiki/Seam_carving>`__, and illustrated in
`this video <http://www.youtube.com/watch?v=6NcIJXTlugc>`__.

"""

from scikits.image import img_as_float
from scikits.image import data, filter, graph

import numpy as np

import matplotlib.pyplot as plt

img = img_as_float(data.camera())
img_src = img.copy()

row_idx = np.arange(img.shape[0])

for i in range(100):
    print 'Image shape:', img.shape

    col_idx = np.arange(img.shape[1])
    p, cost = graph.shortest_path(filter.sobel(img), axis=0)

    for r in row_idx:
        img[r, :-1] = img[r, np.setdiff1d(col_idx, p[r])]

    img = img[:, :-1]

plt.subplot(121)
plt.imshow(img_src, cmap=plt.cm.gray)
plt.title('Input image (%s x %s)' % img_src.shape)

plt.subplot(122)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Shrunk image (%s x %s)' % img.shape)
plt.show()

