"""
======================
Swirling Checkersboard
======================

An animation of slowly increasing both the and radius
of the swirl.
"""

from skimage import data, transform, color, util
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from itertools import chain

FRAMES_PER_SECOND = 50
FILE_NAME = 'sphx_glr_plot_swirling_001.gif'

img = data.checkerboard()
frames = (transform.swirl(img, strength=i / 50, radius=100 + i // 20)
          for i in chain(range(0, 1000, 5), range(1000, 0, -5)))
animation = ImageSequenceClip(
    [util.img_as_ubyte(color.gray2rgb(f)) for f in frames],
    fps=FRAMES_PER_SECOND
)
animation.to_gif(FILE_NAME)
