from sphinx_gallery.scrapers import figure_rst

import shutil
from glob import glob
import os


def image_scraper(block, block_vars, gallery_conf):
    """Scrape image files that are already present in current folder."""
    path_example = os.path.dirname(block_vars['src_file'])

    images = set()
    for ext in ('jpg', 'jpeg', 'png', 'gif'):
        images = images.union(
            set(glob(os.path.join(path_example, '*.' + ext)))
        )

    # Iterate through files, copy them to the SG output directory The
    # image file names are provided by the `image_path_iterator`, and
    # are of the form "sphx_glr_plot_examplename_001.gif".

    image_path_iterator = block_vars['image_path_iterator']
    sphinx_paths = {image: (next(image_path_iterator)) for image in images}
    for (source, target) in sphinx_paths.items():
        shutil.copyfile(source, target)

    if len(images) == 0:
        return ''
    else:
        return figure_rst(sphinx_paths.values(), gallery_conf['src_dir'])
