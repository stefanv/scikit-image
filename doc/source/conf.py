# -*- coding: utf-8 -*-
#
# skimage documentation build configuration file, created by
# sphinx-quickstart on Sat Aug 22 13:00:30 2009.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import skimage
from sphinx_gallery.sorting import ExplicitOrder

import shutil
import time
from glob import glob


class ImageFileScraper(object):
    def __init__(self):
        """Scrape image files that are already present in current folder."""
        self.embedded_images = {}
        self.start_time = time.time()
    def __call__(self, block, block_vars, gallery_conf):
        # Find all image files in the current directory.
        path_example = os.path.dirname(block_vars['src_file'])
        image_files = _find_images(path_example)
        # Iterate through files, copy them to the SG output directory
        image_names = []
        image_path_iterator = block_vars['image_path_iterator']
        for path_orig in image_files:
            print('a')
            # If we already know about this image and it hasn't been modified
            # since starting, then skip it
            mod_time = os.stat(path_orig).st_mtime
            already_embedded = (path_orig in self.embedded_images and
                                mod_time <= self.embedded_images[path_orig])
            existed_before_build = mod_time <= self.start_time
            if already_embedded or existed_before_build:
                continue
            # Else, we assume the image has just been modified and is displayed
            path_new = next(image_path_iterator)
            self.embedded_images[path_orig] = mod_time
            image_names.append(path_new)
            shutil.copyfile(path_orig, path_new)
        if len(image_names) == 0:
            return ''
        else:
            return figure_rst(image_names, gallery_conf['src_dir'])
    def _find_images(path, image_extensions=['jpg', 'jpeg', 'png', 'gif']):
        """Find all unique image paths for a set of extensions."""
        image_files = set()
        for ext in image_extensions:
            this_ext_files = set(glob(os.path.join(path, '*.'+ext)))
            image_files = image_files.union(this_ext_files)
        return image_files

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.append(os.path.abspath('.'))

curpath = os.path.dirname(__file__)
sys.path.append(os.path.join(curpath, '..', 'ext'))

# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx_copybutton',
              'sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'numpydoc',
              'doi_role',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx.ext.linkcode',
              'sphinx_gallery.gen_gallery'
              ]

autosummary_generate = True

#------------------------------------------------------------------------
# Sphinx-gallery configuration
#------------------------------------------------------------------------

sphinx_gallery_conf = {
    'doc_module': ('skimage',),
    # path to your examples scripts
    'examples_dirs': '../examples',
    # path where to save gallery generated examples
    'gallery_dirs': 'auto_examples',
    'backreferences_dir': 'api',
<<<<<<< HEAD
    'reference_url': {'skimage': None},
    'subsection_order': ExplicitOrder([
        '../examples/data',
        '../examples/numpy_operations',
        '../examples/color_exposure',
        '../examples/edges',
        '../examples/transform',
        '../examples/filters',
        '../examples/features_detection',
        '../examples/segmentation',
        '../examples/applications',
        '../examples/developers',
    ]),
}
=======
    'reference_url'     : {
            'skimage': None,},
    'image_scrapers' : ('matplotlib', ImageFileScraper())
    }
>>>>>>> getting ready for config check

# Determine if the matplotlib has a recent enough version of the
# plot_directive, otherwise use the local fork.
try:
    from matplotlib.sphinxext import plot_directive
except ImportError:
    use_matplotlib_plot_directive = False
else:
    try:
        use_matplotlib_plot_directive = (plot_directive.__version__ >= 2)
    except AttributeError:
        use_matplotlib_plot_directive = False

if use_matplotlib_plot_directive:
    extensions.append('matplotlib.sphinxext.plot_directive')
else:
    extensions.append('plot_directive')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'skimage'
copyright = '2013, the scikit-image team'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.

with open('../../skimage/__init__.py') as f:
    setup_lines = f.readlines()
version = 'vUndefined'
for l in setup_lines:
    if l.startswith('__version__'):
        version = l.split("'")[1]
        break

# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = []

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = 'scikit-image'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['themes']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'skimage v%s docs' % version

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
   '**': ['searchbox.html',
          'navigation.html',
          'localtoc.html',
          'versions.html'],
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
#html_use_modindex = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'scikitimagedoc'


# -- Options for LaTeX output --------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'scikit-image.tex', u'The scikit-image Documentation',
   u'scikit-image development team', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
latex_elements = {}
latex_elements['preamble'] = r'''
\usepackage{enumitem}
\setlistdepth{100}

\usepackage{amsmath}
\DeclareUnicodeCharacter{00A0}{\nobreakspace}

% In the parameters section, place a newline after the Parameters header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}

% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother

'''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_domain_indices = False

# -----------------------------------------------------------------------------
# Numpy extensions
# -----------------------------------------------------------------------------
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plot_basedir = os.path.join(curpath, "plots")
plot_pre_code = """
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

import matplotlib
matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 10,
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
})

"""
plot_include_source = True
plot_formats = [('png', 100), ('pdf', 100)]

plot2rst_index_name = 'README'
plot2rst_rcparams = {'image.cmap' : 'gray',
                     'image.interpolation' : 'none'}

# -----------------------------------------------------------------------------
# intersphinx
# -----------------------------------------------------------------------------
_python_version_str = '{0.major}.{0.minor}'.format(sys.version_info)
_python_doc_base = 'https://docs.python.org/' + _python_version_str
intersphinx_mapping = {
    'python': (_python_doc_base, None),
    'numpy': ('https://docs.scipy.org/doc/numpy',
              (None, './_intersphinx/numpy-objects.inv')),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference',
              (None, './_intersphinx/scipy-objects.inv')),
    'sklearn': ('http://scikit-learn.org/stable',
                (None, './_intersphinx/sklearn-objects.inv')),
    'matplotlib': ('https://matplotlib.org/',
                   (None, 'https://matplotlib.org/objects.inv'))
}

# ----------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------

import inspect
from os.path import relpath, dirname


# Function courtesy of NumPy to return URLs containing line numbers
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.findsource(obj)
    except:
        lineno = None

    if lineno:
        linespec = "#L%d" % (lineno + 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(skimage.__file__))

    if 'dev' in skimage.__version__:
        return ("https://github.com/scikit-image/scikit-image/blob/"
                "master/skimage/%s%s" % (fn, linespec))
    else:
        return ("https://github.com/scikit-image/scikit-image/blob/"
                "v%s/skimage/%s%s" % (skimage.__version__, fn, linespec))
