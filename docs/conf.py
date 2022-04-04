import os


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.githubpages',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_automodapi.automodapi',
    'jupyter_sphinx',
    'sphinxcontrib.autoprogram',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

source_suffix = '.rst'
master_doc = 'index'
project = 'Pharmpy'
year = '2018-2022'
authors = ['the Pharmpy development team']
copyright = '{0}; {1}'.format(year, ', '.join(authors))
version = release = '0.62.0'
html_show_sourcelink = False

pygments_style = 'trac'
templates_path = ['.']

html_static_path = ['.']
html_theme = "pydata_sphinx_theme"
html_css_files = [
    'custom.css',
]
html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_theme_options = {
    'globaltoc_maxdepth': 2
}
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

automodapi_inheritance_diagram = False
graphviz_output_format = 'svg'

import doctest
doctest_default_flags = doctest.NORMALIZE_WHITESPACE
doctest_global_setup = '''
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
'''
