import os
import sys

sys.path.append(os.path.abspath('./_ext'))

extensions = [
    'sphinx.ext.autodoc',
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
    'sphinx_tabs.tabs',
    'sphinxcontrib.autoprogram',
    'pharmpy_snippet',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

source_suffix = '.rst'
master_doc = 'index'
project = 'Pharmpy'
year = '2018-2023'
authors = ['the Pharmpy development team']
copyright = '{0}; {1}'.format(year, ', '.join(authors))
version = release = '0.98.2'
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
    'globaltoc_maxdepth': 2,
    "logo": {
        "image_light": "Pharmpy_logo.svg",
        "image_dark": "Pharmpy_logo_dark.svg",
    },
    'github_url': 'https://github.com/pharmpy/pharmpy',
    "icon_links": [
        {
            "name": "Report issues",
            "url": "https://github.com/pharmpy/pharmpy/issues",
            "icon": "fa fa-comments",
        }
    ]
}
html_favicon = 'images/Pharmpy_symbol.svg'
html_context = {
   "default_mode": "light"
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

automodapi_inheritance_diagram = False
autodoc_typehints = 'none'
graphviz_output_format = 'svg'

import doctest
doctest_default_flags = doctest.NORMALIZE_WHITESPACE
doctest_global_setup = '''
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
'''

linkcheck_ignore = [r'https://doi.org/10.1002/psp4.12741']
