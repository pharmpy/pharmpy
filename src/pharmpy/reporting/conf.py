# See issue #20
# extensions = ['altair.sphinxext.altairplot']
extensions = ['pharmpy.reporting.altairplot', 'jupyter_sphinx']  # TODO: remove altairplot
html_sidebars = {
    '**': ['localtoc.html'],
}
html_static_path = ['.']
html_css_files = [
    'custom.css',
]
html_theme_options = {'page_width': '1750px'}
html_title = 'Results'
master_doc = 'results'
