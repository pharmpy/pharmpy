# See issue #20
# extensions = ['altair.sphinxext.altairplot']
extensions = [
    'pharmpy.reporting.altairplot',
    'pharmpy.reporting.jupyter_sphinx',
]  # TODO: Remove altairplot
html_sidebars = {
    '**': ['localtoc.html'],
}
html_static_path = ['.']
html_css_files = [
    'custom.css',
]
html_theme_options = {'body_max_width': 'none'}
html_title = 'Results'
master_doc = 'results'

# Added to avoid zmq.error.ZMQError: Address already in use
jupyter_execute_kwargs = {'extra_arguments': ["--transport=ipc"]}
