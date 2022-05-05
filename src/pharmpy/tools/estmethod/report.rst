.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

Estmethod Results
=================


Settings
~~~~~~~~

.. jupyter-execute::
    :hide-code:

    from pharmpy.results import read_results

    res = read_results('results.json')
    res.settings


Result summary
~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.summary

Results sorted by OFV
~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.sorted_by_ofv()
