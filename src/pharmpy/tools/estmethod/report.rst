.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


Estmethod Results
=================

Result summary
~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    from pharmpy.results import read_results

    res = read_results('results.json')
    res.summary
