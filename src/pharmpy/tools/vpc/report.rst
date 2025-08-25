.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


VPC Results
===========

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results

    res = read_results('results.json')
    res.plot

