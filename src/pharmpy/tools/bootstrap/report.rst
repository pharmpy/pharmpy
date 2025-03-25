.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


Bootstrap Results
=================

Distribution of parameter estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results

    res = read_results('results.json')
    res.parameter_estimates_histogram

OFV distribution
~~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.ofv_plot

Estimated degrees of freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-code:

   res.dofv_quantiles_plot

Model summary
~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-code:

   from pharmpy.visualization import display_table
   display_table(res.summary_models)
