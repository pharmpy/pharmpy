.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


Allometry Results
=================

Final model
~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    from pharmpy.tools.common import table_final_parameter_estimates

    res = read_results('results.json')
    final_model_parameter_estimates = table_final_parameter_estimates(
            res.final_results.parameter_estimates_sdcorr,
            res.final_results.standard_errors_sdcorr
            )


.. jupyter-execute::
   :hide-code:

   from pharmpy.visualization import display_table

   display_table(final_model_parameter_estimates, format={'estimates': '{:,.4f}', 'RSE': '{:,.1%}'})


.. jupyter-execute::
    :hide-code:

    res.final_model_cwres_vs_idv_plot


.. jupyter-execute::
   :hide-code:

   res.final_model_dv_vs_pred_plot


.. jupyter-execute::
   :hide-code:

   res.final_model_dv_vs_ipred_plot


.. jupyter-execute::
   :hide-code:

   res.final_model_eta_distribution_plot

