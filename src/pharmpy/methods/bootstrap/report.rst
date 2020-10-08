.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


Bootstrap Results
=================

OFV distribution
~~~~~~~~~~~~~~~~

.. altair-plot::
    :hide-code:

    from pharmpy.results import read_results

    res = read_results('results.json')
    res.plot_ofv()

Parameter estimates correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.plot_parameter_estimates_correlation()

Delta OFV vs original model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.plot_base_ofv()
