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


Delta Base iOFV
~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.plot_delta_base_ofv()
