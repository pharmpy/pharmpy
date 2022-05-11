=========
Linearize
=========

Pharmpy currently creates results after a PsN linearize run.

~~~~~~~~~~~~~~~~~~~~~
The linearize results
~~~~~~~~~~~~~~~~~~~~~

OFVs
~~~~

The OFVs of the base model and the linearized model before and after estimation are summarized in the ``ofv`` table. These values should be close. A difference signals problems with the linearization.

.. pharmpy-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/linearize_results.json')
    res.ofv


Individual OFVs
~~~~~~~~~~~~~~~

The individual OFVs for the base and linearized models together with their difference is in the ``iofv`` table. If there was a deviation in the ``ofv`` these values can be used to see if some particular individual was problematic to linearize.

.. pharmpy-execute::
    :hide-code:

    res.iofv

This is also plotted in ``iofv_plot``

.. pharmpy-execute::
    :hide-code:

    res.iofv_plot
