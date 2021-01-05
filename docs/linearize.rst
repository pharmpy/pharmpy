=========
Linearize
=========

Pharmpy currently creates results after a PsN linearize run.

~~~~~~~~~~~~~~~~~~~~~
The linearize results
~~~~~~~~~~~~~~~~~~~~~

OFVs
~~~~

The OFVs of the base model and the linearized model before and after estimation are summarized in the `ofv` table.

.. jupyter-execute::
    :hide-code:

    from pharmpy.results import read_results
    res = read_results('tests/testdata/results/linearize_results.json')
    res.ofv


Individual OFVs
~~~~~~~~~~~~~~~

The individual OFVs for the base and linearized models together with their difference is in the `iofv` table.

.. jupyter-execute::
    :hide-code:

    res.iofv
