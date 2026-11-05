.. SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial_scripting:

====================
Scripting in Pharmpy
====================

In this tutorial, we will explore how Pharmpy can be used for end-to-end scripting for a simple example.
Our mission is to create a NONMEM model from scratch, try out some covariate effects, and finally run a bootstrap
for uncertainty estimation.

~~~~~~~~~~~~~~~~~~~~~~
Creating a start model
~~~~~~~~~~~~~~~~~~~~~~

As an example, we will use a built in dataset for phenobarbital administrated intravenously.

.. pharmpy-execute::
   :hide-output:

    from pharmpy.modeling import load_example_dataset
    dataset = load_example_dataset('pheno')

Using this dataset, we will create a basic PK model (see :py:func:`pharmpy.modeling.create_basic_pk_model` for more
details), and convert it to a NONMEM model.

.. pharmpy-execute::
   :hide-output:

    from pharmpy.modeling import create_basic_pk_model, convert_model
    model_start = create_basic_pk_model(administration='iv', input=dataset, cl_init=0.005, vc_init=1.0)
    model_start = convert_model(model_start, to_format='nonmem')

The model code will look as follows:

.. pharmpy-execute::
   :hide-output:

    from pharmpy.modeling import print_model_code
    print_model_code(model_start)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding some covariate effects and comparing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we want to create some models, fit them, and compare their results. In this case,
we'll look at adding covariate effects (see :doc:`api_modeling` for a comprehensive list
of available transformations).

.. pharmpy-execute::
   :hide-output:

    from pharmpy.modeling import add_covariate_effect, set_description, set_name

    run1 = add_covariate_effect(model_start, parameter='CL', covariate='WGT', effect='exp')
    run1 = set_description(run1, 'WGT on CL')
    run1 = set_name(run1, 'run1')

    run2 = add_covariate_effect(model_start, parameter='VC', covariate='WGT', effect='exp')
    run2 = set_description(run2, 'WGT on VC')
    run2 = set_name(run2, 'run2')

    run3 = add_covariate_effect(run1, parameter='VC', covariate='WGT', effect='exp')
    run3 = set_description(run3, 'WGT on CL and VC')
    run3 = set_name(run3, 'run3')

A feature that is intended to facilitate scripting is the Pharmpy :ref:`project`. When creating a project,
it will create a directory and a model database where all estimation results (e.g. the NONMEM result files)
will be stored. The purpose of this common database is that Pharmpy can identify whether a model has been
estimated before, and when calling the functions for fitting or running a tool, if a model has already been
estimated, those results can be read in instead of rerun. We create the project using the :py:func:`pharmpy.tools.open_project` function.

.. pharmpy-code::

    from pharmpy.tools import open_project, fit
    proj = open_project('myproject', tmp_path)

You can estimate these functions via the :py:func:`pharmpy.tools.fit` function, which will call
an estimation tool and parse the results. It will in this case call NONMEM since we converted the
model into a NONMEM model, but it's also possible to select other estimation tools.

Now that we have a project and some models of interest, we can estimate them. The output will be
a list of the modelfit results.

.. pharmpy-code::

    modelfit_results = fit([model_start, run1, run2, run3], name='modelfit', project=proj)
    modelfit_results[0].ofv

Note in the example that we're using two important options: :code:`name` and :code:`project`. The
:code:`name` option will be the name of the :ref:`context` that is used for the model fits. By setting a name,
it allow us to rerun the exact same line of code without rerunning the tool itself, instead relevant results will be
read in. The :code:`project` option tells Phampy where our model database is.

We can summarize these results in a data frame using :py:func:`pharmpy.summarize_modelfit_results`.
This function takes a context and reads in all models in it.

.. pharmpy-code::

    from pharmpy.tools import open_context, summarize_modelfit_results
    ctx = open_context('modelfit', ref='myproject')
    modelfit_summary = summarize_modelfit_results(ctx)
    modelfit_summary

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running a bootstrap on the best model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's proceed with the last candidate.

.. pharmpy-code::

    final_model = run3
    final_results = retrieve_modelfit_results(ctx, name=run3.name)
    final_results.ofv

Now we're interested in running a Bootstrap on this model in order to estimate the
parameter uncertainty.

.. pharmpy-code::

    bootstrap_res = run_bootstrap(model=final_model, results=final_results, samples=100, seed=12345, name='bootstrap1', project=proj)
    bootstrap_res.covariance_matrix
    bootstrap_res.ofvs

Note again that we name the tool run as well as provide the project.

Let's say we realize 100 samples is too little and we would like some more samples, giving the same seed and the project,
we can start a new bootstrap run of 200 samples, without needing to estimate 200 models. This is because the first 100
samples will be identical to the samples in our first bootstrap (since the seed is the same), so when Pharmpy does a
lookup in the common model database for the project, it can find those 100 first samples in the database and just read
those in instead of estimating them.

.. pharmpy-code::

    bootstrap_res = run_bootstrap(model=final_model, results=final_results, samples=200, seed=12345, name='bootstrap2', project=proj)
    bootstrap_res.covariance_matrix
    bootstrap_res.ofvs

