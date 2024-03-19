.. _tmdd:

====
TMDD
====

~~~~~~~
Running
~~~~~~~

The structsearch tool is available both in Pharmpy/pharmr.

The code to initiate structsearch for a TMDD model in Python/R is stated below:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')

    res = run_structsearch(type='tmdd',
                            model=start_model,
                            results=start_model_results)


Arguments
~~~~~~~~~
The arguments of the structsearch tool for TMDD models are listed below.

Mandatory
---------

+-------------------------------------------------+---------------------------------------------------------------------+
| Argument                                        | Description                                                         |
+=================================================+=====================================================================+
| ``type``                                        | Type of model. In this case "tmdd".                                 |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``model``                                       | PK start model.                                                     |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``results``                                     | ModelfitResults of the start model.                                 |
+-------------------------------------------------+---------------------------------------------------------------------+

Optional
--------

+-------------------------------------------------+---------------------------------------------------------------------+
| Argument                                        | Description                                                         |
+=================================================+=====================================================================+
| ``strictness``                                  | :ref:`Strictness<strictness>` criteria for model selection.         |
|                                                 | Default is "minimization_successful or                              |
|                                                 | (rounding_errors and sigdigs>= 0.1)". Optional.                     |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``extra_model``                                 | Extra model for TMDD structsearch. Optional. If specified,          |
|                                                 | 8 additional QSS models will be generated.                          |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``extra_model_results``                         | ModelfitResults object for the extra model for TMDD structsearch.   |
|                                                 | Optional if ``extra_model`` is not specified.                       |
+-------------------------------------------------+---------------------------------------------------------------------+
| ``dv_types``                                    | Dictionary of :ref:`DV types<dv_types>` for multiple DVs            |
|                                                 | (e.g. dv_types = {'target': 2}).                                    |
|                                                 | Allowed keys are: 'drug', 'target', 'complex', 'drug_tot' and       |
|                                                 | 'target_tot'. Optional.                                             |
|                                                 | Default is None which means that all observations are treated as    |
|                                                 | drug observations.                                                  |
+-------------------------------------------------+---------------------------------------------------------------------+

~~~~~~
Models
~~~~~~

Implemented target mediated drug disposition (TMDD) models are:

- Full model
- Irreversible binding approximation (IB)
- Constant total receptor approximation (CR)
- Irreversible binding and constant total receptor approximation (CR+IB)
- Quasi steady-state approximation (QSS)
- Wagner
- Michaelis-Menten approximation (MMAPP)


Full model:
~~~~~~~~~~~

.. graphviz::

    digraph G {

          subgraph cluster {
            peripheries=0;
            concentrate=true
            central [shape=rec]
            plus [label="+", shape=none, height=0, width=0]
            target [shape=rec]
            complex [shape=rec]
            central -> plus [color=none, minlen=0]
            plus -> target [color=none, minlen=0]
            complex -> target [headlabel="\n    koff", constraint=false, minlen=3]
            target -> complex [headlabel="kon    \n\n"]
            
            {rank=same;central;plus;target;complex}
          }

          kel [shape=none]
          ksyn [shape=none]
          kint [shape=none]
          kdeg [shape=none]

          central -> kel
          ksyn -> target
          target -> kdeg
          complex -> kint
    }


.. math:: \frac {dA_{\text{central}}}{dt} = k_{\text{off}} \cdot A_{\text{complex}}(t) \
            + \biggl(- \frac{\text{Cl}}{\text{V}} \
            - \frac{k_{\text{on}} \cdot A_{\text{target}}(t)}{\text{V}} \biggl) \cdot A_{\text{central}}(t)

.. math:: \frac {dA_{\text{target}}}{dt} = - k_{\text{deg}} \cdot A_{\text{target}}(t) \
            + k_{\text{off}} \cdot A_{\text{complex}}(t) \
            - \frac{k_{\text{on}} \cdot A_{\text{central}}(t) \cdot A_{\text{target}}(t)}{\text{V}} \
            + k_{\text{syn}} \cdot \text{V}

.. math:: \frac {dA_{\text{complex}}}{dt} = \frac{k_{\text{on}} \cdot A_{\text{central}}(t) \
            \cdot A_{\text{target}}(t)}{\text{V}}  + ( - k_{\text{int}} - k_{\text{off}}) \cdot A_{\text{complex}}(t)

IB model:
~~~~~~~~~

.. graphviz::

    digraph G {

          subgraph cluster {
            peripheries=0
            central [shape=rec]
            plus [label="+", shape=none, height=0, width=0]
            target [shape=rec]
            complex [shape=rec]
            central -> plus [color=none, minlen=0]
            plus -> target [color=none, minlen=0]
            target -> complex [label="kon", minlen=2]
            
            {rank=same;central;plus;target;complex}
          }

          kel [shape=none]
          ksyn [shape=none]
          kint [shape=none]
          kdeg [shape=none]

          central -> kel
          ksyn -> target
          target -> kdeg
          complex -> kint
    }


.. math:: \frac {dA_{\text{central}}}{dt} = \biggl(- \frac{\text{Cl}}{\text{V}} \
            - \frac{k_{\text{on}} \cdot A_{\text{target}}(t)}{\text{V}} \biggl) \cdot A_{\text{central}}(t)

.. math:: \frac {dA_{\text{target}}}{dt} = - k_{\text{deg}} \cdot A_{\text{target}}(t) \
            - \frac{k_{\text{on}} \cdot A_{\text{central}}(t) \cdot A_{\text{target}}(t)}{\text{V}} \
            + k_{\text{syn}} \cdot \text{V}

.. math:: \frac {dA_{\text{complex}}}{dt} = \frac{k_{\text{on}} \cdot A_{\text{central}}(t) \
            \cdot A_{\text{target}}(t)}{\text{V}} - k_{\text{int}} \cdot A_{\text{complex}}(t)


CR model:
~~~~~~~~~

.. graphviz::

    digraph G {

          subgraph cluster {
            peripheries=0
            central [shape=rec]
            plus [label="+", shape=none, height=0, width=0]
            target [shape=rec]
            complex [shape=rec]
            central -> plus [color=none, minlen=0]
            plus -> target [color=none, minlen=0]
            complex -> target [headlabel="\n    koff", constraint=false, minlen=3]
            target -> complex [headlabel="kon    \n\n"]
            
            {rank=same;central;plus;target;complex}
          }

          kel [shape=none]
          ksyn [shape=none]
          kint [shape=none]
          kdeg [shape=none]

          central -> kel
          ksyn -> target
          target -> kdeg
          complex -> kint
    }

.. math:: \frac {dA_{\text{central}}}{dt} = k_{\text{off}} \cdot A_{\text{complex}}(t) \ 
            + \biggl( - \frac{\text{Cl}}{\text{V}} - k_{\text{on}} \cdot R_0 \
            + \frac{k_{\text{on}} \cdot A_{\text{complex}}(t)}{\text{V}} \biggl) \cdot A_{\text{central}}(t)

.. math:: \frac {dA_{\text{complex}}}{dt} = \biggl( k_{\text{on}} \cdot R_0 -  \frac{ k_{\text{on}} \
            \cdot A_{\text{complex}}(t)}{\text{V}} \biggl) \cdot A_{\text{central}}(t) + \
            (- k_{\text{int}} - k_{\text{off}}) \cdot A_{\text{complex}}(t)

CR + IB model:
~~~~~~~~~~~~~~

.. graphviz::

    digraph G {

          subgraph cluster {
            peripheries=0
            central [shape=rec]
            plus [label="+", shape=none, height=0, width=0]
            target [shape=rec]
            complex [shape=rec]
            central -> plus [color=none, minlen=0]
            plus -> target [color=none, minlen=0]
            target -> complex [label="kon", minlen=2]
            
            {rank=same;central;plus;target;complex}
          }

          kel [shape=none]
          ksyn [shape=none]
          kint [shape=none]
          kdeg [shape=none]

          central -> kel
          ksyn -> target
          target -> kdeg
          complex -> kint
    }

.. math:: \frac {dA_{\text{central}}}{dt} =  \biggl(- \frac{\text{Cl}}{\text{V}} - k_{\text{on}} \cdot R_0 \
            - \frac{k_{\text{on}} \cdot A_{\text{complex}}(t)}{\text{V}} \biggl) \cdot A_{\text{central}}(t)

.. math:: \frac {dA_{\text{complex}}}{dt} = \biggl( k_{\text{on}} \cdot R_0 -  \frac{ k_{\text{on}} \
            \cdot A_{\text{complex}}(t)}{\text{V}} \biggl) \cdot A_{\text{central}}(t) \
            - k_{\text{int}} \cdot A_{\text{complex}}(t)

QSS model:
~~~~~~~~~~

.. graphviz::

    digraph G {

          subgraph cluster {
            peripheries=0
            central [shape=rec]
            plus [label="+", shape=none, height=0, width=0]
            target [shape=rec]
            complex [shape=rec]
            central -> plus [color=none, minlen=0]
            plus -> target [color=none, minlen=0]
            target -> complex [label="kD", minlen=2, dir=both]
            
            {rank=same;central;plus;target;complex}
          }

          kel [shape=none]
          ksyn [shape=none]
          kint [shape=none]
          kdeg [shape=none]

          central -> kel
          ksyn -> target
          target -> kdeg
          complex -> kint
    }

.. math:: \frac {dA_{\text{central}}}{dt} =  - \frac{Cl \cdot \text{LAFREE} \cdot A_{\text{central}}(t)}{V} \
            - \frac{Cl \cdot \text{LAFREE}}{V} - \frac{k_{\text{int}} \cdot \
            \text{LAFREE} \cdot A_{\text{target}}(t)}{k_{\text{D}} + \text{LAFREE}}

.. math:: \frac {dA_{\text{target}}}{dt} = k_{\text{syn}} \cdot V + \biggl(  -k_{\text{deg}} \
            - \frac{\text{LAFREE} \cdot (k_{\text{int}} - k_{\text{deg}})}{k_{\text{D}} + \text{LAFREE}} \biggl) \
            \cdot A_{\text{target}}(t)


Wagner model:
~~~~~~~~~~~~~

.. graphviz::

    digraph G {

          subgraph cluster {
            peripheries=0
            central [shape=rec]
            plus [label="+", shape=none, height=0, width=0]
            target [shape=rec]
            complex [shape=rec]
            central -> plus [color=none, minlen=0]
            plus -> target [color=none, minlen=0]
            target -> complex [label="kD", minlen=2, dir=both]
            
            {rank=same;central;plus;target;complex}
          }

          kel [shape=none]
          ksyn [shape=none]
          kint [shape=none]
          kdeg [shape=none]

          central -> kel
          ksyn -> target
          target -> kdeg
          complex -> kint
    }

.. math:: \frac {dA_{\text{central}}}{dt} =  - \frac{Cl \cdot \text{LAFREE}}{V} \
            + k_{\text{int}} \cdot \text{LAFREE} - k_{\text{int}} \cdot A_{\text{central}}(t)


MMAPP model:
~~~~~~~~~~~~

.. graphviz::

    digraph G {

          subgraph cluster {
            peripheries=0
            central [shape=rec]
            plus [label="+", shape=none, height=0, width=0]
            target [shape=rec]
            central -> plus [color=none, minlen=0]
            plus -> target [color=none, minlen=0]
            out [label="", shape=none]
            target -> out [label="(kdeg-kint) · A/V \n  ――――――― \n kMC + (A/V)"]
            
            {rank=same;central;plus;target;out}
          }

          kel [shape=none]
          ksyn [shape=none]
          kdeg [shape=none]

          central -> kel
          ksyn -> target
          target -> kdeg
    }

.. math:: \frac {dA_{\text{central}}}{dt} = \Biggl( - \frac{Cl}{V} - \frac{k_{\text{int}} \cdot \
            A_{\text{target}}(t)}{ V \cdot \Bigl( k_{\text{MC}} + \frac{A_{\text{central}}(t)}{V} \Bigl) } \Biggl) \
            \cdot A_{\text{entral}}(t)

.. math:: \frac {dA_{\text{target}}}{dt} = -k_{\text{deg}} \cdot A_{\text{target}}(t) + k_{\text{syn}} \
            - \frac{(k_{\text{kint}} - k_{\text{deg}}) \cdot A_{\text{central}}(t) \cdot A_{\text{target}}(t)}{V \
            \cdot \biggl( k_{\text{MC}} + \frac{A_{\text{central}}(t)}{V} \biggl)}


.. _dv_types:

~~~~~~~~
DV types
~~~~~~~~

The ``dv_types`` argument is a dictionary specifiying the DVs. If ``dv_types`` is not specified then all
observations are treated as drug observations.
There are five types implemented: 'drug' (free drug), 'target', 'complex', 'drug_tot' (total drug) and 'target_tot'
(total target).
Only 'drug' and 'drug_tot' can have dv=1.

Example 1:

.. code::

    dv_types = {'drug':1, 'target':2, 'complex':3}

In this case the drug has dv=1, the target has dv=2 and complex has dv=3:

Example 2:

.. code::

    dv_types = {'target':2, 'complex':3}

This is equivalent to the previous example. Per default drug has dv=1.

Example 3:

.. code::

    dv_types = {'target_tot':3, 'complex':2}

The dv types can be arranged in any possible order.

~~~~~~~~~~~~~~~~~~~~~
Structsearch workflow
~~~~~~~~~~~~~~~~~~~~~

There are two workflows for the structsearch tool for TMDD models, depending on whether the tool is used separately
or inside the AMD tool.
If used as a standalone tool, the structsearch workflow is as follows:

1. Create 8 QSS models for the input model and optionally additional 8 QSS models for the extra model if specified.
2. Find best QSS model of the 8(16) QSS models
3. Create 4 full models, 2 CR+IB models, 1 Wagner model, 2 CR models, 2 IB models and 1 MMAPP model.
   Use parameter estimates from the best QSS model as initial estimates for the generated models.
4. Find the best model of these 12 models.


When used inside the AMD tool:

1. Perform modelsearch
2. Get the final model of the modelsearch and a model with the same features as the final model but with one
   less peripheral compartment if one such model exists.
3. Create 8 QSS models for the final model and 8 QSS models for the final model minus one compartment if it exists.
   Otherwise only 8 QSS models are created.
4. Find best QSS model of the 8(16) QSS models
5. Create 4 full models, 2 CR+IB models, 1 Wagner model, 2 CR models,
   2 IB models and 1 MMAPP model. Use parameter estimates from the best QSS model as initial estimates for the
   generated models.
6. Find the best model of these 12 models.

.. graphviz::

    digraph BST {
            node [fontname="Arial"];
            base [label="Base model"]
            s0 [label="Modelsearch"]
            s1 [label="final model (+ final model -1 comp)"]
            s2 [label="8 (+ 8) QSS models"]
            s3 [label="best QSS model"]
            s31 [label="4 full"]
            s32 [label="2 CR+IB"]
            s33 [label="1 Wagner"]
            s34 [label="2 CR"]
            s35 [label="2 IB"]
            s36 [label="1 MMAPP"]

            base -> s0
            s0 -> s1
            s1 -> s2
            s2 -> s3
            s3 -> s31
            s3 -> s32
            s3 -> s33
            s3 -> s34
            s3 -> s35
            s3 -> s36
    }


~~~~~~~
Results
~~~~~~~

The results object contains various summary tables which can be accessed in the results object, as well as files in
.csv/.json format. The name of the selected best model (based on the input selection criteria) is also included.

Below is an example for a TMDD run.

.. pharmpy-code::

    res = run_structsearch(type='tmdd',
                            model=start_model,
                            results=start_model_results)

The ``summary_tool`` table contains information such as which feature each model candidate has, the difference to the
start model (in this case comparing BIC), and final ranking:

.. pharmpy-execute::
   :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/structsearch_results_tmdd.json')
    res.summary_tool

~~~~~~~~
Examples
~~~~~~~~

Run TMDD for multiple DVs:

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools read_modelfit_results, run_structsearch

    start_model = read_model('path/to/model')
    start_model_results = read_modelfit_results('path/to/model')

    res = run_structsearch(type='tmdd',
                            model=start_model,
                            results=start_model_results,
                            dv_types={'drug': 1, 'target': 2, 'complex': 3})

Note: "drug" can be omitted in ``dv_types``. In this case it will be set to 1.
