next version
------------

New features
============

* Add parameter scatter matrix plot to bootstrap
* Add :code:`modeling.insert_ebes_into_dataset`
* Add :code:`modeling.add_placebo_model`
* Add :code:`modeling.binarize_dataset`
* Add option :code:`exclude_reference_model` to the ModelRank tool

Changes
=======

* Make the OFV distribution plot wider in the bootstrap results and report
* Make tables in reports have 50 rows per page
* Let :code:`append_estimation_step_options` operate on the final estimation step by default

Bugfixes
========

* Fix regression having iivsearch crash if the input model is selected before the end
* Fix so that idx=-1 does not duplicate estimation steps for :code:`append_estimation_step_options`

.. _1.10.0:

1.10.0 (2025-09-18)
-------------------

New features
============

* Add support for a built in estimation engine in fit and other tools via :code:`esttool='pharmpy'`. It is so far limited to estimating some very basic models.
* Add support for running parameter uncertainty estimation on only top ranked models in AMD

Changes
=======

* COVSearch no longer supports running only structural covariates, AMD instead runs model for this

Bugfixes
========

* Fix crash when reading in a dataset, which has a column header starting with #, in for example `create_basic_pk_model`
* Stop AMD from crashing when ext file of some model couldn't be parsed.
* Fix recent regression in parsing of some NONMEM models with multiple DVs
* Fix bug in setting initial estimates for :code:`modeling.transform_etas_tdist`

.. _1.9.0:

1.9.0 (2025-09-03)
------------------

New features
============

* Support additive error model in ruvsearch
* Automatically set PD in $SIZES for NONMEM models when needed
* Automatically add a dummy DV column in NONMEM if needed. This will make it possible to do simulations in NONMEM without having a DV in the dataset.
* Add :code:`modeling.map_eta_parameters` for creation of mappings between connected individual parameters, etas or omegas.
* Add :code:`modeling.infer_datatypes` to infer and check if data columns can be converted a simpler datatype (e.g. int32)
* Add :code:`modeling.set_n_transit_compartments`
* Add :code:`modeling.create_basic_pd_model`
* Add :code:`modeling.export_model_files` to export all estimation tool specific files into one directory
* Add first version of a VPC tool
* Add tool ModelRank

Changes
=======

* :code:`set_initial_estimates` with :code:`strict=False` will ignore any new inits being NaN.
* Deprecate :code:`write_csv` in favour of :code:`write_dataset`. The old function will be removed
  in the next minor version. This gives the function a more appropriate name and counters a collision
  in pharmr with tidyverse.
* Use ModelRank in all AMD-tools
* Do not assume PK model in COVSearch (previously raised InputValidationError when central compartment could not be found)

Bugfixes
========

* Fix crash when updating $THETA with some changes to a record of multiple theta parameters
* Fix crash in creating plots using residuals or predictions
* Fix crash when reading NONMEM model with $PK, but no dosing column in the dataset
* Fix crash when reading NONMEM model with $PK and no event, MDV or dose column in the dataset
* Fix incorrect condition number calculation in strictness evaluation (now uses the correlation matrix instead of covariance matrix)

.. _1.8.0:

1.8.0 (2025-06-19)
------------------

New features
============

* Support for Linux on AArch64 (ARM64) added
* Add :code:`modeling.time_of_last_dose`
* Add :code:`strictness` option to the bootstrap tool 

Bugfixes
========

* Fix crash when parsing some numeric options in NONMEM $TABLE (For example NPDTYPE=1)
* Handle MFL as input to structsearch for drug metabolite (#3821) 
* Fix seeds for simulations in amd for NONMEM models to get too big
* Do not warn about not running iovsearch, covsearch or allometry if not requested in strategy
* Fix crash when interrupting a tool multiple times (when clicking stop multiple times in rstudio or spyder for example)
* Fix crash when resuming an interrupted amd run
* Remove IPREDADJ when going from a proportional error to an additive
 
.. _1.7.2:

1.7.2 (2025-05-21)
------------------

Bugfixes
========

* Fix crash when using nmfe in PATH when running NONMEM
* Fix bug causing metadata to not have proper serialization of models and results for input arguments causing rerunning to crash
* Fix bug when end time is added to metadata at workflow abort
* Handle metadata and resume functionality for :code:`fit()`
* Readd input validation for :code:`run_amd`
* pharmr: Make return values invisible in case of a critical error when running a tool
* pharmr: Previous errors could sometimes be shadowing current errors. Fixed by clearing last Python error.
* pharmr: Give proper error message for :code:`DispatchingError`
* Fix crash when using version 2.4.0 of direct dependecy itables

.. _1.7.1:

1.7.1 (2025-05-14)
------------------

Bugfixes
========

* Fix issue causing default values of options not to be used when detecting if a tool run is run with the same options or not
 
.. _1.7.0:

1.7.0 (2025-05-12) 
------------------

New features
============

* Add :code:`dv` option to :code:`get_observations`
* Add context method :code:`spawn_seed`
* Add :code:`modeling.is_simulation_model`
* Add :code:`symbol` attribute to :code:`ColumnInfo` class
* Add :code:`final_results` to :code:`AMDResults` class
* Improve reports of AMD tools
* Add option :code:`dofv` to bootstrap
* Add address of dask dashboard to the log message when using the dask dispatcher
* Add :code:`modeling.get_nested_model`
* Add :code:`modeling.set_weibull_absorption` and :code:`modeling.has_weibull_absorption`

Changes
=======

* Let :code:`minimization_successful` be False if any final estimate is reported as infinity.
* AMD will be aborted if all models failed in modelsearch
* Results objects are no longer serialized into the metadata.json. Instead a reference to the model databases is stored.
* All common options are only present in the metadata.json of the top level tool 
* Let :code:`results.final_model` be the actual model object in AMD
* Serialize all model objects in results objects (and in results.json)
  
Bugfixes
========

* Fix serious bug in the bootstrap tool causing replacements to be done
* Fix bug causing groups!=4 in ruvsearch to crash or give wrong results
* Set :code:`MDVRES` in the NONMEM code for BLQ models
* Handle argument types properly for add_estimation_step in pharmr. Now :code:`add_estimationstep(..., maximum_evalutions=9999)` will not lead to MAXEVAL=9999.0 in the NONMEM code.
* Infinite or NaN values are no longer accepted as parameter initial estimates
* Arguments to :code:`run_amd` will be validated before creating the directory
* Fix running nlmixr on Windows
* Fix cases where :code:`LTH` was removed from :code:`$SIZES` in the NONMEM code when it had a negative value 
* Set :code:`model.value_type` in :code:`modeling.transform_blq`. This didn't affect any generated NONMEM code, but left the Pharmpy model in an incorrect state.
* Make sure to keep the zero protection when going to a combined error model


1.6.0 (2025-02-10)
------------------

New features
============

* Add support for running NONMEM 7.6
* Better support for stopping a tool (via CTRL-C or SIGTERM from Slurm timeout) 
* New option to set :code:`broadcaster` of messages from a tool run. Can currently be set to `terminal`
  which is the default and `null` to turn off broadcasting.
* Option to specify variable for modeling.set_direct_effect

Changes
=======

* Merge contents of nonmem.json into .pharmpy/metadata.json. This was made to reduce the number of files generated by
  Pharmpy.
* Make candidate naming consistent in TMDD structsearch tool
 
Bugfixes
========

* Handle compartment definitions in $MODEL having spaces next to commas, e.g. :code:`(DEPOT, DEFOBS)`
* Fix bug in tools where if a model had less parameters than its parent it would not update initial estimate
* Make sure that a created $SIZES always comes before the first $PROBLEM in NONMEM models
* Make modeling.has_linear_odes_with_real_eigenvalues faster in many common cases
* Fix issue in modeling.set_tmdd where dvid was not extracted correctly from datainfo (#3618)
* Fix issue in modeling.has_mu_reference that caused COVsearch to crash in models with IOV (#3429)
* Raise in RUVSearch if input model has TAD statement
* Add delay when cleaning up temporary directory in Windows (fix sporadic permission errors)
* Fix issue where tools crashed when creating reports in Rstudio on Windows


1.5.0 (2025-01-15)
------------------

New features
============

* Add validation of input models in the simulation tool
* Add more log messages to modelsearch and iivsearch
* Add init and lower bound as arguments to modeling.add_individual_parameter
* Add method Statements.get_assign
* Allow THETA, OMEGA, SIGMA and TABLE to be encoded THETAS, OMEGAS, SIGMAS and TABLES in NONMEM models
* Automatically update the ISAMPLEMAX in $SIZES when needed
* Better support for :code:`NEWIND` in NONMEM code
* Add option :code:`ncores` to set a limit for the number of cpu cores to use when running a tool or :code:`fit`
* New common option :code:`dispatcher` to tools.
* New dispatcher :code:`local_serial` that can use NONMEM parallelization via MPI
* Automatically create the parafile for NONMEM both for running on Slurm and locally when using the :code:`local_serial` dispatcher
* Directly retrieve results if tool is rerun in same context. This will enable scripts to be rerun without changes
* Allow for resuming a previously interrupted run by running the exact same call
* Add :code:`ref` and :code:`name` common options to select run directory for a tool

Changes
=======

* Use a random seed if seed was not given to a tool
* Improved the initial estimate of EMAX for indirect effect degradation models in structsearch
* model and results are now mandatory arguments to modelsearch, covsearch, ruvsearch, structsearch, iovsearch, iivsearch and allometry
* AMD can no longer take a DataFrame as input (only a dataset file). This didn't work previously. 
* Replace the :code:`path` option from :code:`fit` with :code:`name`
* Rename :code:`init_context` to :code:`open_context`
* Remove the now legacy :code:`resume` common option
* Let :code:`fit` by default use the :code:`local_serial` dispatcher

Bugfixes
========

* AMD should now be fully deterministic given the same seed
* Make order of candidate models in covsearch deterinistic (#3488)
* Fix crashes of the vpc in amd if simulation table couldn't be found
* Change lower bound for EMAX parameters to -1
* Let the SLOPE (PD) parameter have no lower bound
* Add allometric scaling to base model in amd when using ALLOMETRY in the search space
* Give proper error when allometric variable couldn't be found in the dataset for modelsearch
* Give proper error if parameter in the keep-option for iivsearch doesn't exist
* Fix bad parsing of some NONMEM models with multiple DVs
* Fix crashes in transformation functions for NONMEM models using T in $DES non-derivative assignments
* Fix issues with roundtrips of NONMEM parameter records having decimal values starting with . (dot)
* Do not test any etas on the RUV model in iivsearch
* Fix issues where tmp directory couldn't be removed on Windows causing a crash


1.4.0 (2024-12-04)
------------------

New features
============

* Support Python 3.13
* Support EFIM in estmethod-tool
* Add STEP and LOGLIN to MFL DIRECTEFFECT and EFFECTCOMP
* Add `create_context`, `print_log`, `retrieve_model` and `retrieve_modelfit_results` to `tools`

Changes
=======

* Allow `set_tmdd` to work for models without dataset
* Do input validation for amd earlier to fail before starting the tool
* Make `reduced_stepwise` the default algorithm in `modelsearch`. There was no default previously.
* Store model database key instead of name in metadata for inputs of type `Model`

Bugfixes
========

* Fix bug causing retries crash with error "Parameters not found in model: ['0']"
* Fix crashes in vpc plotting in amd
* Attempt fix of crashed with error "zmq.error.ZMQError: Address already in use"
* Handle amd input check warnings in context log
* Fix bug causing amd option `lloq_limit` to not work (issue #3404)
* Fix mBIC calculation in IIVsearch bottom-up approach
* Fix bug in COVsearch where incorrect modelfit results are stored as final_results
* Add removed RUVsearch step in AMD algorithm SIR
* Fix covsearch removing allometric variable


1.3.0 (2024-10-24)
------------------

New features
============

* Support `DataFrame` as input to `run_amd`
* Recognize "HESSIAN OF POSTERIOR DENSITY..." error from NONMEM (issue #3326)
* Add modeling.replace_fixed_thetas
* Add two version of the SAMBA method to covsearch
* Add modeling.get_mu_connected_to_parameter and modeling.has_mu_reference
* Support percentages for E-value in mBIC calculations
* Add strict-option in modeling.parameters-functions

Changes
=======

* Add replacement of deterministic random variables (0 FIX) in modeling.cleanup_model
* Add replacement of fixed thetas in modeling.cleanup_model
* Set ONEHEADER to newly created $TABLES for NONMEM
* Make an added RATE column for ZO absorption be int32 instead of float64
* Fix issue with different sample sequences for multivariate normal distribution between arm Macs
  and other platforms. The fix will use another sampling method, which means that it will not
  be possible to reproduce sampled values between this version of Pharmpy and the previous
* Make the default option to remove all in modeling.remove_residuals and modeling.remove_predictions None instead of 'all'
* Do not allow None for ExecutionStep.tool_option. Instead have an empty frozendict as default
* Add separate step for delinearized model in IIVSearch results
* Do not update initial estimates in tools from a model with number of significant digits unreportable
* Remove influential individual and outlier prediction tables in all tools
* Run start model in AMD in subcontext
* Add selected models to AMD models-directory

Bugfixes
========

* Fix reading in NONMEM models with TIME column having hh:mm format (but no DATx column present)
* Fix NONMEM parsing issues where having WRES in $TABLE could lead to parsing other columns incorrectly
* Let translate_nmtran_time return the input model if the input model has no dataset
* Fix bug causing NONMEM code to keep bounds on thetas after unconstrain_parameters
* Fix crashes when starting amd using the command line interface
* Fix crashes in modelsearch when running TMDD and drug metabolite in amd (issue #3203)
* Now all digits of the OFV for a NONMEM run will be read. Previously only about 13 decimals were read correctly
* Check for strictness in ruvsearch
* Fix permission denied error on multi user system for the `jupyter_execute` temp directory. This issue will
  trigger if multiple users happen to run the report generation at the same time or if a previous run crashed without
  removing the temp directory.
* Make remove_iiv handle cases where multiple assignments to same variable is made. For example when allometry has been added.
* IIVSearch bottom up algorithm does no longer run the base model
* Fix issue where delinearized model had the wrong BIC reported in result summary
* Fix bug where results from input model was not used in linearized workflow in IIVSearch
* Fix bug where files where not copied from a failed NONMEM run
* Fix AMD metadata (#3328)
* Fix bug where strictness was not checked in IOVSearch
* Fix bug where model files were overwritten if they already existed in model database
* Calculate mBIC correctly for IOVSearch

1.2.0 (2024-08-22)
------------------

New features
============

* Add `missing_data_token` to `DataInfo`. This will support a per dataset token for missing data
* Add option `missing_data_token` to `modeling.read_model` 

Changes
=======

* Replace configuration items pharmpy.data.na_rep and na_names with pharmpy.missing_data_token.

Bugfixes
========

* Fix problems with the error "[WinError 6] The handle is invalid on Windows" when creating the context
* Fix crash when updating some NONMEM models where a central compartment couldn't be found
* Fix crash when parsing certain error models that use CMT in NONMEM
* Fix multiple problems with running iivsearch with linearization
* Recognize SS and II columns when running amd from dataset
* Fix bad TAD for observation at the same time as SS dose. Was previously II. Now it is 0.

1.1.0 (2024-07-17)
------------------

New features
============

* Add modeling.set_description 

Bugfixes
========

* (pharmr) Fix regression in input conversion of `keep`-option in IIVSearch
* Allow lists in option arguments with length of 1 in modeling.add_iiv (e.g. `expression`)
* Fix regression in setting transit compartments (#3116)
* Fix crash in amd tool for search spaces with only covariate effects (#3113)
* Fix bad error message when only supplying a model to run_iivsearch
* Return NotImplemented instead of False for equality comparison of model objects
* Calculate mBIC correctly for modelsearch
* Fix crash of iivsearch for etas on epsilons in some cases, by not consider such etas in iivsearch

1.0.1 (2024-06-12)
------------------

Bugfixes
========

* Fix issue causing pharmr to crash directly before returning from some tools.
* Fix crash in NONMEM parser for CALLFL=1 statement
* Fix crashes in parsing of NONMEM dataset when AMT column had a synonym, e.g. DOSE=AMT
* Correctly create the statement for F for NONMEM models using SC as scaling factor

1.0.0 (2024-05-30)
------------------

New features
============

* eta and epsilon derivatives can be requested for NONMEM models
* Add a dummy estimation tool that could be used for testing and demonstrations
* Support linearization in iivsearch

Changes
=======

* Problematic input arguments to tools will raise InputValidationError
* The log file is now a csv file in the local direcory context
* The parent model name is no longer a part of a Model object
* All tools stores the input and final models in the context
* Input and final models will now have these names in all tool result tables
* Input models will have a proper (or empty) description in tool result tables
* modeling.vpc_plot renamed to modeling.plot_vpc
* Cleaned up the pheno model in load_example_model
* The "lzma" option in write_results was renamed to "compression"
* summarize_errors, resume_tool, rank_models, get_model_features, create_results and retrieve_final_model in tools are no longer exported
* tools.is_strictness_fulfilled have a new argument structure
* All PsN commands in the CLI have been moved to separate subcommand

Bugfixes
========

* The amd tool will raise an error if instantaneous absorption is combined with oral administration
* Make symlinks in the local directory context be relative so that the directory can be moved (does not work on Windows)
* Fix initial estimates for cat2 in covsearch
* Fix calculation of d_params in iivsearch results. Was using parent and now uses the base model.
* Make sure that predictions and residuals in EstimationStep have sorted order
* Fix broken --explicit-odes option in CLI "model print"
* "results ofv" has been removed from the CLI
* Do not output the full ModelfitResults object in results.csv

0.110.0 (2024-05-08)
--------------------

New features
============

* Add alternative categorical effect "cat2" in covsearch
* Add adaptive scope reduction and maxevals restriction in covsearch

Changes
=======

* ToolDatabase replaced with Context
* Default Context gives a new file system organization for tools
* The same model can have multiple names in a context
* New API for ModelDatabase
* Allow clashes of names in $INPUT and parameter comments
* Rename Model property model_code to code
* Rename estimation_steps to execution_steps
* Change index for modelfit_results.predictions and residuals
* Set BIC as the default selection criteria in AMD

Bugfixes
========

* Do not remove IOV if present in covsearch
* Properly parse OFV for NONMEM runs using SAEM
* Fix crash in retries caused by having fixed thetas

0.109.0 (2024-04-10)
--------------------

New features
============

* The structsearch tool can now take an mfl string as search space
* Allow list of lists of parameters in iovsearch options
* Explicitly handle CMT columns for observations in the expression for F for NONMEM models

Changes
=======

* Deafault to keeping eta on CL in iivsearch

Bugfixes
========

* Handle cases with covariates on MU when parsing phi-file with PHI columns
* Fix issues with allometry and structural covariates for PKPD models in amd
* Fix broken conversion to RxODE for models without ODE system
* Fix handling of datasets with multiple DVIDs in ruvsearch and sructsearch for TMDD models


0.108.0 (2024-03-18)
--------------------

New features
============

* New simulation tool tools.run_simulations
* Add reports with various plots for final model in all AMD subtools
* Add VPC plot to AMD
* Add functions modeling.add_predictions_residuals and modeling.remove_predictions_residuals

Changes
=======

* Force positive definitiveness in retries tool
* Make initial estimates in AMD mandatory
* Add default search space for PKPD models
* Remove TMDD models with less than 2 DVs

Bugfixes
========

* Correct extraction of CL/VC parameters in PSC metabolite models
* Remove unused K-parameters from NONMEM code
* Correct number of expected models in BIC calculation for bottom up algorithm in IIVSearch
* Use input model results when updating initial estimates for first model in bottom up algorithm in IIVSearch
* Add keep-option to bottom up algorithm in IIVSearch

0.107.0 (2024-03-04)
--------------------

New features
============

* Add bottom up algorithm in IIVSearch
* Add modeling.set_dataset
* Add AMD results plots for each DV
* Add default search space for TMDD in AMD
* Support EFIM as parameter uncertainty method
* Allow specific (covariate, parameter) combinations as part of input for mechanistic covariates in AMD

Changes
=======

* Use Pharmpy class Expr instead of sympy/symengine
* Remove ModelfitResults attribute from Model
* Change logic of IIVsearch ``algorithm`` argument, add ``correlation_algorithm`` argument
* Ignore datainfo fallback in AMD (to avoid automatically filling in information not given by user)
* Raise error instead of warn when expression is invalid in modeling.filter_dataset
* Change strategy names in AMD (to "default" and "reevaluation")

Bugfixes
========

* Keep IIV on all clearance parameters of central compartment in AMD
* Fix bug in AMD for TMDD models where `dir_name` was not specified
* Fix bug with naming of K-parameters in models with 9 compartments
* Skip first order absorption with 1 transit (no depot) combination
* Fix bug which caused added IOVs to be removed in covsearch when running AMD (edited)

0.106.0 (2024-01-11)
--------------------

New features
============

* Change to mBIC as default ranking function in modelsearch, iivsearch and iovsearch
* Add modeling.get_central_volume_and_clearance
* New option parameter_uncertainty_method to amd
* New option ignore_datainfo_fallback to amd
* Handle conversion to ETA/ETC for PHI/PHC in NONMEM phi files

Changes
=======

* Remove the order option in amd and instead add strategy with "fixed" orders and options to subtools

Bugfixes
========

* Allow ~ in paths in write_csv
* Have non-linear elimination models in default search space for amd TMDD models
* Fix issue causing removed off-diagonal omegas being transformed into thetas
* Fix issues in frem postprocessing when using mu-referencing

0.105.0 (2023-12-07)
--------------------

New features
============

* Add modeling.bin_observations
* Add modeling.plot_dv_vs_pred
* Add modeling.plot_abs_cwres_vs_ipred
* Support strictness for thetas, omegas and sigmas separately
* Support stagewise addition of covariates in amd
* Support multiple DVs for TMDD models
* Add retries tool
* Use retries in the amd

Changes
=======

* Always keep an iiv eta on clearence in amd

0.104.0 (2023-11-06)
--------------------

New features
============

* Add modeling.replace_non_random_rvs
* Add option keep_index to modeling.get_observations to allow keeping the original dataset index
* Add path-option to tools.fit
* Add function tools.is_strictness_fulfilled
* Add strictness option to AMD and subtools
* Add TMDD models to AMD
* Add option for TMDD models in structsearch
* MFL for COVSearch works the same way as for Modelsearch

Bugfixes
========

* Properly handle 0 FIX etas in calculate_bic
* Fix crash when setting 0 transit compartments
* Fix various bugs in TMDD models (including allometry)

0.103.0 (2023-10-12)
--------------------

Changes
=======

* Update initial estimates in structsearch
* Add option dv to modeling.get_individual_parameters
* Add default search space for drug-metabolite models in AMD

Bugfixes
========

* Fix bug in calculate_bic where parameters were incorrectly set to fixed for PKPD and drug-metabolite models
* Fix bug in COVSearch step numbering
* Fix bug in COVSearch where the final_model was set incorrectly
* Fix bug in COVSearch where p-value wasn't displayed for backward models

0.102.0 (2023-09-28)
--------------------

New features
============

* Add modeling.plot_dv_vs_ipred
* Add modeling.plot_cwres_vs_idv
* Add modeling.add_indirect_effect
* Add option for presystemic circulation for modeling.add_metabolite
* Add bic for multiple testing in modeling.calculate_bic
* Support PKPD models in the amd tool
* Support Drug-metabolite models in the amd tool
* Add first version of report for amd
* Add PKPD models to MFL
* Add modeling.filter_dataset

Changes
=======

* Change default p-value in ruvsearch from 0.05 to 0.001
* Change default p-values in covsearch from 0.05 and 0.01 to 0.01 and 0.001 
* Change the mfl for modelsearch to mean search space and not which transformations to do 
* Change the syntax for LAGTIME in the MFL

Bugfixes
========

* Allow ~ for home directory in read_modelfit_results

0.101.0 (2023-09-01)
--------------------

New features
============

* Add modeling.load_dataset and modeling.unload_dataset
* Add @BIOAVAIL to MFL
* Add support for iv-oral administration for amd tool


0.100.0 (2023-08-25)
--------------------

New features
============

* Support for M5, M6 and M7 methods for blq data
* New symbols @PK and @PD in MFL
* Internal support for multiple doses to one compartment

Changes
=======

* Change the blqdv type to blq in datainfo
* Better usage of BLQ and LLOQ columns for blq data

0.99.0 (2023-08-23)
-------------------

New features
============

* Add function modeling.set_reference_values
* Add function modeling.set_lloq_data
* Parse IV+oral models using CMT column
* Specify DV in RUVSearch
* Option to add logit IIV in ``add_iiv``
* New options for remove_loq_data

Changes
=======

* Make Task and Workflow immutable
* Ignore fixed IIVs/IOVs in IIVSearch and IOVSearch

Bugfixes
========

* Fix bug where epsilons where removed in ``remove_iiv``
* Fix bug in ``create_basic_pk_model`` to handle space separated datasets

0.98.0 (2023-07-21)
-------------------

New features
============

* Support for multiple doses
* Add function ``modeling.add_bioavailability``
* Add function ``modeling.remove_bioavailability``
* Support for PKPD models in structsearch
* Option to keep IIVs in IIVSearch
* Option to test uncertainty methods in Estmethod
* Autogenerate CMT column

Changes
=======

* Rename BLQ flag datainfo typ to ``blqdv``

0.97.0 (2023-06-28)
-------------------

New features
============

* Support BLQ transformations in RUVSearch
* New tool structsearch and support for TMDD models
* Add function ``modeling.set_direct_effect``
* Add function ``modeling.add_effect_compartment``

Changes
=======

* Reorganizing of modeling module
* Support changing error model with BLQ transformation
* Add ``max_iter`` option for RUVSearch

0.96.0 (2023-05-26)
-------------------

Changes
=======

* Rename functions handling the precision matrix (was previously referring to information matrix which was an error)
* Remove saddle reset for default AMD model
* Let LLQ column takes precedence over BLQ column

New features
============

* Add tools.load_example_modelfit_results

Bugfixes
========

* Fix bug where if-statements were reordered incorrectly

0.95.0 (2023-05-22)
-------------------

Changes
=======

* ``ModelfitResults.ofv_iterations`` and ``ModelfitResults.parameter_estimates_iterations`` have NaN rows in failed runs

Bugfixes
========

* Fix bug causing changes in FIX from model1 to model4 to crash frem
* Fix bug causing individual parameters in $ERROR to crash frem
* create_report now does not assume that results.json already exists
* ~ for $HOME is now supported in write_model and create_report
* Fix bug where LLOQ value did not override column in dataset in ``transform_blq``
* Correct BLQ indicator column condition in ``transform_blq``
* Fix bug where modelfit results were not connected to model after a fit

0.94.0 (2023-04-26)
-------------------

New features
============

* Support parsing assignments other than DADT in $DES in NONMEM
* Fix parsing of some complex ODE-systems in NONMEM

Changes
=======

* Drop support for Python 3.8

Bugfixes
========

* Fix bug causing BIC calculation to fail for models having first order absorption and lag_time after going into zero order absorption

0.93.0 (2023-04-19)
-------------------

New features
============

* Add function ``modeling.get_zero_order_inputs``
* Add function ``modeling.set_zero_order_input``
* Add function ``modeling.set_tmdd``
* Added plugin to convert models to RxODE
* Support conversion of more models to nlmixr

Changes
=======

* ``modeling.generate_model_code`` was renamed to ``modeling.get_model_code`` since the code is not generated by this function
* Do not use ADVAN7 because models that should work with ADVAN7 didn't were found

Bugfixes
========

* Fix multiple bugs in parsing $TABLE headers

0.92.0 (2023-04-05)
-------------------

New features
============

* Add function ``modeling.is_linearized``
* Add function ``modeling.plot_transformed_eta_distributions``
* Add function ``modeling.create_config_template``
* Add function ``modeling.get_dv_symbol``
* Add function ``modeling.get_initial_conditions``
* Add function ``modeling.set_initial_condition``
* Add function ``modeling.transform_blq``

Bugfixes
========

* Fix bug where $ABBR wasn't added for etas

0.91.0 (2023-03-03)
-------------------

New features
============

* Add function ``modeling.create_basic_pk_model``
* Add function ``modeling.add_metabolite``
* Add function ``modeling.set_dvid``
* Add function ``modeling.has_weighted_error_model``

Changes
=======

* ``model.dependent_variable`` becomes ``model.dependent_variables``

Bugfixes
========

* Fix regression causing DEFDOSE to sometimes be put on the wrong compartment
* Fix ruvsearch crashing in case of bad modelfit_results (#1551)

0.90.0 (2023-02-24)
-------------------

New features
============

* Add function ``modeling.has_odes``
* Add function ``modeling.has_linear_odes``
* Add function ``modeling.has_linear_odes_with_real_eigenvalues``
* Add function ``modeling.is_real``
* Support for more types of models in the nlmixr plugin
* Automatic selection between ADVAN5 and ADVAN7 for NONMEM models

Changes
=======

* Remove modeling.copy_model
* Support nlmixr2 instead of nlmixr for the nlmixr plugin
* The Model class is now immutable
* update_source is run by all transformation functions

0.89.0 (2023-01-26)
-------------------

New features
============

* Add function ``modeling.display_odes``
* Add support for Python 3.11

Changes
=======

* Naming of parameters for NONMEM models reworked. Configuration options removed.
* Only allow MFL as input to ``run_covsearch``
* Remove ``read_model_from_database`` from ``pharmpy.modeling``
* Merge ``ExplicitODESystem`` into ``CompartmentalSystem``

0.88.0 (2022-12-21)
-------------------

New features
============

* Add algorithm `exhaustive_only_eval` to Estmethod tool
* Add replace methods to Assignment, Compartment, Bolus and Infusion

Changes
=======

* Rename algorithms in Estmethod tool: `reduced` -> `exhaustive`. `exhaustive` -> `exhaustive_only_eval`
* Always add iteration 0 in ofv_iterations and parameter_estimates_iterations for eval models with FO/FOCE

0.87.0 (2022-12-14)
-------------------

Changes
=======

* Allometry model will update initial estimates in allometry tool
* Base model in IIVSearch tool will update initial estimates
* Do not update initial estimates from model that did not minimize successfully (except rounding errors), this affects all AMD subtools
* Rename derive to replace in some base classes

0.86.0 (2022-11-30)
-------------------

Changes
=======

* Add description to proxy-models (#1314)
* Input check covariates (#1355), allometric variable (#1378) occasion-column before running AMD

Bugfixes
========

* Fix typo in COVSearch that caused it to select model with highest OFV (#1377)
* Ignore NaNs when selecting models in COVSearch (#1381)
* Fix issue where initial estimate for KM (in MM-elimination) was set outside of NONMEM's bounds (#1064)
* Fix issue where individuals without observations were not filtered for general model objects (afd7707, #1139)
* Fix issue where saddle reset was not added in start model for AMD (#1394)

0.85.0 (2022-11-18)
-------------------

Changes
=======

* Covariates are defined in search space option in AMD-tool
* Store name of final model instead of final model in AMD
* Change methods and solvers option in estimation method tool: None means none should be tested
* Add FORMAT option if length of IDs are too long (#1139)
* Make Result classes immutable

Bugfixes
========

* Fix bug in results parsing where extracting whether parameters are fixed (#1117)
* Add timeout-loop to wait for .lst-file when renaming

0.84.1 (2022-11-13)
-------------------

Changes
=======

* Much faster parsing of NONMEM models
* 4 times faster parsing of NONMEM phi files

Bug fixes
=========

* Have correct F-statement in $ERROR for $DES NONMEM models 
* Read compartment names correctly when having both NCOMP and COMP in $MODEL of NONMEM models

0.84.0 (2022-11-09)
-------------------

New features
============

* Add modeling.deidentify_data

Changes
=======

* Change CLI anonymize into deidentify

0.83.0 (2022-11-01)
-------------------

Changes
=======

* Only test IOV on statements before ODE

Bug fixes
=========

* Allow spaces in DADT definitions when parsing ODE
* Fix issue where expression setter was used
* Fix issue in IIVSearch where tool doesn't continue to next step if there is a multivariate distribution
* Input dataset into model constructor in convert model (fixes #1293)
* Modelfit should not crash if .lst-file does not exist, warns if .lst and .ext-files do not exist (#1302, #1303)


0.82.0 (2022-10-24)
-------------------

Changes
=======

* modeling.summarize_modelfit_results takes results objects instead of model objects
* Do not include aic and bic in summarize_modelfit_results

0.81.1 (2022-10-24)
-------------------

Bug fixes
=========

* Handle NM-TRAN datasets with one and two digit year in DATx column using default LAST20 (50)

0.81.0 (2022-10-21)
-------------------

New features
============

* Add tools.read_modelfit_results

Changes
=======

* run_modelsearch, run_iovsearch, run_ruvsearch, run_allometry, run_covsearch, run_amd and run_iivsearch now need results as a separate input

Bug fixes
=========

* Correct scaling for F in NOMEM models for ADVAN 2,4,5,7 and 12

0.80.0 (2022-10-19)
-------------------

Changes
=======

* modeling.fit returns ModelfitResults instead of Model
* Let zero_protect default to True for modeling.set_proportional_error_model
* Faster parsing of NONMEM table files

Bug fixes
=========

* Let bioavailability parameters be part of rhs of ode_system
* Make sure initials are non-zero for absorption parameters

0.79.0 (2022-10-16)
-------------------

New features
============

* Relative paths in files, absolute paths in Python objects (#1180, fixes 887)
* Validate tool inputs (#1162, fixes #1032)
* Add allow_nested flag to add_covariate_effect (#1004)
* Add has_covariate_effect and remove_covariate_effect (#1004)
* Generalize get_rv_parameters (#1181)
* 9fd701521 Store input models in tool database
* Replace best_model with final_model_name and retrieve_final_model
* a7fbcbfe2 Handle results and databases as input to retrieve_models
* Add modeling.update_initial_individual_estimates function

Changes
=======

* Include input model as step 0 in summary_models for AMD tools
* Rename all AMD tool candidates such that modelsearch_candidate1 -> modelsearch_run1
* Add columns for number of parameters and delta parameters in summary_tool for AMD tools
* Only include chosen models in AMD summary_tool
* New names and description for COVSearch candidates
* Modify COVSearch summary_tool to include information from the steps-table, remove ranking
* New candidate descriptions in IOVSearch
* Add multiindex to RUVSearch which include step/iteration, remove ranking.
* Add algorithm column to IIVSearch summary_tool, remove algorithm from candidate name
* Compare final model in IIVSearch to input model, return input if worse
* Generalize detection of existing effects in add_covariate_effect (#1004)
* calculate_bic and calculate_aic will need the -2LL as input instead of modelfit_results
* calculate_eta_shrinkage needs the explicit arguments parameter_estimates and individual_estimates
* calculate_individual_shrinkage needs the explicit arguments parameter_estimates and individual_estimates_covariance
* check_parameters_near_bounds needs the parameter estimates given in the arguments
* check_high_correlations needs the correlation matrix as an explicit argument 
* plot_iofv_vs_iofv takes two iofv series instead of two models as input
* plot_individual_predictions takes the predictions dataframe as input
* create_joint_distribution takes an option individual_estimates argument and does not use modelfit_results directly
* evaluate_expression to get parameter estimates from optional argument instead of from modelfit_results
* evaluate_population_prediction will not take parameter estimates from modelfit_results
* evaluate_individual_prediction will not take parameter estimates from modelfit_results
* evaluate_eta_gradient will not take parameter estimates from modelfit_results
* evaluate_epsilon_gradient will not take parameter estimates from modelfit_results
* evaluate_weighted_residuals will not take parameter estimates from modelfit_results
* sample_parameters_from_covariance_matrix will need parameter_estimates and covariance_matrix as explicit arguments
* sample_parameters_uniformly will need parameter_esimtates as explicit arguments
* sample_individual_estimates will need individual_estimates and individual_estimates_covariance as explicit arguments
* calculate_individual_parameter_statistics and calculate_pk_parameters_statistics will need parameter estimates and covariance matrix
* update_inits need explicit estimates as argument and does not use modelfit_results
* update_inits does not update initial individual estimates
* Move predict_* functions from modeling to pharmpy.tools
* Move summarize_individuals and summarize_individuals_count_table to pharmpy.tools
* Move print_fit_summary to pharmpy.tools
* Move write_results to pharmpy.tools
* Move summarize_errors to pharmpy.tools
* Move rank_models to pharmpy.tools
* Move summarize_modelfit_results to pharmpy.tools
* Speedup parsing of NONMEM results

Bug fixes
=========

* 297a64041 Handle individual_ofv is None in dofv (fixes #1101)
* 57fc4fee8 Fix adding categorical covariate effects (#1004)

0.78.0 (2022-09-20)
-------------------

Changes
=======

* fd417aaf Always return a new model in convert_model
* d5458e36 Raise KeyError in LocalModelDirectory#retrieve_model (instead of FileNotFoundError)
* 1193bd39 Remove unused pharmpy.symbols submodule

Bugfixes
========

* bb96a13c Fix update_parameters when parameters are added
* 0ca786c5 Fix backward search of covsearch
* dd056da3 Fix for models with bioavailability parameters
* 915bc9c7 Fix get_config_path output when config file is disabled
* 82b32278 Remove some unwanted debug printing
* 1131a610 Fix issue in PsN SCM results parsing
* ebfafb45 Assign ODE as compartmental system to variable (#1173)

0.77.0 (2022-09-08)
-------------------

Changes
========

* fb070ee1 Return input model if allometry model fails (#1049)

0.76.1 (2022-09-06)
-------------------

Bugfixes
========

* Fix issue with adding allometry to models with MM elimination
* Make pyreadr an optional dependency, making Pharmpy easier to install on Mac M1

0.76.0 (2022-09-05)
-------------------

New features
============

* Add modeling.get_evid to get or create evid from a model
* Add modeling.get_cmt to get or create a cmt column from a model
* New column type: "rate"

Changes
=======

* Rename "resmod" tool to "ruvsearch"
* Return only DataFrame in modeling.rank_models
* Fall back to rank value if model fails in modeling.rank_models (fix #916)
* Rename "strictness" to "errors_allowed" in modeling.rank_models
* Only allow "rounding errors" by amd, iivsearch, iovsearch, modelsearch and covsearch (fix #1055)
* Add attibute significant_digits to ModelfitResults

Bugfixes
========

* Serialize modelfit results #1092
* Exlude "unreportable number of significant digits" in modeling.rank_models (fix #1076)

0.75.0 (2022-08-10)
-------------------

New features
============

* SCM forward search followed by backward search in covsearch (#988)

Changes
=======

* Change initial estimates of IIV parameters of start model in AMD tool (1c65359)
* Change default order of subtools in AMD tool (42fe72f)

Bugfixes
========

* Make NONMEM column renaming work in more cases (#1001)
* Fix issue when search spaces which lead to uneven branch length in reduced stepwise algorithm (#694)
* Fix issue with error record not numbering amounts properly with non-linear elimination (#708)
* Fix issue with comments being removed in omega blocks (#790, #974)
* Fix ranking issue when candidate models do not produce an OFV (#1017)
* Fix issue with reading datasets in AMD with RATE column (#989)

0.74.0 (2022-07-18)
-------------------

Changes
=======

* Rename `pharmpy.parameter` to `pharmpy.parameters` (71f4cf23)
* Merge COVsearch DSL into MFL (#932, #973)
* Add ZO absorption to default search space in AMD (cfc09bad)

Bugfixes
========

* Make `run_amd` work in more cases (#975)
* Make `run_iovsearch` work in more cases (#917, #977)
* Make `remove_iov` work in more cases (#917)
* Make `get_pk_parameters`/`run_covsearch` work in more cases (#908)
* Make NONMEM `.mod` parsing work in more cases (#917, #975, #977)
* Make NONMEM `.mod` updating work in more cases (fd564168)
* Make NONMEM dataset column dropping work in more cases (088a046a)
* Make ODES updates work in more cases (c76fa476, 430f1d2e)

0.73.0 (2022-06-21)
-------------------

New features
============

* Add covsearch tool
* Add function tools.retrieve_models to read in models from a tool database
* Add functions modeling.get_individual_parameters, modeling.get_pk_parameters, modeling.get_rv_parameter, and modeling.has_random_effect

Changes
=======

* Include covsearch tool in AMD
* Add results for AMD tool
* Move fit, run_tool, run_amd, and all tool wrappers from modeling module to tool module
* Rename 'diagonal' -> 'add_diagonal' in iiv strategy option for iivsearch and modelsearch tool
* Include column for selection criteria in rank_models

0.72.0 (2022-06-08)
-------------------

New features
============

* Add iovsearch tool
* Add function modeling.summarize_errors to get a summary dataframe of parsed errors from result files
* Add modeling.make_declarative
* Add modeling.cleanup_model
* Add modeling.greekify_model

Changes
=======

* Use 'no_add', 'diagonal', 'fullblock', or 'absorption_delay' instead of numbers for iiv_strategy in iivsearch and modelsearch-tool
* Add results and documentation for allometry tool
* Add error summaries to iivsearch, modelsearch, resmod, and allometry tools
* Add algorithm argument in estmethod tool ('exhaustive' and 'reduced')

Bugfixes
========

* Handle etas after ODEs in iivsearch-tool

0.71.0 (2022-05-24)
-------------------

New features
============

* Add functions find_clearance_parameters and find_volume_parameters

Changes
=======

* Rename candidate models in estmethod tool

Bugfixes
========

* Add upper limit to VP parameter in modelsearch tool
* Fix issue with matrices not being considered positive semidefinite but considered positive definite


0.70.1 (2022-05-17)
-------------------

Bugfixes
========

* Require pandas 1.4 or newer for multiindex joins. (Fixes #820)

0.70.0 (2022-05-13)
-------------------

New features
============

* New tool allometry added
* Add modeling.summarize_individuals_count_table
* Add modeling.calculate_ucp_scale
* Add modeling.calculate_parameters_from_ucp
* Add description attribute to model objects
* Add wrappers for iivsearch and modelsearch tools (run_iivsearch and run_modelsearch)
* Add documentation for iivsearch tool

Changes
=======

* resmod can now iterate and add multiple residual error models
* Automatically generate R examples in a seprate tab in documentation
* Merge iiv functions into one iivsearch algorithm ('brute_force')
* Use parameter names instead of eta names in iivsearch tool features

Bugfixes
========

* Make sure dropping of DATE columns in NONMEM models are handled correctly
* Solve issue with sporadic crashes because of a database race condition
* Solve issue with sporadic crashes caused by race in lazy parsing of NONMEM records
* Fix issues with converting some piecewise functions to NONMEM code correctly
* Fix issue with generating candidate models for iivsearch tool
* Fix issue with duplicate candidate models in iivsearch tool (#745)

0.69.0 (2022-04-29)
-------------------

New features
============

* Add 1st and 3rd quantiles of residual to simeval results

Changes
=======

* Rename mfl -> search_space in modelsearch and amd
* Use BIC as default ranking function in modelsearch
* Start model in modelsearch is not fitted
* Update modelsearch documentation

Bugfixes
========

* Fix bad odes when adding two peripheral compartments to model with MM elimination
* Fix bug in block splitting in IIV-tool (fixes #745)

0.68.0 (2022-04-27)
-------------------

Bugfixes
========

* Fix bad odes when adding peripheral compartment to model with MM elimination (fixes #710)

0.67.0 (2022-04-25)
-------------------

New features
============

* Add modeling.get_thetas, modeling.get_omegas and modeling.get_sigmas
* Add configuration option for NONMEM license file path

Bugfixes
========

* Correct parsing of ADVAN=ADVANx in $SUBROUTINES in NONMEM models
* Fix issue with duplicated TAD in $INPUT after add_time_after_dose
* Fix issue with not being able to use models with assignments in $DES in estmethod tool
* Set an upper limit for intercompartmental clearances in the modelsearch tool (fixes #695)

0.66.0 (2022-04-20)
-------------------

Bugfixes
========

* Fix NONMEM model parsing issue causing ADVAN not to change for models with DEFOBS in $MODEL

0.65.0 (2022-04-14)
-------------------

New features
============

* Add option in `add_iiv` and `add_pk_iiv` to choose initial estimate

Changes
=======

* Replace different iiv-options in IIV-tool with `iiv_strategy`
* Use 0.01 as initial estimate for added IIVs in modelsearch tool

Bugfixes
========

* Add K-parameters in NONMEM model when changing to general linear (GL) solvers

0.64.0 (2022-04-12)
-------------------

New features
============

* Add modeling.solve_ode_system
* Add documentation for .datainfo file
* Add iofv plot to linearize results
* Store tool meta data in metadata.json

Changes
=======

* New options for modelsearch tool: switch order of mfl and algorithm, replace different iiv-options with `iiv_strategy`

0.63.0 (2022-04-07)
-------------------

New features
============

* Support ~ as HOME in file paths input by users
* Add modeling.read_dataset_from_datainfo
* Store unique datasets for tool runs

Bugfixes
========

* Fix problem with TAD calculation for datasets with ADDL
* Handle LinAlgError when updating initial estimates in modelsearch (#656)

0.62.0 (2022-04-04)
-------------------

New feature
===========

* Store unique datasets in models/.datasets

Changes
=======

* New name for final model in resmod

Bugfixes
========

* Use NaN in summary_individuals if tflite cannot be used

0.61.1 (2022-03-31)
-------------------

Bugfixes
========

* Fix time after dose calculation for steady state dosing
* Fix issue where create_joint_distribution could create matrices that are not positively definite (#649)
* Keep IIV from MAT in MDT when adding a transit (#654)

0.61.0 (2022-03-29)
-------------------

New features
============

* Add modeling.summarize_individuals

Changes
=======

* Change initial estimates for QP1/QP2 ratio to 0.1/0.9

Bugfixes
========

* Handle ADDL columns for add_time_after_dose

0.59.0 (2022-03-25)
-------------------

New features
============

* Add modeling.expand_additional_doses


0.58.4 (2022-03-24)
-------------------

Bugfixes
========

* Fix issue with start model not being selected if no candidates are better in IIV- and modelsearch-tool
* Fix issue with ranking models by dBIC in IIV-tool


0.58.1 (2022-03-22)
-------------------

Bugfixes
========

* Fix ordering of TAD values for dose at some time as observation
* Fix TAD values for datasets with reset time event
* Handle models with no covariates for predict_outliers and predict_influential_individuals

0.58.0 (2022-03-22)
-------------------

New features
============

* Add modeling.add_pk_iiv to add iiv to all pk parameters of a model

Changes
=======

* Change cutoff for zero protection in proportional error ModelSyntaxError
* Change to checking for positive semidefiniteness instead of only positive definiteness when validating omegas

Bugfixes
========

* Fix BIC-mixed calculation to not count thetas related to non-random etas (0 FIX) towards random parameters
* Read 0 FIX diagonal etas as random variables


0.57.0 (2022-03-21)
-------------------

Bugfixes
========

* Keep thetas/etas when going across absorption transformations (#588, #625)
* Fix missing ALAG-parameter in non-linear elimination (#578)
* Fix issue with added VC1-parameter when adding transits to non-linear elimination (#577)
* Fix missing D1-parameter and RATE-column when adding zero order absorption to non-linear elimination (#578)
* Only do update_inits if start model was successful in IIV-tool (#632)
* Fix issue where etas where added to KA/K-parameters instead of MAT/MDT (#636)

0.56.0 (2022-03-17)
-------------------

Changes
=======

* Remove ZO elimination from the default search space in model search
* Do not apply resmod mode if no change on the full model

Bugfixes
========

* Fix bad calculation of number of observations for datasets with both EVID and MDV
* Properly handle observations and dose at same time for time after dose calculation
* Handle DATE column for time after dose calculation
* Handle NONMEM models with no ETAs


0.55.0 (2022-03-16)
-------------------

New features
============

* Option in modeling.update_inits to move estimates that are close to boundary

Changes
=======

* Set different initial estimates of clearance for peripherals (#590)


Bugfixes
========

* Fix issue with duplicated features with IIV-options in modelsearch-tool
* Fix issue where $MODEL was not added when setting ODE solver to GL or GL_REAL
* Fix issue where reduced_stepwise failed for certain search spaces (#616)
* Fix issue with reading in sampled_iofv in simeval
* Use the same time varying cutoff for resmod models and best model

0.54.0 (2022-03-08)
-------------------

New features
============

* New IIV version of BIC in calculate_bic
* Use IIV BIC in iiv tool
* Add allometry step in amd tool
* Reduced stepwise algorithm in modelsearch
* Add cutoff option to predict_outliers etc

Bugfixes
========

* Fix issue with failing to creating correct subblocks of fullblock of random variables
* Set index name to id-name in predict_outliers

0.53.0 (2022-03-04)
-------------------

New features
============

* Add option to remove specific IOV random variables
* Support Python 3.10
* Add modeling.check_dataset

Changes
=======

* modeling.calculate_bic: Count epsilons interacting with etas to random parameters
* Updated tensorflow models for prediction of outliers and influential individuals
* Only consider parameters with etas for covariate modelbuilding in amd
* Include AIC/BIC in modeling.summarize_modelfit_results

Bugfixes
========

* Update solvers in estmethod-tool
* Handle Q-parameters when adding IIV on structural parameters in iiv-tool
* Only add IIV on MDT-parameter with add_mdt_iiv-option in modelsearch-tool

0.52.0 (2022-02-25)
-------------------

New features
============

* Add covariate search to amd tool

0.51.0 (2022-02-24)
-------------------

New features
============

* Add option to add IIV to start model in iiv-tool
* Add solver option in estmethod-tool
* Add option to add IIV only on MDT in modelsearch-tool


Changes
=======

* | modeling.calculate_bic can Calculate three different versions of the BIC
  | default has switched to be a mixed effects version
* Remove etas instead of setting to 0 fix in iiv-tool
* Parse more errors and warnings in .lst-file
* Rename add_eta -> add_iivs, etas_as_fullblock -> iiv_as_fullblock, add_mdt_eta -> add_mdt_iiv in modelsearch

0.50.1 (2022-02-16)
-------------------

Bugfixes
========

* Handle long paths when fitting NONMEM model

0.50.0 (2022-02-16)
-------------------

New features
============

* Add modeling.write_results
* Add modeling.print_fit_summary
* Add modeling.remove_loq_data
* Add first version of WIP scm wrapper

Changes
=======

* Change in mfl in modelsearch such that transits that don't keep depot will have additional transit
* Make it possible to set $DATA directly for NONMEM models (via datainfo.path) (#130)

0.49.0 (2022-02-10)
-------------------

New features
============

* Add modeling.calculate_se_from_cov
* Add modeling.calculate_se_from_inf
* Add modeling.calculate_corr_from_cov
* Add modeling.calculate_cov_from_inf
* Add modeling.calculate_cov_from_corrse
* Add modeling.calculate_inf_from_cov
* Add modeling.calculate_inf_from_corrse
* Add modeling.calculate_corr_from_inf
* Add modeling.create_report
* Add modeling.check_high_correlations
* Add modeling.calculate_bic
* Add modeling.check_parameters_near_bounds
* Add option to choose search space in AMD-tool

Changes
=======

* Use p-value instead of OFV cutoff in resmod

Bugfixes
========

* Fix issue with no conversion to $DES for some models (#528)

0.48.0 (2022-02-04)
-------------------

New features
============

* Parse estimation step runtime from NONMEM results file

Changes
=======

* Force initial estimates when reading model file to be positive definite

Bugfixes
========

* Random block was not split properly in some cases when random variable was removed
* Add $COV correctly in NM-TRAN models (#457)


0.47.0 (2022-01-28)
-------------------

* Add modeling.drop_columns
* Add modeling.drop_dropped_columns
* Add modeling.undrop_columns
* Add modeling.translate_nmtran_time


0.46.0 (2022-01-27)
-------------------

* Add modeling.calculate_aic
* Add modeling.print_model_code
* Add modeling.has_michaelis_menten_elimination
* Add modeling.has_zero_order_elimination
* Add modeling.has_first_order_elimination
* Add modeling.has_mixed_mm_fo_elimination
* Add parent_model attribute to Model object
* Support non-linear elimination in search space in modelsearch tool
* Rename summary -> summary_tool in IIV and modelsearch tool, add summary_models
* Update modelsearch algorithm to only run 2C if previous model is 1C
* Fix bug in transformation order in features column of summary in modelsearch tool

0.45.0 (2022-01-21)
-------------------

* Add timevarying models to resmod

0.44.0 (2022-01-20)
-------------------

* Add modeling.create_symbol
* Add modeling.remove_unused_parameters_and_rvs
* Add modeling.mu_reference_model
* Add modeling.simplify_expression
* Add option keep_depot to modeling.set_transit_compartments
* Add CLI for estmethod tool
* Add attributes isample, niter, auto and keep_every_nth_iter to EstimationStep
* Remove stepwise algorithm in modelsearch tool

0.43.0 (2022-01-12)
-------------------

* Add modeling.bump_model_number
* Fix regression in detection of dv column when synonym was used

0.42.0 (2022-01-11)
-------------------

* Add modeling.get_doseid
* Add modeling.get_unit_of
* Add modeling.get_concentration_parameters_from_data
* Add modeling.write_csv
* Add modeling.resample_data
* Add modeling.omit_data
* Add modeling.get_observation_expression
* Add modeling.get_individual_prediction_expression
* Add modeling.get_population_prediction_expression
* Add modeling.evaluate_individual_prediction
* Add modeling.evaluate_population_prediction
* Add modeling.calculate_eta_gradient_expression
* Add modeling.calculate_epsilon_gradient_expression
* Add modeling.evaluate_eta_gradient
* Add modeling.evaluate_epsilon_gradient
* Add modeling.evaluate_weighted_residuals
* Support for Python 3.7 dropped

0.41.0 (2021-12-21)
-------------------

* Add modeling.get_individuals
* Add modeling.get_baselines
* Add modeling.get_covariate_baselines
* Add modeling.get_doses
* Add modeling.list_time_varying_covariates
* Add combined error model to resmod
* Add option to zero_protect to set_proportional_error_model
* Add tool estmeth to find optimal estimation method for a model
* Fix bug causing resmod models to be incorrect
* New model.datainfo object

0.40.0 (2021-12-16)
-------------------

* Add modeling.add_allometry

0.39.0 (2021-12-15)
-------------------

* Add AMD and IIV tool and respective functions run_amd and run_iiv
* Add function add_covariance_step and remove_covariance_step
* Add method insert_after to ModelStatements
* Add option to set limit or no limit for power_on_ruv theta
* Rename EstimationMethod to EstimationStep and add EstimationSteps class
* Parse eta and epsilon derivatives from $TABLE
* Fix bug where lag time is removed when changing to ZO or FO absorption

0.38.0 (2021-12-08)
-------------------

* Add function to get path to user configuration file
* Add function to get missing DVs
* Add option to add IIV on structural parameters (as diagonal and block)
* Add guard for log(0) in proportional error for log data
* Avoid crash if plots cannot be created in CDD results
* Fix issue saving modelsearch results
* Fix bipp issues with etas outside of FREM matrix

0.37.1 (2021-11-26)
-------------------

* Fix bug causing frem report to crash with #IDs > 5000
* Fix bug for shifted uncertainty in frem with bipp

0.37.0 (2021-11-24)
-------------------

* First version of IIV-tool
* Rename set_lag_time to add_lag_time
* Include run type in summarize_modelfit_results
* Fix bug with force option in write_model
* Fix bug in parsing .ext-files with tables without header
* Fix bug with nested update_source crashing due to incorrect handling of diagonal records
* Fix bug with inserted IGNORE on comment lines

0.36.0 (2021-11-11)
-------------------

* Add option to set_dtbs_error_model to fix parameters to 0 (i.e. get data on log-scale)
* Create model file when fitting a model that has no model file
* Fix bug where files are missing during e.g. modelsearch
* Fix crash when including a model with no results in summarize_modelfit_results
* Fix bug in Pharmr where integers where interpreted as floats
* Fix issue with extra IPRED on power_on_ruv model

0.35.0 (2021-11-02)
-------------------

* Option to include all estimation steps in summarize_modelfit_results
* Use kwargs in set_estimation_step and add_estimation_step
* First version of logger (via model.modelfit_results.log)

0.34.3 (2021-10-28)
-------------------

* Let parametrization of peripheral compartment rates be kept if volume parameter can be found in the expression for K.
* Fix bug causing crashes when parsing some lst-files due to mixed encodings.

0.34.2 (2021-10-26)
-------------------

* Fix broken parallelization for tools (workflows)
* Fix bug causing parsing of some NM-TRAN datasets to set a column index

0.34.1 (2021-10-20)
-------------------

* Fix issues with retrieving results after model runs

0.34.0 (2021-10-14)
-------------------

* Remove the need for update_source. Instead use model.model_code or modeling.generate_model_code(model)
* str(model) can no longer be used to get the model_code
* Fix crash in model database when using copies of models

0.33.0 (2021-10-11)
-------------------

* Add modeling.read_model_from_database
* Add modeling.print_model_symbols
* Add modeling.append_estimation_step_options
* Fix crash for $DES models with RATE in dataset
* Fix estimation status for evaluation steps to use latest estimation

0.32.0 (2021-09-28)
-------------------

* Move plot_iofv_vs_iofv to modeling
* Add modeling.get_observations
* Add modeling.plot_individual_predictions

0.31.0 (2021-09-21)
-------------------

* Move parameter_sampling-functions into modeling module
* Add run_tool function to modeling
* Add predict_outliers, predict_influential_individuals and predict_influential_outliers functions to modeling
* Update API documentation (e.g. add examples, and improved index)

0.30.0 (2021-09-06)
-------------------

* Add modeling.load_example_model
* Move eta_shrinkage results method to modeling.calculate_eta_shrinkage
* Add first version of resmod tool
* Update documentation (including API reference)
* Rename summarize_models to summarize_modelfit_results
* Fix bug related in running NONMEM on Windows via Rstudio

0.29.0 (2021-08-25)
-------------------

* Rename zero_order_absorption to set_zero_order_absorption
* Rename first_order_absorption to set_first_order_absorption
* Rename bolus_absorption to set_bolus_absorption
* Rename seq_zo_fo_absorption to set_seq_zo_fo_absorption
* Rename have_zero_order_absorption to has_zero_order_absorption
* Rename power_on_ruv to set_power_on_ruv
* Rename add_lag_time to set_lag_time
* Move individual_shrinkage results method to modeling.calculate_individual_shrinkage

0.28.0 (2021-08-24)
-------------------

* Move method individual_parameter_statistics from Results to a function in modeling and rename to calculate_individual_parameter_statistics
* Move method pk_parameters from Results to a function in modeling and rename to calculate_pk_parameters_statistics
* Rename create_rv_block to create_joint_distribution
* Rename split_rv_block to split_joint_distribution
* New default option force=True for write_model
* Rename ninds to get_number_of_individuals
* Rename nobs to get_number_of_observations
* Rename nobsi to get_number_of_observations_per_individual
* Rename remove_error to remove_error_model
* Rename additive_error to set_additive_error_model
* Rename proportional_error to set_proportional_error_model
* Rename combined_error to set_combined_error_model
* Rename has_additive_error to has_additive_error_model
* Rename has_proportional_error to has_proportional_error_model
* Rename has_combined_error to has_combined_error_model
* Rename theta_as_stdev to use_thetas_for_error_stdev
* Rename set_dtbs_error to set_dtbs_error_model
* Rename boxcox to transform_etas_boxcox
* Rename tdist to transform_etas_tdist
* Rename john_draper to transform_etas_john_draper
* Rename iiv_on_ruv to set_iiv_on_ruv
* Rename add_parameter to add_individual_parameter
* Rename first_order_elimination to set_first_order_elimination
* Rename zero_order_elimination to set_zero_order_elimination
* Rename michaelis_menten_elimination to set_michaelis_menten_elimination
* Rename mixed_mm_fo_elimination to mixed_mm_fo_elimination
* Function summarize_models to create a summary of models
* Parse total runtime
* Revert to dask distributed

0.27.0 (2021-08-09)
-------------------

* Use dask threaded for Windows, allow configuration of dispatcher type
* Filter out individuals without observations in .phi-file

0.26.1 (2021-08-04)
-------------------

* Correct residual calculation in simeval
* Correct how laplace estimation method is written

0.26.0 (2021-07-13)
-------------------

* Add functions to set, add, and remove estimation step
* Add supported estimation methods (ITS, LAPLACE, IMPMAP, IMP, SAEM)
* When updating estimation step, old options are kept

0.25.1 (2021-07-08)
-------------------

* Read site path if user path doesn't exist (previously read user path)
* Change return type of covariates to a list for easier handling in R

0.25.0 (2021-06-24)
-------------------

* Add modeling.ninds, nobs and nobsi to get number of individuals and observations of dataset
* Add reading results for resmod and crossval
* Add structural bias, simeval and resmod results to qa results
* Update index of cdd case_results to plain numbers
* Support line continuation (&) in NM-TRAN code
* Fix error in calculation of sdcorr form of parameter estimates
* Fix crash of cdd results retrieval
* Various fixes for running NONMEM models

0.24.0 (2021-05-25)
-------------------

* Added theta_as_stdev, set_weighted_error_model and set_dtbs_error
* Error models can be added with log transformed DV using `data_trans` option
* Added model attributes data_transformation and observation_transformation
* Protected functions in NM-TRAN translated to Piecewise. Should now give the
  same result as when evalutated by NONMEM.
* Bugfixes for frem, scm and bootstrap results generation
* Rename model attribute dependent_variable_symbol to dependent_variable
* Added simplify method on Parameter class to simplify expressions given parameter constraints

0.23.4 (2021-05-03)
-------------------

* 10-100 times Speedup of modeling.evaluate_expression

0.23.3 (2021-04-29)
-------------------

* Documentation fix for pharmr release
* Handle implicit ELSE clauses for NM-TRAN IF

0.23.2 (2021-04-28)
-------------------

* Fix bug #177


0.23.1 (2021-04-28)
-------------------

* Bugfixes

0.23.0 (2021-04-28)
-------------------

* Add function modeling.evaluate_expression
* Some documentation for modelfit_results
* Reworked interface to RandomVariables and Parameters
* Bugfixes

0.22.0 (2021-03-29)
-------------------

* Support COM(n) in NONMEM abbreviated code
* Fix stdin handling issue when running NONMEM from R

0.21.0 (2021-03-22)
-------------------

*  New function `read_results` in modeling
*  Add method to convert ExplicitODESystem to CompartmentalSystem
*  Support running NONMEM 7.3 and 7.5
*  Bugfixes:

   * Allow protected functions in NONMEM abbreviated code
   * Fix bad rates when changing number of transit compartments (#123)

0.20.1 (2021-03-11)
-------------------

* Fix regression for calling NONMEM

0.20.0 (2021-03-11)
-------------------

* New function modeling.set_peripheral_compartments
* New tool Model Search
* New model attribute `estimation_steps` to read and change $ESTIMATION
* Bugfixes (#99, #118)

0.19.0 (2021-03-02)
-------------------

* Add create_result to create results from PsN
* Add documentation for covariate effects

0.18.0 (2021-03-01)
-------------------

* Add functions to fix and unfix values to a specified value
* Add documentation for using Pharmpy with NONMEM models
* New execution system for modelfit
* Support for single string input for transformations of etas and epsilons (e.g. add_iov)
* Various bugfixes, including running NONMEM via Pharmpy on Windows

0.17.0 (2021-02-15)
-------------------

* Add function to split an eta from a block structure
* New names for covariance between etas in create_rv_block
* Clearer error messages when adding IOVs (if only one level of occasion) and for parameter_names config

0.16.0 (2021-02-08)
-------------------

* Improve initial estimates for adding peripheral compartments
* Parameter names are set according to priority in config
* Avoid duplication of e.g. median/mean when having multiple covariate effects with the same covariate
* Change assignments when multiple covariate effects are applied to the same parameter to be combined in one line
* Do not change error model if it is the same error model transformation multiple times
* Add AggregatedModelfitResults
* Document scm results

0.15.0 (2021-02-01)
-------------------

* Change parameter_names config option to be a list of prioritized methods
* Option to read names from $ABBR for NONMEM models
* Add option to give parameter names to methods.add_iiv
* Add calculation of elimination half-life to one comp models in modelfit_results.pk_parameters
* Document cdd results
* Add set_initial_estimates, set_name and copy_model to modeling
* Allow single str as input to add_iiv and add_iov

0.14.0 (2021-01-25)
-------------------

* Support reading $DES-records
* Add individual_parameter_statistics to ModelfitResults
* Add pk_parameters to ModelfitResults
* Add add_iov to modeling
* Rename add_etas -> add_iiv

0.13.0 (2021-01-18)
-------------------

* Change names of covariate effect parameters for add_covariate_effects
* Improve ordering of terms in created NONMEM expressions
* Add parameter_inits, base_parameter_change, parameter_variability and coefficients to frem_results
* Add SimevalResults class
* Add fit and read_model_from_string functions to modeling
* Add solver attribute to ODESystem to be able to select ODE-system solver. Currently ADVANs for NONMEM
* New method nonfixed_inits to ParameterSet
* Add residuals attribute to ModelfitResults
* Various bug fixes
* Migrate to github actions for continuous integration

0.12.0 (2020-12-18)
-------------------

* Add modeling.update_inits, modeling.add_peripheral_compartment and modeling.remove_peripheral_compartment
* Update FREM documentation
* Switch to using modelled covariate values for baselines in FREM
* Add methods for retrieving doses and Cmax, Tmax, Cmin and Tmin from dataset
* Various bugfixes and support for more ADVAN/TRANS combinations

0.11.0 (2020-11-20)
-------------------

* Method df.pharmpy.observations to extract observations from dataframe
* Add ColumnTypes EVENT and DOSE
* Add model.to_base_model to convert model to its raw base model form
* New functions in modeling: remove_iiv, zero_order_elimination,
  comined_mm_fo_elimination and add_parameter
* Split modeling.absorption_rate and error into multiple functions
* Add calculations of AIC and BIC to ModelfitResults
* Improved pretty printing

0.10.0 (2020-11-16)
-------------------

* modeling.create_rv_block
* modeling.michaelis_menten_elimination
* modeling.set_transit_compartments
* First version of modelfit method
* Add first version of bootstrap method
* Add parameter estimates histograms to bootstrap report
* Add automatic update of $SIZES PD when writing/updating NONMEM model
* Additions to QAResults
* NMTRanParseError replaced with ModelSyntaxError
* Multiple bugfixes to frem and scm result calculations

0.9.0 (2020-10-26)
------------------

* Add error_model function to the modeling module
* Added more standard models for modeling.add_etas
* Improve BootstrapResults
* Add plots to bootstrap
* Add support for the PHARMPYCONFIGPATH environment variable
* Add QAResults and LinearizeResults classes
* Bugfixes for some Windows specific issues

0.8.0 (2020-10-08)
------------------

* Add basic modeling functions to the modeling module
* modeling.add_etas
* Improved bootstrap results generation and additional plots
* Bugfix: Labelled OMEGAS could sometimes get wrong symbol names

0.7.0 (2020-09-28)
------------------

* Add method reset_indices in Results to flatten multiindices. Useful from R.
* absorption_rate can also set sequential zero first absorption
* New functionsadd_lag_time and remove_lag_time in modeling
* Add basic functions fix/unfix_parameter, update_source and read_model to modeling API
* Updated reading of NONMEM results
* Bugfixes in add_covariate_effects and absorption_rate
* Fix crash in FREM results if no log option could be found in meta.yaml

0.6.0 (2020-09-18)
------------------

* Add eta transformations: boxcox, t-dist and John Draper
* Add results cdd and scm to CLI
* Add different views for scm results
* Add support for taking parameter names from comment in NONMEM model
* Remove assumptions for symbols
* Add modeling.absorption_rate to set 0th or first order absorption
* Add update of $TABLE numbers

0.5.0 (2020-09-04)
------------------

* Many bugfixes and improvements to NONMEM code record parser
* Add calculation of symbolic and numeric eta and eps gradients, population and individulal prediction and wres for PRED models
* Add option to use comments in NONMEM parameter records as names for parameters
* Reading of ODE systems from NONMEM non-$DES models
* Calculation of compartmental matrix and ODE system
* New module 'modeling'
* Function in modeling and CLI to change ADVAN implicit compartmental models to explicit $DES
* Function in modeling and CLI to add covariate effects
* Functions for reading cdd and scm results from PsN runs
* Many API updates
* Extended CLI documentation

0.4.0 (2020-07-24)
------------------

* Add categorical covariates to covariate effects plot in FREM
* Better support for reading NONMEM code statements (PK and PRED)
* Support for updating NONMEM code statements (PK and PRED)
* Bugfixes for CLI


0.3.0 (2020-06-16)
------------------

* New CLI command 'data append'
* Parameter names is now the index in Parameters.summary()
* FREM postprocessing
* Standardized results.yaml and results.csv

0.2.0 (2020-03-27)
------------------

First release


0.1.0 (2018-07-22)
------------------

Initial library development/testing directory structure.
