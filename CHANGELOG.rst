next version
------------

New features
============

* Add modeling.get_evid to get or create evid from a model
* Add modeling.get_cmt to get or create a cmt column from a model

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
