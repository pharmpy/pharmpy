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
