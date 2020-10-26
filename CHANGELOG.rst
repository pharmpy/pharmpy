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
