========
Modeling
========

.. currentmodule:: pharmpy.modeling

General functions
-----------------

.. autosummary::
   :toctree: api/

    bump_model_number
    convert_model
    create_config_template
    filter_dataset
    get_config_path
    get_model_code
    get_model_covariates
    get_nested_model
    load_example_model
    print_model_code
    print_model_symbols
    read_model
    read_model_from_string
    remove_unused_parameters_and_rvs
    rename_symbols
    set_description
    set_name
    write_csv
    write_dataset
    write_model

Creating basic models
---------------------

.. autosummary::
   :toctree: api/

    create_basic_kpd_model
    create_basic_pd_model
    create_basic_pk_model

Dataset handling
----------------

.. autosummary::
   :toctree: api/

    add_admid
    add_cmt
    add_time_after_dose
    add_time_of_last_dose
    bin_observations
    binarize_dataset
    deidentify_data
    drop_columns
    drop_dropped_columns
    expand_additional_doses
    is_binary
    load_dataset
    read_dataset_from_datainfo
    remove_loq_data
    remove_unused_columns
    set_covariates
    set_dataset
    set_dvid
    set_lloq_data
    set_reference_values
    translate_nmtran_time
    undrop_columns
    unload_dataset

Dataset exploration
--------------------

.. autosummary::
   :toctree: api/

    calculate_summary_statistic
    check_dataset
    get_admid
    get_baselines
    get_cmt
    get_column_name
    get_concentration_parameters_from_data
    get_covariate_baselines
    get_doseid
    get_doses
    get_evid
    get_ids
    get_mdv
    get_number_of_individuals
    get_number_of_observations
    get_number_of_observations_per_individual
    get_observations
    get_unit_of
    infer_datatypes
    list_time_varying_covariates

Parameters
----------

.. autosummary::
   :toctree: api/

    add_population_parameter
    fix_or_unfix_parameters
    fix_parameters
    fix_parameters_to
    get_omegas
    get_sigmas
    get_thetas
    map_eta_parameters
    replace_fixed_thetas
    set_initial_estimates
    set_lower_bounds
    set_upper_bounds
    unconstrain_parameters
    unfix_parameters
    unfix_parameters_to

Parameter variability
---------------------

.. autosummary::
   :toctree: api/

    add_iiv
    add_iov
    add_pd_iiv
    add_pk_iiv
    create_joint_distribution
    remove_iiv
    remove_iov
    split_joint_distribution
    transform_etas_boxcox
    transform_etas_john_draper
    transform_etas_tdist
    update_initial_individual_estimates

Random variables
----------------

.. autosummary::
   :toctree: api/

    replace_non_random_rvs

Covariate effects
-----------------

.. autosummary::
   :toctree: api/

    add_allometry
    add_covariate_effect
    get_covariate_effects
    has_covariate_effect
    remove_covariate_effect

Error model
-----------

.. autosummary::
   :toctree: api/

    has_additive_error_model
    has_combined_error_model
    has_proportional_error_model
    has_weighted_error_model
    remove_error_model
    set_additive_error_model
    set_combined_error_model
    set_dtbs_error_model
    set_iiv_on_ruv
    set_power_on_ruv
    set_proportional_error_model
    set_time_varying_error_model
    set_weighted_error_model
    transform_blq
    use_thetas_for_error_stdev

PK modeling
-----------

.. autosummary::
   :toctree: api/

    add_bioavailability
    add_individual_parameter
    add_lag_time
    add_peripheral_compartment
    find_clearance_parameters
    find_volume_parameters
    get_bioavailability
    get_central_volume_and_clearance
    get_lag_times
    get_number_of_peripheral_compartments
    get_number_of_transit_compartments
    has_first_order_absorption
    has_first_order_elimination
    has_instantaneous_absorption
    has_michaelis_menten_elimination
    has_mixed_mm_fo_elimination
    has_seq_zo_fo_absorption
    has_weibull_absorption
    has_zero_order_absorption
    has_zero_order_elimination
    remove_bioavailability
    remove_lag_time
    remove_peripheral_compartment
    set_first_order_absorption
    set_first_order_elimination
    set_instantaneous_absorption
    set_michaelis_menten_elimination
    set_mixed_mm_fo_elimination
    set_n_transit_compartments
    set_peripheral_compartments
    set_seq_zo_fo_absorption
    set_transit_compartments
    set_weibull_absorption
    set_zero_order_absorption
    set_zero_order_elimination

PD modeling
-----------

.. autosummary::
   :toctree: api/

    add_effect_compartment
    add_indirect_effect
    add_placebo_model
    set_baseline_effect
    set_direct_effect

Other model types
-----------------

.. autosummary::
   :toctree: api/

    add_metabolite
    has_presystemic_metabolite
    set_tmdd

ODEs
----

.. autosummary::
   :toctree: api/

    display_odes
    get_initial_conditions
    get_zero_order_inputs
    has_linear_odes
    has_linear_odes_with_real_eigenvalues
    has_odes
    set_initial_condition
    set_ode_solver
    set_zero_order_input
    solve_ode_system

Estimation steps
----------------

.. autosummary::
   :toctree: api/

    add_derivative
    add_estimation_step
    add_parameter_uncertainty_step
    add_predictions
    add_residuals
    append_estimation_step_options
    is_simulation_model
    remove_derivative
    remove_estimation_step
    remove_parameter_uncertainty_step
    remove_predictions
    remove_residuals
    set_estimation_step
    set_evaluation_step
    set_simulation

Results
-------

.. autosummary::
   :toctree: api/

    calculate_aic
    calculate_bic
    calculate_eta_shrinkage
    calculate_individual_parameter_statistics
    calculate_individual_shrinkage
    calculate_pk_parameters_statistics
    check_high_correlations
    check_parameters_near_bounds
    insert_ebes_into_dataset

Plots
-----

.. autosummary::
   :toctree: api/

    plot_abs_cwres_vs_ipred
    plot_cwres_vs_idv
    plot_dv_vs_ipred
    plot_dv_vs_pred
    plot_eta_distributions
    plot_individual_predictions
    plot_iofv_vs_iofv
    plot_transformed_eta_distributions
    plot_vpc

Parameter sampling
------------------

.. autosummary::
   :toctree: api/

    create_rng
    sample_individual_estimates
    sample_parameters_from_covariance_matrix
    sample_parameters_uniformly

Expressions
-----------

.. autosummary::
   :toctree: api/

    calculate_epsilon_gradient_expression
    calculate_eta_gradient_expression
    cholesky_decompose
    cleanup_model
    create_symbol
    get_dv_symbol
    get_individual_parameters
    get_individual_prediction_expression
    get_mu_connected_to_parameter
    get_observation_expression
    get_parameter_rv
    get_pd_parameters
    get_pk_parameters
    get_population_prediction_expression
    get_rv_parameters
    greekify_model
    has_mu_reference
    has_random_effect
    is_linearized
    is_real
    make_declarative
    mu_reference_model
    simplify_expression

Estimation & evaluation
-----------------------

.. autosummary::
   :toctree: api/

    calculate_parameters_from_ucp
    calculate_ucp_scale
    evaluate_epsilon_gradient
    evaluate_eta_gradient
    evaluate_expression
    evaluate_individual_prediction
    evaluate_population_prediction
    evaluate_weighted_residuals

Math functions
--------------

.. autosummary::
   :toctree: api/

    calculate_corr_from_cov
    calculate_corr_from_prec
    calculate_cov_from_corrse
    calculate_cov_from_prec
    calculate_prec_from_corrse
    calculate_prec_from_cov
    calculate_se_from_cov
    calculate_se_from_prec

Iterators
---------

.. autosummary::
   :toctree: api/

    omit_data
    resample_data
