from pharmpy.modeling import get_model_covariates, get_omegas, get_sigmas


def check_input_model(model):
    # Check if input model is supported and can be estimated
    # raise otherwise
    nests = len(model.execution_steps)

    if nests == 0:
        raise ValueError("No estimation step specified in model")

    if nests > 1:
        raise ValueError(
            f"Only one estimation step is currently supported. Found {nests} in model."
        )

    step = model.execution_steps[0]

    if step.method != "FO":
        raise ValueError(
            f"Currently only the FO estimation method is supported. {step.method} was specified."
        )

    if model.statements.ode_system is not None:
        raise ValueError("Currently models having ODEs cannot be estimated.")

    ndvs = len(model.dependent_variables)
    if ndvs > 1:
        raise ValueError(f"Currently only one DV is supported. Found {ndvs} in model.")

    ncovs = len(get_model_covariates(model))
    if ncovs > 0:
        raise ValueError(
            f"Found {ncovs} covariates in model. Currently covariates are not supported."
        )

    omsi_symbols = {p.symbol for p in get_omegas(model) + get_sigmas(model)}
    if model.statements.free_symbols.intersection(omsi_symbols):
        raise ValueError(
            "Found omegas and/or sigmas in model statements. This is currently not supported."
        )
