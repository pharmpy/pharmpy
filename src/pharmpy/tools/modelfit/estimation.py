import time
from functools import partial

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import symengine
from pharmpy.deps.scipy import linalg
from pharmpy.modeling import cleanup_model, get_thetas, remove_unused_columns
from pharmpy.tools.modelfit.evaluation import SymengineSubsEvaluator
from pharmpy.tools.modelfit.input import check_input_model
from pharmpy.tools.modelfit.ucp import (
    build_initial_values_matrix,
    build_parameter_coordinates,
    build_starting_ucp_vector,
    calculate_gradient_scale,
    descale_matrix,
    descale_thetas,
    get_parameter_symbols,
    scale_matrix,
    scale_thetas,
    split_ucps,
)
from pharmpy.workflows import ModelEntry, ModelfitResults


class EstimationState:
    pass


def build_matrix_gradients(coords):
    # Build dA/dx for each matrix parameter x
    n = coords[-1][0] + 1
    grads = []
    for row, col in coords:
        D = np.zeros((n, n))
        D[row, col] = 1.0
        grads.append(D)
    return grads


def build_zero_gradients(coords, n):
    size = coords[-1][0] + 1
    A = np.zeros((size, size))
    zeros = [A] * n
    return zeros


def build_parameter_symbolic_gradients(nthetas, omega_coords, sigma_coords):
    omegas = (
        build_zero_gradients(omega_coords, nthetas)
        + build_matrix_gradients(omega_coords)
        + build_zero_gradients(omega_coords, len(sigma_coords))
    )
    sigmas = build_zero_gradients(
        sigma_coords, nthetas + len(omega_coords)
    ) + build_matrix_gradients(sigma_coords)
    return omegas, sigmas


def init(model):
    model = cleanup_model(model)
    model = remove_unused_columns(model)
    dv = next(iter(model.dependent_variables))  # Assuming only one DV
    y = symengine.sympify(
        model.statements.full_expression(dv)
    )  # Function of THETA, ETA , EPS and COVs
    y_norvs = y.subs({rv: 0 for rv in model.random_variables.names})

    symbolic_eta_gradient = [y.diff(eta_name) for eta_name in model.random_variables.etas.names]
    symbolic_eps_gradient = [y.diff(eps_name) for eps_name in model.random_variables.epsilons.names]

    parameter_symbols = get_parameter_symbols(model)

    symbolic_dG_dx_all = [
        [eta.diff(param) for eta in symbolic_eta_gradient] for param in parameter_symbols
    ]
    symbolic_dH_dx_all = [
        [eps.diff(param) for eps in symbolic_eps_gradient] for param in parameter_symbols
    ]

    df = model.dataset
    dvcol = model.datainfo.dv_column.name
    idcol = model.datainfo.id_column.name
    ids = df[idcol].unique()

    omega_inits = build_initial_values_matrix(model.random_variables.etas, model.parameters)
    sigma_inits = build_initial_values_matrix(model.random_variables.epsilons, model.parameters)
    omega_scale = scale_matrix(omega_inits)
    sigma_scale = scale_matrix(sigma_inits)
    omega_coords = build_parameter_coordinates(omega_inits)
    sigma_coords = build_parameter_coordinates(sigma_inits)

    theta_scale = scale_thetas(get_thetas(model))
    omega_grads, sigma_grads = build_parameter_symbolic_gradients(
        len(theta_scale[0]), omega_coords, sigma_coords
    )
    x = build_starting_ucp_vector(theta_scale, omega_coords, sigma_coords)

    state = EstimationState()
    state.parameter_symbols = parameter_symbols
    state.final_ofv = float("inf")

    func = partial(
        ofv_func,
        theta_scale,
        omega_scale,
        sigma_scale,
        omega_coords,
        sigma_coords,
        symbolic_eta_gradient,
        symbolic_eps_gradient,
        y_norvs,
        parameter_symbols,
        ids,
        df,
        idcol,
        dvcol,
        symbolic_dG_dx_all,
        symbolic_dH_dx_all,
        omega_grads,
        sigma_grads,
        state,
    )
    return x, func, state


def ofv_func(
    theta_scale,
    omega_scale,
    sigma_scale,
    omega_coords,
    sigma_coords,
    symbolic_eta_gradient,
    symbolic_eps_gradient,
    y_norvs,
    parameter_symbols,
    ids,
    df,
    idcol,
    dvcol,
    symbolic_dG_dx_all,
    symbolic_dH_dx_all,
    omega_grads,
    sigma_grads,
    state,
    x,
):

    theta_ucp, omega_ucp, sigma_ucp = split_ucps(x, omega_coords, sigma_coords)

    theta = descale_thetas(theta_ucp, theta_scale)
    omega = descale_matrix(omega_ucp, omega_scale)
    sigma = descale_matrix(sigma_ucp, sigma_scale)

    theta_subs = {parameter_symbols[i]: value for i, value in enumerate(theta)}
    subs_eta_gradient = [deta.subs(theta_subs) for deta in symbolic_eta_gradient]
    subs_eps_gradient = [deps.subs(theta_subs) for deps in symbolic_eps_gradient]
    subs_y_norvs = y_norvs.subs(theta_subs)

    OFVsum = 0.0
    gradsum = [0.0] * len(x)

    evaluator = SymengineSubsEvaluator()
    all_PRED = []
    all_RES = []
    all_Ci_inv = []

    for curid in ids:
        curdf = df[df[idcol] == curid]
        DVi = np.array(curdf[dvcol])
        # FIXME: Remove ID, DV from curdf. Better to know beforehand
        # FIXME: Could give empty dataset
        # curdf = curdf[list(set(curdf.columns) - {idcol, dvcol})]
        Gi = evaluator.evaluate_vector(subs_eta_gradient, curdf)
        Hi = evaluator.evaluate_vector(subs_eps_gradient, curdf)
        PREDi = evaluator.evaluate_scalar(subs_y_norvs, curdf)
        RESi = DVi - PREDi
        Ci = Gi @ omega @ Gi.T + (Hi @ sigma @ Hi.T) * np.eye(len(DVi))
        try:
            Ci_inv = np.linalg.inv(Ci)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(x)
        OFVi = np.log(np.linalg.det(Ci)) + RESi.T @ Ci_inv @ RESi
        OFVsum += OFVi

        # gradient calculation
        for i, param in enumerate(parameter_symbols):
            symbolic_dG_dx = symbolic_dG_dx_all[i]
            symbolic_dH_dx = symbolic_dH_dx_all[i]
            symbolic_dP_dx = y_norvs.diff(param)
            symbolic_dG_dx_subs = [e.subs(theta_subs) for e in symbolic_dG_dx]
            symbolic_dH_dx_subs = [e.subs(theta_subs) for e in symbolic_dH_dx]
            symbolic_dP_dx_subs = symbolic_dP_dx.subs(theta_subs)
            dGi = evaluator.evaluate_vector(symbolic_dG_dx_subs, curdf)
            dHi = evaluator.evaluate_vector(symbolic_dH_dx_subs, curdf)
            neg_dPi = -evaluator.evaluate_scalar(symbolic_dP_dx_subs, curdf)
            symb_omega = omega_grads[i]
            symb_sigma = sigma_grads[i]
            dCi = (
                dGi @ omega @ Gi.T
                + Gi @ symb_omega @ Gi.T
                + Gi @ omega @ dGi.T
                + (dHi @ sigma @ Hi.T) * np.eye(len(DVi))
                + (Hi @ symb_sigma @ Hi.T) * np.eye(len(DVi))
                + (Hi @ sigma @ dHi.T) * np.eye(len(DVi))
            )
            grad_i = (
                np.trace(Ci_inv @ dCi)
                + (neg_dPi).T @ Ci_inv @ RESi
                + RESi.T @ (-Ci_inv @ dCi @ Ci_inv @ RESi + Ci_inv @ (neg_dPi))
            )
            gradsum[i] += grad_i

        all_PRED.append(PREDi)
        all_RES.append(RESi)
        all_Ci_inv.append(Ci_inv)

    grad_scale = calculate_gradient_scale(
        theta_ucp,
        omega_ucp,
        sigma_ucp,
        theta_scale,
        omega_scale,
        sigma_scale,
        omega_coords,
        sigma_coords,
    )
    grad = gradsum * grad_scale

    if OFVsum < state.final_ofv:
        state.theta = theta
        state.omega = omega
        state.sigma = sigma
        state.final_ofv = OFVsum
        state.final_PREDs = all_PRED
        state.final_RESs = all_RES
        state.final_Ci_inv = all_Ci_inv

    return OFVsum, grad


def get_parameter_estimates(state, model):
    names = [s.name for s in state.parameter_symbols]
    values = list(state.theta)

    omegas = extract_parameter_estimates_from_matrix(
        state.omega, model.random_variables.etas.covariance_matrix
    )
    values.extend(omegas)

    sigmas = extract_parameter_estimates_from_matrix(
        state.sigma, model.random_variables.epsilons.covariance_matrix
    )
    values.extend(sigmas)

    pe = pd.Series(values, index=names, name="estimates")
    return pe


def get_predictions(state, model):
    requested_predictions = model.execution_steps[-1].predictions
    if "PRED" in requested_predictions:
        PRED_array = np.concatenate(state.final_PREDs)
        predictions = pd.DataFrame({"PRED": PRED_array}, index=model.dataset.index)
    else:
        predictions = None
    return predictions


def get_residuals(state, model):
    requested_residuals = model.execution_steps[-1].residuals
    d = {}
    if "RES" in requested_residuals:
        RES_array = np.concatenate(state.final_RESs)
        d["RES"] = RES_array
    if "WRES" in requested_residuals:
        idcol = model.datainfo.id_column.name
        dvcol = model.datainfo.dv_column.name
        df = model.dataset
        ids = df[idcol].unique()
        # WRESi = Ci^-(1/2) * (DVi - PREDi)
        final_WRESs = [
            linalg.sqrtm(Ci_inv) @ (np.array(df[df[idcol] == curid][dvcol]) - PREDi)
            for Ci_inv, PREDi, curid in zip(state.final_Ci_inv, state.final_PREDs, ids)
        ]
        WRES_array = np.concatenate(final_WRESs)
        d['WRES'] = WRES_array
    if d:
        residuals = pd.DataFrame(d, index=model.dataset.index)
    else:
        residuals = None
    return residuals


def extract_parameter_estimates_from_matrix(numeric, symbolic):
    values = []
    for row in range(symbolic.rows):
        for col in range(row + 1):
            if symbolic[row, col] != 0:
                values.append(numeric[row, col])
    return values


def estimate(model):
    check_input_model(model)
    x, func, state = init(model)

    start_time = time.time()
    from scipy.optimize import minimize

    optres = minimize(func, x, jac=True, method='BFGS')
    end_time = time.time()

    parameter_estimates = get_parameter_estimates(state, model)

    res = ModelfitResults(
        estimation_runtime=end_time - start_time,
        function_evaluations=optres.nfev,
        minimization_successful=optres.success,
        ofv=float(optres.fun),
        parameter_estimates=parameter_estimates,
        predictions=get_predictions(state, model),
        residuals=get_residuals(state, model),
    )
    return res


def execute_model(me, context):
    res = estimate(me.model)
    return ModelEntry(model=me.model, modelfit_results=res)
