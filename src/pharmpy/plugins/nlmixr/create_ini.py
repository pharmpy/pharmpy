from pharmpy.deps import sympy
from pharmpy.modeling import get_sigmas

from .CodeGenerator import CodeGenerator
from .name_mangle import name_mangle


def add_theta(model, cg):
    thetas = [p for p in model.parameters if p.symbol not in model.random_variables.free_symbols]
    for theta in thetas:
        if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
            add_ini_parameter(cg, theta, boundary=True)
        else:
            add_ini_parameter(cg, theta)


def add_eta(model, cg):
    for dist in model.random_variables.etas:
        omega = dist.variance
        if len(dist.names) == 1:
            init = model.parameters[omega.name].init
            cg.add(f'{name_mangle(dist.names[0])} ~ {init}')
        else:
            inits = []
            for row in range(omega.rows):
                for col in range(row + 1):
                    inits.append(model.parameters[omega[row, col].name].init)
            cg.add(
                f'{" + ".join([name_mangle(name) for name in dist.names])} ~ c({", ".join([str(x) for x in inits])})'
            )


def add_sigma(model, cg):
    for sigma in get_sigmas(model):
        if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
            add_ini_parameter(cg, sigma, boundary=True)
        else:
            add_ini_parameter(cg, sigma)


def add_ini_parameter(cg: CodeGenerator, parameter: sympy.Symbol, boundary: bool = False) -> None:
    """
    Add a parameter to the ini block in nlmixr2. This is performed for theta
    and sigma parameter values as they are handled in the same manner.

    Parameters
    ----------
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    parameter : sympy.Symbol
        The parameter to be added. Either theta or sigma
    boundary : bool, optional
        Decide if the parameter should be added with or without parameter
        boundries. The default is False.

    Returns
    -------
    None
        Modifies the given CodeGenerator object. Returns nothing

    """
    parameter_name = name_mangle(parameter.name)
    if parameter.fix:
        cg.add(f'{parameter_name} <- fixed({parameter.init})')
    else:
        limit = 1000000.0
        if boundary:
            if parameter.lower > -limit and parameter.upper < limit:
                cg.add(
                    f'{parameter_name} <- c({parameter.lower}, {parameter.init}, {parameter.upper})'
                )
            elif parameter.lower <= -limit and parameter.upper < limit:
                cg.add(f'{parameter_name} <- c(-Inf, {parameter.init}, {parameter.upper})')
            elif parameter.lower > -limit and parameter.upper >= limit:
                cg.add(f'{parameter_name} <- c({parameter.lower}, {parameter.init}, Inf)')
            else:
                cg.add(f'{parameter_name} <- {parameter.init}')
        else:
            cg.add(f'{parameter_name} <- {parameter.init}')
