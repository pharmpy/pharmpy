import pharmpy.model
from pharmpy.deps import sympy
from pharmpy.internals.code_generator import CodeGenerator

from .name_mangle import name_mangle


def add_theta(model: pharmpy.model.Model, cg: CodeGenerator) -> None:
    """
    Add THETAs to code generator when converting a model

    Parameters
    ----------
    model : pharmpy.model.Model
        Pharmpy model object.
    cg : CodeGenerator
        Code generator to add code upon.
    """
    cg.add("# --- THETAS ---")
    thetas = [p for p in model.parameters if p.symbol not in model.random_variables.free_symbols]
    for theta in thetas:
        if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
            add_ini_parameter(cg, theta, boundary=True)
        else:
            add_ini_parameter(cg, theta)


def add_eta(model: pharmpy.model.Model, cg: CodeGenerator, as_list=False) -> None:
    """
    Add ETAs to code generator when converting a model

    Parameters
    ----------
    model : pharmpy.model.Model
        Pharmpy model object.
    cg : CodeGenerator
        Code generator to add code upon.
    as_list : bool
        Add with separation character ","
    """
    cg.add("")
    cg.add("# --- ETAS ---")
    for dist in model.random_variables.etas:
        omega = dist.variance
        if len(dist.names) == 1:
            init = model.parameters[omega.name].init
            code_line = f'{name_mangle(dist.names[0])} ~ {init}'
            if as_list and dist != model.random_variables.etas[-1]:
                code_line += ","
            cg.add(code_line)
        else:
            cg.add(f'{" + ".join([name_mangle(name) for name in dist.names])} ~ c(')
            inits = []
            for row in range(omega.rows):
                for col in range(row + 1):
                    if col == 0 and row != 0:
                        cg.add(f'{", ".join([str(x) for x in inits])},')
                        inits = []
                        inits.append(f'{model.parameters[omega[row, col].name].init}')
                    else:
                        inits.append(model.parameters[omega[row, col].name].init)
            cg.add(f'{", ".join([str(x) for x in inits])}')
            if as_list and dist != model.random_variables.etas[-1]:
                cg.add("),")
            else:
                cg.add(")")


def add_sigma(model: pharmpy.model.Model, cg: CodeGenerator) -> None:
    """
    Add SIGMAs to code generator when converting a model.

    Parameters
    ----------
    model : pharmpy.model.Model
        Pharmpy model object.
    cg : CodeGenerator
        Code generator to add code upon.
    """
    cg.add("")
    cg.add("# --- EPSILONS ---")
    for dist in model.random_variables.epsilons:
        sigma = dist.variance
        if len(dist.names) == 1:
            sigma_param = model.parameters[sigma]
            if sigma_param.init != 1:
                if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
                    add_ini_parameter(cg, sigma_param, boundary=True)
                else:
                    add_ini_parameter(cg, sigma_param)
            elif not sigma_param.fix:
                if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
                    add_ini_parameter(cg, sigma_param, boundary=True)
                else:
                    add_ini_parameter(cg, sigma_param)
        else:
            for row, col in zip(range(sigma.rows), range(sigma.rows + 1)):
                sigma_param = model.parameters[sigma[row, col]]
                if sigma_param.init != 1:
                    if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
                        add_ini_parameter(cg, sigma_param, boundary=True)
                    else:
                        add_ini_parameter(cg, sigma_param)
                elif not sigma_param.fix:
                    if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
                        add_ini_parameter(cg, sigma_param, boundary=True)
                    else:
                        add_ini_parameter(cg, sigma_param)


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
