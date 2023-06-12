from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import pharmpy.model
from pharmpy.deps import sympy
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.model.external.nlmixr.error_model import convert_eps_to_sigma
from pharmpy.model.external.nlmixr.model import add_evid
from pharmpy.model.external.nlmixr.model_block import add_bio_lag, add_ode, convert_eq
from pharmpy.modeling import (
    drop_columns,
    get_bioavailability,
    get_lag_times,
    get_thetas,
    translate_nmtran_time,
)


@dataclass
class RxODEModelInternals:
    src: Optional[str] = None
    path: Optional[Path] = None


class Model(pharmpy.model.Model):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )

    def update_source(self):
        cg = CodeGenerator()
        cg.add(f'{self.name} <- rxode2({{')
        cg.indent()
        create_model(cg, self)
        cg.dedent()
        cg.add('})')
        cg.empty_line()
        create_theta(cg, self)
        cg.empty_line()
        create_eta(cg, self)
        cg.empty_line()
        create_sigma(cg, self)
        cg.empty_line()
        create_fit(cg, self)
        self.internals.src = str(cg).replace("AMT", "amt").replace("TIME", "time")
        self.internals.path = None
        code = str(cg)
        internals = replace(self.internals, src=code)
        model = self.replace(internals=internals)
        return model

    @property
    def model_code(self):
        model = self.update_source()
        code = model.internals.src
        assert code is not None
        return code


def convert_model(model, skip_check=False):
    if isinstance(model, Model):
        return model

    rxode_model = Model(
        internals=RxODEModelInternals(),
        parameters=model.parameters,
        random_variables=model.random_variables,
        statements=model.statements,
        dependent_variables=model.dependent_variables,
        estimation_steps=model.estimation_steps,
        filename_extension='.R',
        datainfo=model.datainfo,
        dataset=model.dataset,
        name=model.name,
        description=model.description,
    )

    # Update dataset
    if model.dataset is not None:
        rxode_model = translate_nmtran_time(rxode_model)

        if all(x in rxode_model.dataset.columns for x in ["RATE", "DUR"]):
            rxode_model = drop_columns(rxode_model, ["DUR"])
        rxode_model = rxode_model.replace(
            datainfo=rxode_model.datainfo.replace(path=None),
            dataset=rxode_model.dataset.reset_index(drop=True),
        )

        # Add evid
        rxode_model = add_evid(rxode_model)

    # Check model for warnings regarding data structure or model contents
    from pharmpy.model.external.nlmixr.sanity_checks import check_model

    rxode_model = check_model(rxode_model, skip_error_model_check=skip_check)

    rxode_model.update_source()

    return rxode_model


def create_model(cg, model):
    add_true_statements(model, cg, model.statements.before_odes)

    if model.statements.ode_system:
        add_ode(model, cg)

    # Add bioavailability statements
    if model.statements.ode_system is not None:
        add_bio_lag(model, cg, bio=True)
        add_bio_lag(model, cg, lag=True)

    # Add statements after ODE
    add_true_statements(model, cg, model.statements.after_odes)


def add_true_statements(model, cg, statements):
    for s in statements:
        if model.statements.ode_system is not None and (
            s.symbol in get_bioavailability(model).values()
            or s.symbol in get_lag_times(model).values()
        ):
            pass
        else:
            expr = s.expression
            expr = convert_eps_to_sigma(expr, model)
            if expr.is_Piecewise:
                add_piecewise(model, cg, s)
            else:
                cg.add(f'{s.symbol.name} <- {expr}')


def add_piecewise(model: pharmpy.model.Model, cg: CodeGenerator, s):
    expr = s.expression
    first = True
    for value, cond in expr.args:
        if cond is not sympy.S.true:
            if cond.atoms(sympy.Eq):
                cond = convert_eq(cond)
            if first:
                cg.add(f'if ({cond}) {{')
                first = False
            else:
                cg.add(f'}} else if ({cond}) {{')
        else:
            cg.add('} else {')
            if "NEWIND" in [t.name for t in expr.free_symbols] and value == 0:
                largest_value = expr.args[0].expr
                largest_cond = expr.args[0].cond
                for value, cond in expr.args[1:]:
                    if cond is not sympy.S.true:
                        if cond.rhs > largest_cond.rhs:
                            largest_value = value
                            largest_cond = cond
                        elif cond.rhs == largest_cond.rhs:
                            if not isinstance(cond, sympy.LessThan) and isinstance(
                                largest_cond, sympy.LessThan
                            ):
                                largest_value = value
                                largest_cond = cond
                value = largest_value
        cg.indent()
        value = convert_eps_to_sigma(value, model)
        cg.add(f'{s.symbol.name} <- {value}')
        cg.dedent()
    cg.add('}')


def create_theta(cg, model):
    cg.add("thetas <-")
    cg.add("c(")
    thetas = get_thetas(model)
    for n, theta in enumerate(thetas):
        if n != len(thetas) - 1:
            cg.add(f'{theta.name} = {theta.init}, ')
        else:
            cg.add(f'{theta.name} = {theta.init}')
    cg.add(")")


def create_eta(cg, model):
    from pharmpy.model.external.nlmixr.ini import add_eta

    cg.add("omegas = lotri(")
    add_eta(model, cg, as_list=True)
    cg.add(")")


def create_sigma(cg, model):
    cg.add("sigmas <-")
    cg.add("lotri(")
    all_eps = model.random_variables.epsilons
    for n, eps in enumerate(all_eps):
        sigma = eps.variance
        if len(eps.names) == 1:
            name = model.parameters[sigma].name
            init = model.parameters[sigma].init
            if n != len(all_eps) - 1:
                cg.add(f'{name} ~ {init},')
            else:
                cg.add(f'{name} ~ {init}')
        else:
            cg.add(f'{" + ".join([name for name in eps.names])} ~ c(')
            inits = []
            for row in range(sigma.rows):
                for col in range(row + 1):
                    if col == 0 and row != 0:
                        cg.add(f'{", ".join([str(x) for x in inits])},')
                        inits = []
                        inits.append(f'{model.parameters[sigma[row, col].name].init}')
                    else:
                        inits.append(model.parameters[sigma[row, col].name].init)
            cg.add(f'{", ".join([str(x) for x in inits])}')
            if eps != model.random_variables.epsilons[-1]:
                cg.add("),")
            else:
                cg.add(")")
    cg.add(")")


def create_fit(cg, model):
    cg.add(f'fit <- {model.name} %>% rxSolve(thetas, ev, omega = omegas, sigma = sigmas)')
