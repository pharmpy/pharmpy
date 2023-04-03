from dataclasses import dataclass, replace
from typing import Optional
from pathlib import Path
import pharmpy.model
from pharmpy.internals.code_generator import CodeGenerator

from pharmpy.deps import sympy
from pharmpy.model import Assignment
from pharmpy.modeling import drop_columns
from pharmpy.plugins.nlmixr.model import (
    add_evid,
    )
from pharmpy.plugins.nlmixr.model_block import (
    add_statements,
    add_ode,
    add_piecewise
    )
from pharmpy.modeling import (
    get_sigmas,
    get_thetas
    )
from pharmpy.plugins.nlmixr.error_model import convert_eps_to_sigma

@dataclass
class RxODEModelInternals:
    src: Optional[str] = None
    path: Optional[Path] = None
    DES: Optional = None


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


def convert_model(model):
    
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
    
    if all(x in rxode_model.dataset.columns for x in ["RATE", "DUR"]):
        rxode_model = drop_columns(rxode_model, ["DUR"])
    rxode_model = rxode_model.replace(
        datainfo=rxode_model.datainfo.replace(path=None),
        dataset=rxode_model.dataset.reset_index(drop=True),
    )

    # Add evid
    rxode_model = add_evid(rxode_model)
    
    rxode_model.update_source()
    
    return rxode_model

def create_model(cg, model):
    add_statements(model, cg, model.statements.before_odes)

    if model.statements.ode_system:
        add_ode(model, cg)

    add_true_statements(model,
                   cg,
                   model.statements.after_odes)

def add_true_statements(model, cg, statements):
    for s in statements:
        expr = s.expression
        expr = convert_eps_to_sigma(expr, model)
        if expr.is_Piecewise:
            add_piecewise(model, cg, s)
        else:
            cg.add(f'{s.symbol.name} <- {expr}')

def create_theta(cg, model):
    cg.add("thetas <-")
    cg.add("c(")
    thetas = get_thetas(model)
    for n, theta in enumerate(thetas):
        if n != len(thetas)-1:
            cg.add(f'{theta.name} = {theta.init}, ')
        else:
            cg.add(f'{theta.name} = {theta.init}')
    cg.add(")")

def create_eta(cg, model):
    from pharmpy.plugins.nlmixr.ini import add_eta
    cg.add("omegas = lotri(")
    add_eta(model, cg, as_list = True)
    cg.add(")")

def create_sigma(cg, model):
    cg.add("sigmas <-")
    cg.add("c(")
    sigmas = get_sigmas(model)
    for n, sigma in enumerate(sigmas):
        if n != len(sigmas)-1:
            cg.add(f'{sigma.name} ~ {sigma.init}, ')
        else:
            cg.add(f'{sigma.name} ~ {sigma.init}')
    cg.add(")")

def create_fit(cg, model):
    cg.add(f'fit <- {model.name} %>% rxSolve(thetas, ev, omega = omegas, sigma = sigmas)')