from dataclasses import dataclass, replace
from typing import Optional
from pathlib import Path
import pharmpy.model
from pharmpy.internals.code_generator import CodeGenerator

from pharmpy.deps import sympy
from pharmpy.model import Assignment

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
        cg.add(f'{self.name} <- rxode2{{')
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
    
    rxode_model.update_source()
    
    return rxode_model

def create_model(cg, model):
    from pharmpy.plugins.nlmixr.model_block import (
        add_statements,
        add_ode
        )
    add_statements(model, cg, model.statements.before_odes)

    if model.statements.ode_system:
        add_ode(model, cg)

    add_statements(model, cg, model.statements.after_odes)

def create_theta(cg, model):
    from pharmpy.modeling import get_thetas
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
    from pharmpy.modeling import get_omegas
    cg.add("etas <-")
    cg.add("c(")
    omegas = get_omegas(model)
    for n, omega in enumerate(omegas):
        if n != len(omegas)-1:
            cg.add(f'{omega.name} = {omega.init}, ')
        else:
            cg.add(f'{omega.name} = {omega.init}')
    cg.add(")")

def create_sigma(cg, model):
    from pharmpy.modeling import get_sigmas
    cg.add("sigmas <-")
    cg.add("c(")
    sigmas = get_sigmas(model)
    for n, sigma in enumerate(sigmas):
        if n != len(sigmas)-1:
            cg.add(f'{sigma.name} = {sigma.init}, ')
        else:
            cg.add(f'{sigma.name} = {sigma.init}')
    cg.add(")")

def create_fit(cg, model):
    cg.add(f'fit <- {model.name} %>% rxSolve(theta = thetas, omega = omegas, sigma = sigmas, events = ev)')