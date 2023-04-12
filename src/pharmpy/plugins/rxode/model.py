import uuid
import os
import json
import subprocess
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
    convert_eq
    )
from pharmpy.modeling import (
    get_sigmas,
    get_thetas,
    write_csv
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

def execute_model(model: pharmpy.model.Model, db: str) -> pharmpy.model.Model:
    """
    Executes a model using rxode2.

    Parameters
    ----------
    model : pharmpy.model.Model
        An pharmpy model object.
    db : str
        Name of folder in home directory to store resulting files in.

    Returns
    -------
    model : pharmpy.model.Model
        Model with accompanied results.

    """
    db = pharmpy.workflows.LocalDirectoryToolDatabase(db)
    database = db.model_database
    model = convert_model(model)
    path = Path.cwd() / f'rxode_run_{model.name}-{uuid.uuid1()}'
    model.internals.path = path
    meta = path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    write_csv(model, path=path)
    model = model.replace(datainfo=model.datainfo.replace(path=path))

    dataname = f'{model.name}.csv'
    pre = f'library(rxode2)\n\nev <- read.csv("{path / dataname}")\n'

    pre += "\n"

    code = pre + model.model_code
    cg = CodeGenerator()
    cg.add("res <- as.data.frame(fit)")
    cg.add(
        f'save(file="{path}/{model.name}.RDATA", res)'
    )
    
    with open(path / f'{model.name}.R', 'w') as fh:
        fh.write(code)

    from pharmpy.plugins.nlmixr import conf

    rpath = conf.rpath / 'bin' / 'Rscript'

    newenv = os.environ
    # Reset environment variables incase started from R
    # and calling other R version.
    newenv['R_LIBS_USERS'] = ''
    newenv['R_LIBS_SITE'] = ''

    stdout = path / 'stdout'
    stderr = path / 'stderr'

    args = [str(rpath), str(path / (model.name + '.R'))]

    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, stderr=err, stdout=out, env=newenv)

    rdata_path = path / f'{model.name}.RDATA'

    metadata = {
        'plugin': 'nlmixr',
        'path': str(path),
    }

    plugin = {
        'rpath': str(rpath),
        'commands': [
            {
                'args': args,
                'returncode': result.returncode,
                'stdout': 'stdout',
                'stderr': 'stderr',
            }
        ],
    }

    with database.transaction(model) as txn:
        txn.store_local_file(path / f'{model.name}.R')
        txn.store_local_file(rdata_path)

        txn.store_local_file(stdout)
        txn.store_local_file(stderr)
        txn.store_local_file(path / f'{model.name}.csv')

        txn.store_local_file(model.datainfo.path)

        plugin_path = path / 'nlmixr.json'
        with open(plugin_path, 'w') as f:
            json.dump(plugin, f, indent=2)

        txn.store_local_file(plugin_path)

        txn.store_metadata(metadata)
        txn.store_modelfit_results()

    #res = parse_modelfit_results(model, path)
    #model = model.replace(modelfit_results=res)
    return model

def convert_model(model):
    
    if isinstance(model, Model):
        return model
    
    if model.internals.control_stream.get_records("DES"):
        des = model.internals.control_stream.get_records("DES")[0]
    else:
        des = None
    
    rxode_model = Model(
        internals=RxODEModelInternals(DES=des),
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
    add_true_statements(model, cg, model.statements.before_odes)

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
        cg.add(f'{s.symbol.name} <- {value}')
        cg.dedent()
    cg.add('}')

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