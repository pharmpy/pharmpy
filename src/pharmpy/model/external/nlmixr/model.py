from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pharmpy.model
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.model import Assignment, Compartment, get_and_check_dataset, get_and_check_odes
from pharmpy.model.model import ModelInternals
from pharmpy.modeling import drop_columns, get_evid, translate_nmtran_time

from .error_model import res_error_term
from .ini import add_eta, add_sigma, add_theta
from .model_block import add_bio_lag, add_ode, add_statements
from .sanity_checks import check_model


def convert_model(
    model: pharmpy.model.Model,
    skip_check: bool = False,
) -> pharmpy.model.Model:
    """
    Convert a NONMEM model into an nlmixr model

    Parameters
    ----------
    model : pharmpy.model.Model
        A NONMEM pharmpy model object
    skip_check : bool, optional
        Skip determination of error model type. Could speed up conversion. The default is False.

    Returns
    -------
    pharmpy.model.Model
        A model converted to nlmixr format.

    """

    if isinstance(model, Model):
        return model

    nlmixr_model = Model(
        internals=NLMIXRModelInternals(),
        parameters=model.parameters,
        random_variables=model.random_variables,
        statements=model.statements,
        dependent_variables=model.dependent_variables,
        execution_steps=model.execution_steps,
        datainfo=model.datainfo,
        dataset=model.dataset,
        name=model.name,
        description=model.description,
    )

    # Update dataset
    if nlmixr_model.dataset is not None:
        nlmixr_model = translate_nmtran_time(nlmixr_model)
        # FIXME: Dropping columns runs update source which becomes redundant.
        # drop_dropped_columns(nlmixr_model)
        df = get_and_check_dataset(nlmixr_model)
        if all(x in df.columns for x in ("RATE", "DUR")):
            nlmixr_model = drop_columns(nlmixr_model, ["DUR"])
            df = get_and_check_dataset(nlmixr_model)
        nlmixr_model = nlmixr_model.replace(
            datainfo=nlmixr_model.datainfo.replace(path=None),
            dataset=df.reset_index(drop=True),
        )

        # Add evid
        nlmixr_model = add_evid(nlmixr_model)

    # Check model for warnings regarding data structure or model contents
    nlmixr_model = check_model(nlmixr_model, skip_error_model_check=skip_check)

    nlmixr_model.update_source()

    return nlmixr_model


def create_dataset(cg: CodeGenerator, model: pharmpy.model.Model, path=None) -> None:
    """
    Create dataset for nlmixr

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model.Model
        A pharmpy.model object.
    path : TYPE, optional
        Path to add file to. The default is None.

    """
    dataname = f'{model.name}.csv'
    if path is None:
        path = ""
    path = Path(path) / dataname
    cg.add(f'dataset <- read.csv("{path}")')


def create_ini(cg: CodeGenerator, model: pharmpy.model.Model) -> None:
    """
    Create the nlmixr ini block code

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model.Model
        A pharmpy.model object.

    """
    cg.add('ini({')
    cg.indent()

    add_theta(model, cg)

    add_eta(model, cg)

    add_sigma(model, cg)

    cg.dedent()
    cg.add('})')


def create_model(cg: CodeGenerator, model: pharmpy.model.Model) -> None:
    """
    Create the nlmixr model block code

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model.Model
        A pharmpy.model object.

    """

    cg.add('model({')

    # Add statements before ODEs
    cg.indent()
    if len(model.statements.after_odes) != 0:
        add_statements(model, cg, model.statements.before_odes)

    # Add the ODEs
    cg.add("")
    cg.add("# --- DIFF EQUATIONS ---")
    if model.statements.ode_system:
        add_ode(model, cg)
    cg.add("")

    # Find what kind of error model we are looking at
    dv = list(model.dependent_variables.keys())[0]
    dv_statement = model.statements.find_assignment(dv)
    assert isinstance(dv_statement, Assignment)

    only_piecewise = False
    if dv_statement.expression.is_piecewise():
        only_piecewise = True
        dependencies = set()
        res_alias = set()
        for s in model.statements.after_odes:
            if s.symbol == dv:
                if s.expression.is_piecewise:
                    for value, cond in s.expression.args:
                        if value != dv:
                            dv_term = res_error_term(model, value)
                            dependencies.update(dv_term.dependencies())

                            dv_term.create_res_alias()
                            res_alias.update(dv_term.res_alias)
                else:
                    dv_term = res_error_term(model, s.expression)
                    dependencies.update(dv_term.dependencies())

                    dv_term.create_res_alias()
                    res_alias.update(dv_term.res_alias)
    else:
        dv_term = res_error_term(model, dv_statement.expression)
        dependencies = dv_term.dependencies()
        dv_term.create_res_alias()
        res_alias = dv_term.res_alias

    # Add bioavailability statements
    if model.statements.ode_system is not None:
        add_bio_lag(model, cg, bio=True)
        add_bio_lag(model, cg, lag=True)

    # Add statements after ODEs
    if len(model.statements.after_odes) == 0:
        statements = model.statements
    else:
        statements = model.statements.after_odes
        comp_name_dict = {}
        odes = get_and_check_odes(model)
        for name in odes.compartment_names:
            comp = odes.find_compartment(name)
            assert isinstance(comp, Compartment)
            comp_name_dict[comp.amount] = comp.amount.name
        statements = statements.subs(comp_name_dict)
    add_statements(
        model, cg, statements, only_piecewise, dependencies=dependencies, res_alias=res_alias
    )

    cg.dedent()
    cg.add('})')


def create_fit(cg: CodeGenerator, model: pharmpy.model.Model) -> None:
    """
    Create the call to fit for the nlmixr model with appropriate methods and datasets

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model
        A pharmpy.model.Model object.

    """
    # FIXME: Raise error if the method does not match when evaluating
    execution_steps = model.execution_steps[0]
    if "fix_eta" in execution_steps.tool_options:
        fix_eta = True
    else:
        fix_eta = False

    if [s.evaluation for s in model.execution_steps._steps][0] is True:
        max_eval = 0
    else:
        max_eval = execution_steps.maximum_evaluations

    method = execution_steps.method
    interaction = execution_steps.interaction

    nonmem_method_to_nlmixr = {"FOCE": "foce", "FO": "fo", "SAEM": "saem"}

    if method not in nonmem_method_to_nlmixr.keys():
        nlmixr_method = "focei"
    else:
        nlmixr_method = nonmem_method_to_nlmixr[method]

    if interaction and nlmixr_method != "saem":
        nlmixr_method += "i"

    if max_eval is not None:
        if max_eval == 0 and nlmixr_method not in ["fo", "foi", "foce", "focei"]:
            nlmixr_method = "posthoc"
            cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}"')
        else:
            f = f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}", '
            if fix_eta:
                f += f'control=foceiControl(maxOuterIterations={max_eval}, maxInnerIterations=0, etaMat = etas))'
            else:
                f += f'control=foceiControl(maxOuterIterations={max_eval}))'
            cg.add(f)
    else:
        cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}")')


def add_evid(model: pharmpy.model.Model) -> pharmpy.model.Model:
    df = get_and_check_dataset(model)
    if "EVID" not in df.columns:
        df = df.copy()
        df["EVID"] = get_evid(model)
        model = model.replace(dataset=df)
    return model


@dataclass(frozen=True)
class NLMIXRModelInternals(ModelInternals):
    src: Optional[str] = None
    path: Optional[Path] = None


class Model(pharmpy.model.Model):
    filename_extension = '.R'

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )

    def update_source(self):
        cg = CodeGenerator()
        cg.add(f'{self.name} <- function() {{')
        cg.indent()
        create_ini(cg, self)
        create_model(cg, self)
        cg.dedent()
        cg.add('}')
        cg.empty_line()
        create_fit(cg, self)
        # Create lowercase id, time and amount symbols for nlmixr to be able
        # to run
        code = str(cg).replace("AMT", "amt").replace("TIME", "time")
        path = None
        internals = self.internals
        assert internals is not None
        internals = internals.replace(src=code, path=path)
        model = self.replace(internals=internals)
        return model

    @property
    def code(self):
        model = self.update_source()
        internals = model.internals
        assert isinstance(internals, NLMIXRModelInternals)
        code = internals.src
        assert code is not None
        return code
