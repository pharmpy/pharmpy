"""
:meta private:
"""

from operator import add, mul

import sympy
from sympy import Eq, Piecewise

from pharmpy.modeling.help_functions import _format_input_list, _format_options, _get_etas
from pharmpy.parameter import Parameter
from pharmpy.random_variables import RandomVariable
from pharmpy.statements import Assignment, ModelStatements, sympify
from pharmpy.symbols import symbol as S


def add_iiv(model, list_of_parameters, expression, operation='*', eta_names=None):
    """Adds IIVs to :class:`pharmpy.model`.

    Effects that currently have templates are:

    - Additive (*add*)
    - Proportional (*prop*)
    - Exponential (*exp*)
    - Logit (*log*)

    For all except exponential the operation input is not needed. Otherwise user specified
    input is supported. Initial estimates for new etas are 0.09.

    Parameters
    ----------
    model : Model
        Pharmpy model to add new IIVs to.
    list_of_parameters : str, list
        Name/names of parameter to add new IIVs to.
    expression : str, list
        Effect/effects on eta. Either abbreviated (see above) or custom.
    operation : str, list, optional
        Whether the new IIV should be added or multiplied (default).
    eta_names : str, list, optional
        Custom name/names of new eta

    Return
    ------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_iiv(model, "CL") # doctest: +ELLIPSIS
    <...>
    >>> add_iiv(model, "CL", "add")  # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("CL")
    CL := ETA_CL + TVCL

    See also
    --------
    add_pk_iiv
    add_iov
    remove_iiv
    remove_iov

    """
    rvs, pset, sset = (
        model.random_variables.copy(),
        model.parameters.copy(),
        model.statements.copy(),
    )

    list_of_parameters = _format_input_list(list_of_parameters)
    list_of_options = _format_options([expression, operation, eta_names], len(list_of_parameters))
    expression, operation, eta_names = list_of_options

    if all(eta_name is None for eta_name in eta_names):
        eta_names = None

    if not all(len(opt) == len(list_of_parameters) for opt in list_of_options if opt):
        raise ValueError(
            'The number of provided expressions and operations must either be equal '
            'to the number of parameters or equal to 1'
        )

    for i in range(len(list_of_parameters)):
        omega = S(f'IIV_{list_of_parameters[i]}')
        if not eta_names:
            eta_name = f'ETA_{list_of_parameters[i]}'
        else:
            eta_name = eta_names[i]

        eta = RandomVariable.normal(eta_name, 'iiv', 0, omega)

        rvs.append(eta)
        pset.append(Parameter(str(omega), init=0.09))

        statement = sset.find_assignment(list_of_parameters[i])

        if statement is None:
            raise ValueError(f'Could not find parameter: {list_of_parameters[i]}')

        eta_addition = _create_template(expression[i], operation[i])
        eta_addition.apply(statement.expression, eta.name)

        statement.expression = eta_addition.template

    model.random_variables = rvs
    model.parameters = pset
    model.statements = sset

    return model


def add_iov(model, occ, list_of_parameters=None, eta_names=None):
    """Adds IOVs to :class:`pharmpy.model`.

    Initial estimate of new IOVs are 10% of the IIV eta it is based on.

    Parameters
    ----------
    model : Model
        Pharmpy model to add new IOVs to.
    occ : str
        Name of occasion column.
    list_of_parameters : str, list
        List of names of parameters and random variables. Accepts random variable names, parameter
        names, or a mix of both.
    eta_names : str, list
        Custom names of new etas. Must be equal to the number of input etas times the number of
        categories for occasion.

    Return
    ------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> add_iov(model, "TIME", "CL")  # doctest: +SKIP
    <...>
    >>> model.statements.find_assignment("CL")  # doctest: +SKIP
    CL := ETA_CL + TVCL

    See also
    --------
    add_iiv
    add_pk_iiv
    remove_iiv
    remove_iov

    """
    rvs, pset, sset = (
        model.random_variables.copy(),
        model.parameters.copy(),
        model.statements.copy(),
    )

    list_of_parameters = _format_input_list(list_of_parameters)
    etas = _get_etas(model, list_of_parameters, include_symbols=True)
    categories = _get_occ_levels(model.dataset, occ)

    if eta_names and len(eta_names) != len(etas) * len(categories):
        raise ValueError(
            f'Number of provided names incorrect, need {len(etas) * len(categories)} names.'
        )
    elif len(categories) == 1:
        raise ValueError(f'Only one value in {occ} column.')

    iovs, etais = ModelStatements(), ModelStatements()
    for i, eta in enumerate(etas, 1):
        omega_name = str(next(iter(eta.sympy_rv.pspace.distribution.free_symbols)))
        omega = S(f'OMEGA_IOV_{i}')  # TODO: better name
        pset.append(Parameter(str(omega), init=pset[omega_name].init * 0.1))

        iov = S(f'IOV_{i}')

        values, conditions = [], []

        for j, cat in enumerate(categories, 1):
            if eta_names:
                eta_name = eta_names[j - 1]
            else:
                eta_name = f'ETA_IOV_{i}{j}'

            eta_new = RandomVariable.normal(eta_name, 'iov', 0, omega)
            rvs.append(eta_new)

            values += [S(eta_new.name)]
            conditions += [Eq(cat, S(occ))]

        expression = Piecewise(*zip(values, conditions))

        iovs.append(Assignment(iov, sympify(0)))
        iovs.append(Assignment(iov, expression))
        etais.append(Assignment(S(f'ETAI{i}'), eta.symbol + iov))

        sset.subs({eta.name: S(f'ETAI{i}')})

    iovs.extend(etais)
    iovs.extend(sset)

    model.random_variables, model.parameters, model.statements = rvs, pset, iovs

    return model


def add_pk_iiv(model):
    """Adds IIVs to all PK parameters in :class:`pharmpy.model`.

    Will add exponential IIVs to all parameters that are included in the ODE.

    Parameters
    ----------
    model : Model
        Pharmpy model to add new IIVs to.

    Return
    ------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_first_order_absorption(model) # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("MAT")
    MAT := POP_MAT
    >>> add_pk_iiv(model) # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("MAT")
                    ETA_MAT
    MAT := POP_MAT⋅ℯ

    See also
    --------
    add_iiv
    add_iov
    remove_iiv
    remove_iov

    """
    sset, rvs = model.statements, model.random_variables
    odes = sset.ode_system

    params_to_add_etas = []

    for param in odes.free_symbols:
        assign = sset.find_assignment(param)
        if assign:
            if _has_iiv(sset, rvs, assign):
                continue
            dep_assignments = _get_dependent_assignments(sset, assign)
            if dep_assignments:
                for dep_assign in dep_assignments:
                    param_name = dep_assign.symbol.name
                    if not _has_iiv(sset, rvs, dep_assign) and param_name not in params_to_add_etas:
                        params_to_add_etas.append(param_name)
            else:
                if param.name not in params_to_add_etas:
                    params_to_add_etas.append(param.name)

    if params_to_add_etas:
        add_iiv(model, params_to_add_etas, 'exp')

    return model


def _create_template(expression, operation):
    operation_func = _get_operation_func(operation)
    if expression == 'add':
        return EtaAddition.additive()
    elif expression == 'prop':
        return EtaAddition.proportional()
    elif expression == 'exp':
        return EtaAddition.exponential(operation_func)
    elif expression == 'log':
        return EtaAddition.logit()
    else:
        expression = sympify(f'original {operation} {expression}')
        return EtaAddition(expression)


def _get_operation_func(operation):
    """Gets sympy operation based on string"""
    if operation == '*':
        return mul
    elif operation == '+':
        return add


def _get_occ_levels(df, occ):
    levels = df[occ].unique()
    return _round_categories(levels)


def _round_categories(categories):
    categories_rounded = []
    for c in categories:
        if not isinstance(c, int) or c.is_integer():
            categories_rounded.append(int(c))
        else:
            categories_rounded.append(c)
    categories_rounded.sort()
    return categories_rounded


def _get_dependent_assignments(sset, assignment):
    # Finds dependant assignments one layer deep
    dep_assignments = [sset.find_assignment(symb) for symb in assignment.expression.free_symbols]
    return list(filter(None, dep_assignments))


def _has_iiv(sset, rvs, assignment):
    full_expression = sset.before_odes.full_expression(assignment.symbol)
    symb_names = {symb.name for symb in full_expression.free_symbols}
    if symb_names.intersection(rvs.iiv.names):
        return True
    return False


class EtaAddition:
    """
    Eta addition consisting of expression.

    Attributes
    ----------
    template
        Expression consisting of original statement +/* new eta with effect.

    :meta private:

    """

    def __init__(self, template):
        self.template = template

    def apply(self, original, eta):
        self.template = self.template.subs({'original': original, 'eta_new': eta})

    @classmethod
    def additive(cls):
        template = S('original') + S('eta_new')

        return cls(template)

    @classmethod
    def proportional(cls):
        template = S('original') * S('eta_new')

        return cls(template)

    @classmethod
    def exponential(cls, operation):
        template = operation(S('original'), sympy.exp(S('eta_new')))

        return cls(template)

    @classmethod
    def logit(cls):
        template = S('original') * (sympy.exp(S('eta_new')) / (1 + sympy.exp(S('eta_new'))))

        return cls(template)
