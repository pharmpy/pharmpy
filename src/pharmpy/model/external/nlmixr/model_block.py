import re
from typing import Set, Union

import pharmpy.model
from pharmpy.deps import sympy, sympy_printing
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.model import Assignment
from pharmpy.modeling import get_bioavailability, get_lag_times

from .error_model import res_error_term
from .name_mangle import name_mangle


class ExpressionPrinter(sympy_printing.str.StrPrinter):
    def __init__(self, amounts):
        self.amounts = amounts
        super().__init__()

    def _print_Symbol(self, expr):
        return name_mangle(expr.name)

    def _print_Derivative(self, expr):
        fn = expr.args[0]
        return f'd/dt({fn.name})'

    def _print_Function(self, expr):
        name = expr.func.__name__
        if name in self.amounts:
            return expr.func.__name__
        else:
            return expr.func.__name__ + f'({self.stringify(expr.args, ", ")})'


def add_statements(
    model: pharmpy.model.Model,
    cg: CodeGenerator,
    statements: pharmpy.model.statements,
    only_piecewise: Union[bool, None] = None,
    dependencies: Set[sympy.Symbol] = set(),
    res_alias: Set[sympy.Symbol] = set(),
):
    """
    Will add the provided statements to the code generator objects, translated
    to nlmixr format.

    Parameters
    ----------
    model : pharmpy.model.Model
        A pharmpy model object to add statements to.
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    statements : pharmpy.model.statements
        Statements to be added to the code generator.
    only_piecewise : Union[bool, None], optional
        Is the dependent variable only dependent on piecewise statements.
        The default is None.
    dependencies : set[sympy.Symbols], optional
        A set with symbols that the dependent variable are dependent on.
        Could for instance a term 'W' which define the error model.
        The default is set().
    res_alias : set[sympy.Symbols], optional
        A set with aliases for the dependent or resulting term for the
        dependent variable. The default is set().

    """

    # FIXME: handle other DVs?
    dv = list(model.dependent_variables.keys())[0]

    for s in statements:
        if isinstance(s, Assignment):
            if s.symbol == dv and not s.expression.is_Piecewise:
                # FIXME : Find another way to assert that a sigma exist
                sigma = None
                for dist in model.random_variables.epsilons:
                    sigma = dist.variance
                assert sigma is not None

                if only_piecewise is False:
                    dv_term = res_error_term(model, s.expression)
                    res = dv_term.res
                    if len(dependencies) != 0:
                        cg.add(f'{s.symbol} <- {res}')
                        cg.add(f'{s.symbol} ~ add(add_error) + prop(prop_error)')

                        # TODO: Remove sigma here instead of in ini
                        # also remove aliases

                    else:
                        cg.add(f"{s.symbol} <- {res}")
                        cg.add(f'add_error <- {dv_term.add.expr}')
                        cg.add(f'prop_error <- {dv_term.prop.expr}')
                        cg.add(f'{s.symbol} ~ add(add_error) + prop(prop_error)')
                else:
                    dv_term = res_error_term(model, s.expression)
                    cg.add(f"res <- {dv_term.res}")
                    cg.add(f'add_error <- {dv_term.add.expr}')
                    cg.add(f'prop_error <- {dv_term.prop.expr}')

            elif model.statements.ode_system is not None and (
                s.symbol in get_bioavailability(model).values()
                or s.symbol in get_lag_times(model).values()
            ):
                pass
            else:
                expr = s.expression
                if expr.is_Piecewise:
                    first = True
                    for value, cond in expr.args:
                        if cond is not sympy.S.true:
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

                        if only_piecewise is False:
                            cg.add(f'{s.symbol.name} <- {value}')

                            if s.symbol in dependencies:
                                if not value.is_constant() and not isinstance(value, sympy.Symbol):
                                    add, prop = extract_add_prop(value, res_alias, model)
                                    cg.add(f'add_error <- {add}')
                                    cg.add(f'prop_error <- {prop}')
                        elif s.symbol == dv:
                            if value != dv:
                                t = res_error_term(model, value)
                                # FIXME : Remove sigma here instead of in ini
                                # also remove aliases for sigma
                                cg.add(f"res <- {t.res}")
                                cg.add(f'add_error <- {t.add.expr}')
                                cg.add(f'prop_error <- {t.prop.expr}')
                            else:
                                cg.add("res <- res")
                                cg.add('add_error <- add_error')
                                cg.add('prop_error <- prop_error')
                        else:
                            cg.add(f'{s.symbol.name} <- {value}')

                        cg.dedent()
                    cg.add('}')
                elif s.symbol in dependencies:
                    add, prop = extract_add_prop(s.expression, res_alias, model)
                    cg.add(f'{s.symbol.name} <- {expr}')  # TODO : Remove ?
                    cg.add(f'add_error <- {add}')
                    cg.add(f'prop_error <- {prop}')
                else:
                    cg.add(f'{s.symbol.name} <- {expr}')

    if only_piecewise:
        cg.add(f'{dv} <- res')
        cg.add(f'{dv} ~ add(add_error) + prop(prop_error)')


def extract_add_prop(s, res_alias: Set[sympy.symbols], model: pharmpy.model.Model):
    """
        Extract additiv and proportional error terms from a sympy expression

        Parameters
        ----------
        s : A sympy expression
            A sympy expression from which an additive and a proportional error
            term are to be extracted.
        res_alias : set[sympy.symbols]
            A set with aliases for the dependent or resulting term for the
            dependent variable.
        model : pharmpy.model.Model
            The connected pharmpy model.

    Returns
        -------
        add : sympy.Symbol
            The symbol representing the additive error. Zero if none found
        prop : sympy.Symbol
            The symbol representing  the proportional error. Zero if none found

    """
    if isinstance(s, sympy.Symbol):
        terms = [s]
    elif isinstance(s, sympy.Pow):
        terms = sympy.Add.make_args(s.args[0])
    elif isinstance(s, sympy.Mul):
        terms = [s]
    elif isinstance(s, sympy.Integer):
        terms = [s]
    elif isinstance(s, sympy.Float):
        terms = [s]
    else:
        terms = sympy.Add.make_args(s.expression)
    assert len(terms) <= 2

    w = False

    if isinstance(s, sympy.Pow):
        s_arg = sympy.Add.make_args(s.args[0])
        if len(s_arg) <= 2:
            all_pow = True
            for t in s_arg:
                for f in sympy.Mul.make_args(t):
                    if (
                        isinstance(f, sympy.Pow)
                        or isinstance(f, sympy.Integer)
                        or isinstance(f, sympy.Float)
                    ):
                        pass
                    else:
                        all_pow = False
        if all_pow:
            w = True

    prop = 0
    add = 0
    prop_found = False
    for term in terms:
        for symbol in term.free_symbols:
            if symbol in res_alias:
                if prop_found is False:
                    term = term.subs(symbol, 1)
                    if w:
                        prop = sympy.sqrt(term)
                    else:
                        prop += term
                    prop_found = True
        if prop_found is False:
            if w:
                add = sympy.sqrt(term)
            else:
                add += term
    return add, prop


def add_bio_lag(model: pharmpy.model.Model, cg: CodeGenerator, bio=False, lag=False):
    if bio:
        bio_lag = get_bioavailability(model)
    elif lag:
        bio_lag = get_lag_times(model)
    else:
        return

    for s in model.statements.before_odes:
        if s.symbol in bio_lag.values():
            comp = list(bio_lag.keys())[list(bio_lag.values()).index(s.symbol)]

            if s.expression.is_Piecewise:
                first = True
                for value, cond in s.expression.args:
                    if cond is not sympy.S.true:
                        cond = convert_eq(cond)
                        if first:
                            cg.add(f'if ({cond}) {{')
                            first = False
                        else:
                            cg.add(f'}} else if ({cond}) {{')
                    else:
                        cg.add('} else {')

                    if bio:
                        cg.add(f'f(A_{comp}) <- {value}')
                    elif lag:
                        cg.add(f'alag(A_{comp}) <- {value}')
            else:
                if bio:
                    cg.add(f'f(A_{comp}) <- {s.expression}')
                elif lag:
                    cg.add(f'alag(A_{comp}) <- {s.expression}')


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


def add_ode(model: pharmpy.model.Model, cg: CodeGenerator) -> None:
    """
    Add the ODEs from a model to a code generator

    Parameters
    ----------
    model : pharmpy.model.Model
        A pharmpy model object to add ODEs to.
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    """
    amounts = [am.name for am in list(model.statements.ode_system.amounts)]
    printer = ExpressionPrinter(amounts)

    for eq in model.statements.ode_system.eqs:
        # Should remove piecewise from these equations in nlmixr
        if eq.atoms(sympy.Piecewise):
            lhs = remove_piecewise(printer.doprint(eq.lhs))
            rhs = remove_piecewise(printer.doprint(eq.rhs))

            cg.add(f'{lhs} = {rhs}')
        else:
            cg.add(f'{printer.doprint(eq.lhs)} = {printer.doprint(eq.rhs)}')


def remove_piecewise(expr: str) -> str:
    """
    Return an expression without Piecewise statements

    Parameters
    ----------
    expr : str
        A sympy expression, given as a string

    Returns
    -------
    str
        Return given expression but without Piecewise statements

    """
    all_piecewise = find_piecewise(expr)
    # Go into each piecewise found
    for p in all_piecewise:
        expr = piecewise_replace(expr, p, "")
    return expr


def find_piecewise(expr: str) -> list:
    """
    Locate all Piecewise statements in an expression and return them as a list

    Parameters
    ----------
    expr : str
        A sympy expression in string form

    Returns
    -------
    list
        A list of all piecewise statements found in the expression

    """
    d = find_parentheses(expr)

    piecewise_start = [m.start() + len("Piecewise") for m in re.finditer("Piecewise", expr)]

    all_piecewise = []
    for p in piecewise_start:
        if p in d:
            all_piecewise.append(expr[p + 1 : d[p]])
    return all_piecewise


def find_parentheses(s: str) -> dict:
    """
    Find all matching parenthesis in a given string

    Parameters
    ----------
    s : str
        Any string

    Returns
    -------
    dict
        A dictionary with keys corresponding to positions of the opening
        parenthesis and value being the position of the closing parenthesis.

    """
    start = []  # all opening parentheses
    d = {}

    for i, c in enumerate(s):
        if c == '(':
            start.append(i)
        if c == ')':
            try:
                d[start.pop()] = i
            except IndexError:
                print('Too many closing parentheses')
    if start:  # check if stack is empty afterwards
        print('Too many opening parentheses')

    return d


def piecewise_replace(expr: str, piecewise: sympy.Piecewise, s: str) -> str:
    """
    Will replace a given piecewise expression with wanted string, s

    Parameters
    ----------
    expr : str
        A string representive a sympy expression
    piecewise : sympy.Piecewise
        A sympy Piecewise statement to be changed
    s : str
        A string with wanted information to change piecewise to

    Returns
    -------
    str
        Expression with replaced piecewise expressions.

    """
    if s == "":
        expr = re.sub(r'([\+\-\/\*]\s*)(Piecewise)', r'\2', expr)
        return expr.replace(f'Piecewise({piecewise})', s)
    else:
        return expr.replace(f'Piecewise({piecewise})', s)


def convert_eq(cond: sympy.Eq) -> str:
    """
    Convert a sympy equal statement to R syntax

    Parameters
    ----------
    cond : sympy.Eq
        Sympy equals statement

    Returns
    -------
    str
        A string with R format for the same statement

    """
    cond = sympy.pretty(cond)
    cond = cond.replace("=", "==")
    cond = cond.replace("≠", "!=")
    cond = cond.replace("≤", "<=")
    cond = cond.replace("≥", "<=")
    cond = cond.replace("∧", "&")
    cond = cond.replace("∨", "|")

    cond = re.sub(r'(ID\s*==\s*)(\d+)', r"\1'\2'", cond)

    if (
        re.search(r'(ID\s*<=\s*)(\d+)', cond)
        or re.search(r'(ID\s*>=\s*)(\d+)', cond)
        or re.search(r'(ID\s*<\s*)(\d+)', cond)
        or re.search(r'(ID\s*>\s*)(\d+)', cond)
    ):
        print(f"Condition '{cond}' not supported by nlmixr. Model will not run.")
    return cond
