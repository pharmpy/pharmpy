import pharmpy.model
from pharmpy.deps import sympy, sympy_printing
from pharmpy.model import Assignment
from pharmpy.modeling import (
    has_additive_error_model,
    has_proportional_error_model,
    has_combined_error_model,
    )

import re

from .name_mangle import name_mangle
from .error_model import (
    find_term,
    add_error_model,
    add_error_relation,
    convert_piecewise,
    )

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

def add_statement(model: pharmpy.model.Model, cg, before_ode):
    if before_ode is True:
        statements = model.statements.before_odes
    else:
        statements = model.statements.after_odes
    
    # FIXME: handle other DVs?
    dv = list(model.dependent_variables.keys())[0]
    
    for s in statements:
        if isinstance(s, Assignment):
            if s.symbol == dv:
                # FIXME : Find another way to assert that a sigma exist
                sigma = None
                for dist in model.random_variables.epsilons:
                    sigma = dist.variance
                assert sigma is not None
                
                if has_additive_error_model(model):
                    expr, error = find_term(model, s.expression)
                    add_error_model(cg, expr, error, s.symbol.name, force_add = True)
                    add_error_relation(cg, error, s.symbol)
                elif has_proportional_error_model(model):
                    if len(sympy.Add.make_args(s.expression)) == 1:
                        expr, error = find_term(model, sympy.expand(s.expression))
                    else:
                        expr, error = find_term(model, s.expression)
                    add_error_model(cg, expr, error, s.symbol.name, force_prop = True)
                    add_error_relation(cg, error, s.symbol)
                elif has_combined_error_model(model):
                    # FIXME : Combine with the unknown case
                    expr, error = find_term(model, s.expression)
                    add_error_model(cg, expr, error, s.symbol.name, force_comb = True)
                    add_error_relation(cg, error, s.symbol)
                else:
                    if s.expression.is_Piecewise:
                        # Convert eps to sigma name
                        #piecewise = convert_eps_to_sigma(s, model)
                        convert_piecewise(s, cg, model)
                    else:         
                        expr, error = find_term(model, s.expression)
                        add_error_model(cg, expr, error, s.symbol.name)
                        add_error_relation(cg, error, s.symbol)
                    
            else:
                expr = s.expression
                if expr.is_Piecewise:
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
                                            if not isinstance(cond, sympy.LessThan) and isinstance(largest_cond, sympy.LessThan):
                                                largest_value = value
                                                largest_cond = cond
                                value = largest_value
                        cg.indent()
                        cg.add(f'{s.symbol.name} <- {value}')
                        cg.dedent()
                    cg.add('}')
                else:
                    cg.add(f'{s.symbol.name} <- {expr}')
                
def add_ode(model, cg):
    
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

def remove_piecewise(expr:str) -> str:
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
    #Go into each piecewise found
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
            all_piecewise.append(expr[p+1:d[p]])
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
    start = [] # all opening parentheses
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
    TYPE
        DESCRIPTION.

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
    cond = cond.replace("∧", "&")
    cond = cond.replace("∨", "|")
    cond = re.sub(r'(ID\s*==\s*)(\d+)', r"\1'\2'", cond)
    return cond
            
            
            