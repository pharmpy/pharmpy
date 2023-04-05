from typing import Dict, Tuple, Union

import pharmpy.model
from pharmpy.deps import sympy
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.internals.expr.subs import subs
from pharmpy.modeling import get_sigmas


def find_term(
    model: pharmpy.model.Model,
    expr: sympy.Add,
    cg: CodeGenerator
) -> Tuple[Union[sympy.Symbol, sympy.Add], Dict]:
    """
    For a given expression for the dependent variable, find the terms
    connected to the actual result and the terms connected to the error model.

    Parameters
    ----------
    model : pharmpy.model
        A pharmpy model object
    expr : sympy.Add
        An expression for the dependent variable. Should be a sympy.Add statement
    cg : CodeGenerator
        CodeGenerator holding the model code

    Raises
    ------
    ValueError
        If the model either has multiple additative- or proportional error
        terms, the function will raise a ValueError

    Returns
    -------
    res : Union[sympy.Symbol, sympy.Add]
        will return a sympy statement. Either a symbol or Add depending on the
        state of the res
    errors_add_prop : Dict
        A dictionary with two keys. One called "add" containing the additative
        error term (if found, otherwise None) and one called "prop" containing the
        proportional error term (if found, otherwise None)

    """
    errors = []
    
    dv = list(model.dependent_variables.keys())[0]
    expr = sympy.expand(expr)
    terms = sympy.Add.make_args(expr)
    res_found = False
    for term in terms:
        
        if res_found:
            #TODO: check that current term contain epsilon
            error_term = True
        else:
            #FIXME: long runtime for complex piecewise statements
            full_term = full_expression(term, model)
            error_term = False
            all_symbols = full_term.free_symbols.union(term.free_symbols)
            for symbol in all_symbols:
                if str(symbol) in model.random_variables.epsilons.names:
                    error_term = True
    
        if error_term:
            errors.append((term, full_term))
        else:
            if "res" not in locals():
                res = term
                res_found = True
                    

    errors_add_prop = {"add": None, "prop": None}
    prop = False
    res_alias = []
    for s in res.free_symbols:
        all_a = find_aliases(s, model)
        for a in all_a:
            if a not in res_alias:
                res_alias.append(a)
    for t in errors:
        term = t[0]
        full_term = t[1]
        for symbol in full_term.free_symbols:
            for ali in find_aliases(symbol, model).union(term.free_symbols):
                if ali in res_alias:
                    prop = True
                    # Remove the resulting symbol from the error term
                    term = term.subs(ali, 1)
                    

        if prop:
            if errors_add_prop["prop"] is None:
                errors_add_prop["prop"] = term
            else:
                errors_add_prop["prop"] = errors_add_prop["prop"] + term
        else:
            if errors_add_prop["add"] is None:
                errors_add_prop["add"] = term
            else:
                errors_add_prop["add"] = errors_add_prop["add"] + term

    for pair in errors_add_prop.items():
        key = pair[0]
        term = pair[1]
        if term is not None:
            term = convert_eps_to_sigma(term, model)
        errors_add_prop[key] = term
    
    # Check if error is on the form Y = F + W * EPS(1)
    errors_add_prop = var_3_check(errors_add_prop, res, model, cg)
    
    return res, errors_add_prop

def var_3_check(error: dict,
                res,
                model: pharmpy.model.Model,
                cg: CodeGenerator) -> dict:
    sigmas = get_sigmas(model)
    sigma = sigmas[0]
    if len(sigmas) == 1 and sigma.init == 1:
        if (error["add"] and not error["prop"]) or (error["prop"] and not error["add"]):
            if error["add"]:
                term = error["add"]
            else:
                term = error["prop"]
            term_symbols = term.free_symbols
            
            if len(term_symbols) == 2 and sigma.symbol in term_symbols:
                for symbol in term_symbols:
                    if symbol != sigma.symbol:
                        w = symbol
                
                w_full = full_expression(w, model)
                if isinstance(w_full, sympy.Symbol):
                    return error
            
                if w_full.is_Piecewise:
                    for expr, cond in w_full.args:
                        if cond == sympy.true:
                            w_full = expr
                
                if isinstance(w_full, sympy.Pow):
                    new_add, new_prop = var_3_check_find_term(w_full.args[0], res, model)
                else:
                    new_add, new_prop = var_3_check_find_term(w_full, res, model)
                
                error["add"] = new_add
                error["prop"] = new_prop
                
                # remove SIGMA definition due to redundancy
                cg.remove(f"{sigma.name} <- fixed({sigma.init})")
                return error
    return error

def var_3_check_find_term(expr, res, model):
    res_full = full_expression(res, model)
    
    new_add = None
    new_prop = None
    prop = False
    for term in sympy.Add.make_args(expr):
        for symbol in term.free_symbols:
            if symbol in res_full.free_symbols:
                prop = True
                
        if prop == True and new_prop is None:
            for symbol in term.free_symbols:
                if symbol not in res_full.free_symbols:        
                    new_prop = symbol
        else:
            new_add = list(term.free_symbols)[0]

    return new_add, new_prop
    
def add_error_model(
    model: pharmpy.model.Model,
    cg: CodeGenerator,
    expr: sympy.Symbol or sympy.Add,
    error: dict,
    symbol: str,
) -> None:
    """
    Adds one or multiple error variables to the model code if needed. This is only needed if
    the error model follows non-convential syntax. If the error model follows
    convential format. Nothing is added

    Parameters
    ----------
    model : pharmpy.model.Model
        Connected pharmpy model object
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    expr : sympy.Symbol or sympy.Add
        Expression for the dependent variable.
    error : dict
        Dictionary with additive and proportional error terms.
    symbol : str
        Symbol of dependent variable.

    Raises
    ------
    ValueError
        Will raise ValueError if model has defined error model that does not
        match the format of the found error terms.
    """
    cg.add(f'{symbol} <- {expr}')

    # Add term for the additive and proportional error (if exist)
    # as solution for nlmixr error model handling
    if error["add"]:
        if not isinstance(error["add"], sympy.Symbol):
            n = 0
            args = error_args(error["add"])

            for term in args:
                if n == 0:
                    cg.add(f'add_error <- {term}')
                else:
                    cg.add(f'add_error_{n} <- {term}')
                n += 1

    if error["prop"]:
        if not isinstance(error["prop"], sympy.Symbol):
            n = 0
            args = error_args(error["prop"])

            for term in args:
                if n == 0:
                    cg.add(f'prop_error <- {term}')
                else:
                    cg.add(f'prop_error_{n} <- {term}')
                n += 1


def add_error_relation(cg: CodeGenerator, error: Dict, symbol: str) -> None:
    """
    Add a code line in nlmixr2 deciding the error model of the resulting prediction

    Parameters
    ----------
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    error : dict
        Dictionary with additive and proportional error terms.
    symbol : str
        Symbol of dependent variable.
    """
    
    # Add the actual error model depedent on the previously
    # defined variable add_error and prop_error
    error_relation = ""

    first = True
    if error["add"] is not None:
        if isinstance(error["add"], sympy.Symbol):
            add_error = error["add"]
            if first:
                error_relation += f'add({add_error})'
                first = False
            else:
                error_relation += " + " + f'add({add_error})'
        else:
            n = 0
            last = len(error_args(error["add"])) - 1
            for n in range(last + 1):
                if n == 0:
                    error_relation += "add(add_error)"
                    if n != last:
                        error_relation += " + "
                    first = False
                else:
                    error_relation += f"add(add_error_{n})"
                    if n != last:
                        error_relation += " + "

    if error["prop"] is not None:
        if isinstance(error["prop"], sympy.Symbol):
            prop_error = error["prop"]
            if first:
                error_relation += f'prop({prop_error})'
                first = False
            else:
                error_relation += " + " + f'prop({prop_error})'
        else:
            n = 0
            last = len(error_args(error["prop"])) - 1
            for n in range(last + 1):
                if n == 0:
                    if first:
                        error_relation += "prop(prop_error)"
                    else:
                        error_relation += " + " + 'prop(prop_error)'
                    if n != last:
                        error_relation += " + "
                else:
                    error_relation += f"prop(prop_error_{n})"
                    if n != last:
                        error_relation += " + "

    if error_relation == "":
        cg.add("")
        cg.add("# Fake error term since error model could not be determined")
        cg.add('# Note that conditional error models cannot be converted')
        cg.add("FAKE_ERROR <- 0.0")
        error_relation += "FAKE_ERROR"

    cg.add(f'{symbol} ~ {error_relation}')


def error_args(s: Union[sympy.Add, sympy.Symbol, sympy.Mul]) -> list:
    """
    Find all additive terms in a given expression and return all terms in an
    iterable list

    Parameters
    ----------
    s : Union[sympy.Add, sympy.symbol, sympy.Mul]
        Expression to extract terms from.

    Returns
    -------
    args : list
        List with all terms from the given expression.

    """
    if isinstance(s, sympy.Add):
        args = s.args
    else:
        args = [s]
    return args


def full_expression(expression: sympy.Expr, model: pharmpy.model.Model) -> sympy.Expr:
    """
    Return the full expression of an expression (used for model statements)

    Parameters
    ----------
    expression : sympy.Expr
        Expression to be expanded.
    model : pharmpy.model.Model
        A pharmpy mode object with the expression as a statement.

    Returns
    -------
    expression : sympy.Expr
        The fully expanded expression

    """
    if len(model.statements.after_odes) == 0:
        statements = model.statements
    else:
        statements = model.statements.after_odes
        
    for statement in reversed(statements):
        expression = subs(expression, {statement.symbol: statement.expression}, simultaneous=True)
    return expression


def find_aliases(symbol: str, model: pharmpy.model, aliases: set = None) -> list:
    """
    Returns a list of all variable names that are the same as the inputed symbol

    Parameters
    ----------
    symbol : str
        The name of the variable to find aliases to.
    model : pharmpy.model
        A model by which the inputed symbol is related to.

    Returns
    -------
    aliases: list
        A list of aliases for the symbol.

    """
    if aliases is None:
        aliases = set([symbol])
    else:
        aliases.add(symbol)
        
    for expr in model.statements.after_odes:
        if symbol == expr.symbol and isinstance(expr.expression, sympy.Symbol):
            if expr.expression not in aliases:
                aliases.union(find_aliases(expr.expression, model, aliases))
        if symbol == expr.symbol and expr.expression.is_Piecewise:
            for e, c in expr.expression.args:
                if isinstance(e, sympy.Symbol):
                    if e not in aliases:
                        aliases.union(find_aliases(e, model, aliases))
        if symbol == expr.expression:
            if expr.symbol not in aliases:
                aliases.union(find_aliases(expr.symbol, model, aliases))
    return aliases


def convert_eps_to_sigma(
    expr: Union[sympy.Symbol, sympy.Mul], model: pharmpy.model.Model
) -> Union[sympy.Symbol, sympy.Mul]:
    """
    Change the use of epsilon names to sigma names instead. Mostly used for
    converting NONMEM format to nlmxir2

    Parameters
    ----------
    expr : Union[sympy.Symbol,sympy.Mul]
        A sympy term to change a variable name in
    model : pharmpy.Model
        A pharmpy model object

    Returns
    -------
    Union[sympy.Symbol,sympy.Mul]
        Same expression as inputed, but with epsilon names changed to sigma.

    """
    eps_to_sigma = {
        sympy.Symbol(eps.names[0]): sympy.Symbol(str(eps.variance))
        for eps in model.random_variables.epsilons
    }
    return expr.subs(eps_to_sigma)


def convert_piecewise(
    piecewise: sympy.Piecewise, cg: CodeGenerator, model: pharmpy.model.Model
) -> None:
    """
    For an expression of the dependent variable contating a piecewise statement
    this function will convert the expression to an if/else if/else statement
    compatible with nlmixr.
    NOTE(!) nlmixr2 conversion cannot handle conditional error models, which are ignored.

    Parameters
    ----------
    piecewise : sympy.Piecewise
        A sympy expression contining made up of a Piecewise statement
    cg : CodeGenerator
        CodeGenerator class object for creating code
    model : pharmpy.Model
        Pharmpy model object

    """
    first = True
    for expr, cond in piecewise.expression.args:
        if first:
            cg.add(f'if ({cond}){{')
            expr, error = find_term(model, expr, cg)
            cg.add(f'{piecewise.symbol} <- {expr}')
            cg.add('}')
            first = False
        else:
            if cond is not sympy.S.true:
                cg.add(f'else if ({cond}){{')
                expr, error = find_term(model, expr, cg)
                cg.add(f'{piecewise.symbol} <- {expr}')
                cg.add('}')
            else:
                cg.add('else {')
                expr, error = find_term(model, expr, cg)
                cg.add(f'{piecewise.symbol} <- {expr}')
                cg.add('}')

    # FIXME : Add error relation where the error model is conditional
    # add_error_relation(cg, error, piecewise.symbol)
