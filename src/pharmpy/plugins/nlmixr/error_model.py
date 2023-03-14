import pharmpy.model
from pharmpy.deps import sympy
from .CodeGenerator import CodeGenerator

from .sanity_checks import print_warning

def find_term(model: pharmpy.model, expr: sympy.Add) -> tuple[sympy.Symbol or sympy.Add, dict]:
    """
    For a given expression for the dependent variable, find the terms 
    connected to the actual result and the terms connected to the error model.

    Parameters
    ----------
    model : pharmpy.model
        A pharmpy model object
    expr : sympy.Add
        An expression for the dependent variable. Should be a sympy.Add statement

    Raises
    ------
    ValueError
        If the model either has multiple additative- or proportional error 
        terms, the function will raise a ValueError

    Returns
    -------
    res : sympy.Symbol or sympy.Add
        will return a sympy statement. Either a symbol or Add depending on the 
        state of the res
    errors_add_prop : dict
        A dictionary with two keys. One called "add" containing the additative
        error term (if found, otherwise None) and one called "prop" containing the 
        proportional error term (if found, otherwise None)

    """
    errors = []
    
    terms = sympy.Add.make_args(expr)
    
    for term in terms:
        full_term = full_expression(term, model)
        error_term = False
        for symbol in full_term.free_symbols:
            if str(symbol) in model.random_variables.epsilons.names:
                error_term = True
            
        if error_term:
            errors.append((term, full_term))
        else:
            if "res"  not in locals():
                res = term
            else:
                res = res + term
    
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
            for ali in find_aliases(symbol, model):
                if ali in res_alias:
                    prop = True
                    # Remove the resulting symbol from the error term
                    term = term.subs(symbol,1)
            
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
        if term != None:
            term = convert_eps_to_sigma(term, model)
        errors_add_prop[key] = term
    
    return res, errors_add_prop

def add_error_model(cg: CodeGenerator,
                    expr: sympy.Symbol or sympy.Add,
                    error: dict,
                    symbol: str,
                    force_add: bool = False,
                    force_prop: bool = False,
                    force_comb: bool = False
                    ) -> None:
    """
    Adds an error parameter to the model code if needed. This is only needed if
    the error model follows non-convential syntax. If the error model follows 
    convential format. Nothing is added

    Parameters
    ----------
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    expr : sympy.Symbol or sympy.Add
        Expression for the dependent variable.
    error : dict
        Dictionary with additive and proportional error terms.
    symbol : str
        Symbol of dependent variable.
    force_add : bool, optional
        If known error model, this can be set to force the error model to be 
        an additive one. The default is False.
    force_prop : bool, optional
        If known error model, this can be set to force the error model to be 
        an proportional one. The default is False.
    force_comb : bool, optional
        If known error model, this can be set to force the error model to be 
        an combination based. The default is False.

    Raises
    ------
    ValueError
        Will raise ValueError if model has defined error model that does not
        match the format of the found error terms.

    Returns
    -------
    None
        Modifies the given CodeGenerator object. Returns nothing
    
    Example
    -------
    TODO
        
    """
    cg.add(f'{symbol} <- {expr}')
    
    # Add term for the additive and proportional error (if exist)
    # as solution for nlmixr error model handling
    print(error)
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
        
def add_error_relation(cg: CodeGenerator, error: dict, symbol: str) -> None:
    """
    Add a code line in nlmixr2 deciding the error model of the dependent variable

    Parameters
    ----------
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    error : dict
        Dictionary with additive and proportional error terms.
    symbol : str
        Symbol of dependent variable.

    Returns
    -------
    None
        Modifies the given CodeGenerator object. Returns nothing

    """
    # Add the actual error model depedent on the previously
    # defined variable add_error and prop_error
    error_relation = ""
    
    first = True
    if error["add"] != None:
        if isinstance(error["add"], sympy.Symbol):
            add_error = error["add"]
            if first:
                error_relation += add_error
                first = False
            else:
                error_relation += " + "+add_error
        else:
            n = 0
            last = len(error_args(error["add"])) - 1
            for n in range(last+1):
                if n == 0:
                    error_relation += "add(add_error)"
                    if n != last:
                        error_relation += " + "
                else:
                    error_relation += f"add(add_error_{n})"
                    if n != last:
                        error_relation += " + "
    
    if error["prop"] != None:
        if isinstance(error["prop"], sympy.Symbol):
            prop_error = error["prop"]
            if first:
                error_relation += prop_error
                first = False
            else:
                error_relation += " + "+prop_error
        else:
            n = 0
            last = len(error_args(error["prop"])) - 1
            for n in range(last+1):                
                if n == 0:
                    error_relation += "prop(prop_error)"
                    if n != last:
                        error_relation += " + "
                else:
                    error_relation += f"prop(prop_error_{n})"
                    if n != last:
                        error_relation += " + "
    
    if error_relation == "":
        print_warning("Error model could not be determined. Note that conditional error models cannot be converted.\nWill add fake error term.")
        cg.add("FAKE_ERROR <- 0.0")
        error_relation += "FAKE_ERROR"
        
    cg.add(f'{symbol} ~ {error_relation}')
    
def error_args(s):
    if isinstance(s, sympy.Add):
        args = s.args
    else:
        args = [s]
    return args

def full_expression(expression, model):
    from pharmpy.internals.expr.subs import subs
    for statement in reversed(model.statements.after_odes):
        expression = subs(
                expression, {statement.symbol: statement.expression}, simultaneous=True
                )
    return expression
        
def find_aliases(symbol:str, model: pharmpy.model) -> list:
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
    list
        A list of aliases for the symbol.

    """
    aliases = [symbol]
    for expr in model.statements.after_odes:
        if symbol == expr.symbol and isinstance(expr.expression, sympy.Symbol):
            aliases.append(expr.expression)
        if symbol == expr.symbol and expr.expression.is_Piecewise:
            for e, c in expr.expression.args:
                if isinstance(e, sympy.Symbol):
                    aliases.append(e)
    return aliases

def convert_eps_to_sigma(expr: sympy.Symbol or sympy.Mul, model: pharmpy.model) -> sympy.Symbol or sympy.Mul:
    """
    Change the use of epsilon names to sigma names instead. Mostly used for 
    converting NONMEM format to nlmxir2

    Parameters
    ----------
    expr : sympy.Symbol or sympy.Mul
        A sympy term to change a variable name in
    model : pharmpy.Model
        A pharmpy model object

    Returns
    -------
    TYPE : sympy.Symbol or sympy.Mul
        Same expression as inputed, but with epsilon names changed to sigma.

    """
    eps_to_sigma = {sympy.Symbol(eps.names[0]): sympy.Symbol(str(eps.variance)) for eps in model.random_variables.epsilons}
    return expr.subs(eps_to_sigma)

def convert_piecewise(piecewise: sympy.Piecewise, cg: CodeGenerator, model: pharmpy.model) -> None:
    """
    For an expression of the dependent variable contating a piecewise statement 
    this function will convert the expression to an if/else if/else statement 
    compatible with nlmixr.

    Parameters
    ----------
    piecewise : sympy.Piecewise
        A sympy expression contining made up of a Piecewise statement
    cg : CodeGenerator
        CodeGenerator class object for creating code
    model : pharmpy.Model
        Pharmpy model object

    Returns
    -------
    None
        CodeGenerator object is modified. Nothing is returned

    """
    first = True
    for expr, cond in piecewise.expression.args:
        if first:
            cg.add(f'if ({cond}){{')
            expr, error = find_term(model, expr)
            cg.add(f'{piecewise.symbol} <- {expr}')
            cg.add('}')
            first = False
        else:
            if cond is not sympy.S.true:
                cg.add(f'else if ({cond}){{')
                expr, error = find_term(model, expr)
                cg.add(f'{piecewise.symbol} <- {expr}')
                cg.add('}')
            else:
                cg.add('else {')
                expr, error = find_term(model, expr)
                cg.add(f'{piecewise.symbol} <- {expr}')
                cg.add('}')
    
    #add_error_relation(cg, error, piecewise.symbol)