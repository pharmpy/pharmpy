"""
This module serves the purpose of collecting functions needed for modifying 
NONMEM created code in order to fit the format for nlmixr2. These functions 
are to be used during conversion of NONMEM models to nlmixr2.
"""

from pharmpy.deps import sympy, sympy_printing
import pharmpy.model
import re

class CodeGenerator:
    def __init__(self):
        self.indent_level = 0
        self.lines = []

    def indent(self):
        self.indent_level += 4

    def dedent(self):
        self.indent_level -= 4

    def add(self, line):
        self.lines.append(f'{" " * self.indent_level}{line}')

    def empty_line(self):
        self.lines.append('')

    def __str__(self):
        return '\n'.join(self.lines)


def name_mangle(s:str) -> str:
    """
    Changes the format of parameter name to avoid using parenthesis

    Parameters
    ----------
    s : str
        Parameter name to be changed

    Returns
    -------
    str
        Parameter name with parenthesis removed
    
    Example
    -------
    name_mangle("ETA(1)")
    -> "ETA1"

    """
    return s.replace('(', '').replace(')', '').replace(',', '_')

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
            add_error_model(cg, expr, error, piecewise.symbol)
            cg.add('}')
            first = False
        else:
            if cond is not sympy.S.true:
                cg.add(f'else if ({cond}){{')
                expr, error = find_term(model, expr)
                add_error_model(cg, expr, error, piecewise.symbol)
                cg.add('}')
            else:
                cg.add('else {')
                expr, error = find_term(model, expr)
                add_error_model(cg, expr, error, piecewise.symbol)
                cg.add('}')
    
    add_error_relation(cg, error, piecewise.symbol)

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
        error_term = False
        for symbol in term.free_symbols:
            if str(symbol) in model.random_variables.epsilons.names:
                error_term = True
            
        if error_term:
            errors.append(term)
        else:
            if "res"  not in locals():
                res = term
            else:
                res = res + term
    
    errors_add_prop = {"add": None, "prop": None}
    
    prop = False
    res_alias = find_aliases(res, model)
    for term in errors:
        for symbol in term.free_symbols:
            for ali in find_aliases(symbol, model):
                if ali in res_alias:
                    prop = True
                    # Remove the symbol that was found
                    # and substitute res to that symbol to avoid confusion
                    term = term.subs(symbol,1)
                    res = symbol
            
        if prop:
            if errors_add_prop["prop"] is None:
                errors_add_prop["prop"] = term    
            else:
                raise ValueError("Multiple proportional error terms found. Check format of error model")
        else:
            if errors_add_prop["add"] is None:
                errors_add_prop["add"] = term
            else:
                raise ValueError("Multiple additive error term found. Check format of error model")
    
    for pair in errors_add_prop.items():
        key = pair[0]
        term = pair[1]
        if term != None:
            term = convert_eps_to_sigma(term, model)
        errors_add_prop[key] = term
        
    return res, errors_add_prop

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
        DIf known error model, this can be set to force the error model to be 
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
    
    if force_add:
        assert error["prop"] is None
        
        if error["add"]:
            if not isinstance(error["add"], sympy.Symbol):
                cg.add(f'add_error <- {error["add"]}')
        else:
            raise ValueError("Model should have additive error but no such error was found.")
    elif force_prop:
        assert error["add"] is None
        
        if error["prop"]:
            if not isinstance(error["prop"], sympy.Symbol):
                cg.add(f'prop_error <- {error["prop"]}')
        else:
            raise ValueError("Model should have proportional error but no such error was found.")
    elif force_comb:
        assert error["add"] is not None and error["prop"] is not None
        
        if error["add"]:
            if not isinstance(error["add"], sympy.Symbol):
                cg.add(f'add_error <- {error["add"]}')
        else:
            raise ValueError("Model should have additive error but no such error was found.")
            
        if error["prop"]:
            if not isinstance(error["prop"], sympy.Symbol):
                cg.add(f'prop_error <- {error["prop"]}')
        else:
            raise ValueError("Model should have proportional error but no such error was found.")
    else:
        # Add term for the additive and proportional error (if exist)
        # as solution for nlmixr error model handling
        if error["add"]:
            if not isinstance(error["add"], sympy.Symbol):
                cg.add(f'add_error <- {error["add"]}')
        if error["prop"]:
            if not isinstance(error["prop"], sympy.Symbol):
                cg.add(f'prop_error <- {error["prop"]}')
        
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
    if isinstance(error["add"], sympy.Symbol):
        add_error = error["add"]
    else:
        add_error = "add_error"
    if isinstance(error["prop"], sympy.Symbol):
        prop_error = error["prop"]
    else:
        prop_error = "prop_error"
    
        
    if error["add"] and error["prop"]:
        cg.add(f'{symbol} ~ add({add_error}) + prop({prop_error})')
    elif error["add"] and not error["prop"]:
        cg.add(f'{symbol} ~ add({add_error})')
    elif not error["add"] and error["prop"]:
        cg.add(f'{symbol} ~ prop({prop_error})')
        
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
    
def add_ini_parameter(cg: CodeGenerator, parameter: sympy.Symbol, boundary: bool = False) -> None:
    """
    Add a parameter to the ini block in nlmixr2. This is performed for theta
    and sigma parameter values as they are handled in the same manner.

    Parameters
    ----------
    cg : CodeGenerator
        Codegenerator object holding the code to be added to.
    parameter : sympy.Symbol
        The parameter to be added. Either theta or sigma
    boundary : bool, optional
        Decide if the parameter should be added with or without parameter 
        boundries. The default is False.

    Returns
    -------
    None
        Modifies the given CodeGenerator object. Returns nothing

    """
    parameter_name = name_mangle(parameter.name)
    if parameter.fix:
        cg.add(f'{parameter_name} <- fixed({parameter.init})')
    else:
        limit = 1000000.0
        if boundary:
            if parameter.lower > -limit and parameter.upper < limit:
                cg.add(f'{parameter_name} <- c({parameter.lower}, {parameter.init}, {parameter.upper})')
            elif parameter.lower == -limit and parameter.upper < limit:
                cg.add(f'{parameter_name} <- c(-Inf, {parameter.init}, {parameter.upper})')
            elif parameter.lower > -limit and parameter.upper == limit:
                cg.add(f'{parameter_name} <- c({parameter.lower}, {parameter.init}, Inf)')
            else:
                cg.add(f'{parameter_name} <- {parameter.init}')
        else:
            cg.add(f'{parameter_name} <- {parameter.init}')
            
def change_same_time(model: pharmpy.model) -> pharmpy.model:
    """
    Force dosing to happen after observation, if bolus dose is given at the
    exact same time.

    Parameters
    ----------
    model : pharmpy.model
        A pharmpy.model object

    Returns
    -------
    model : TYPE
        The same model with a changed dataset.

    """
    dataset = model.dataset
    time = dataset["TIME"]
    for index, row in dataset.iterrows():
        if index != 0:
            if (row["ID"] == dataset.loc[index-1]["ID"] and
                row["TIME"] == dataset.loc[index-1]["TIME"] and
                row["EVID"] not in [0,3] and 
                dataset.loc[index-1]["EVID"] == 0):
                time[index] = row["TIME"] + 10**-6
    model.dataset["TIME"] = time
    return model
                
                
                
                
                