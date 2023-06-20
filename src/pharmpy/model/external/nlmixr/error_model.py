from typing import Union

import pharmpy.model
from pharmpy.deps import sympy
from pharmpy.internals.expr.subs import subs
from pharmpy.modeling import get_thetas


class res_error_term:
    def __init__(self, model, expr):
        self.model = model
        self.expr = sympy.expand(expr)

        self.only_piecewise = None
        self.is_only_piecewise()

        self.res = None
        self.res_alias = None
        self.add = error()
        self.prop = error()
        self.find_term()

    def create_res_alias(self):
        if self.res is not None:
            res_alias = set()
            for s in self.res.free_symbols:
                all_a = find_aliases(s, self.model)
                for a in all_a:
                    if a not in res_alias:
                        res_alias.add(a)
            self.res_alias = res_alias

    def dependencies(self):
        dependencies = set()
        if self.add.expr is not None:
            dependencies.update(self.add.dependencies)
        elif self.prop.expr is not None:
            dependencies.update(self.prop.dependencies)
        return dependencies

    def find_term(self):
        # NEED TO BE
        # VAR 1 : y = F + F*eps + eps
        # VAR 2 : y = F + F*theta*eps + theta*eps (EPS FIX 1)
        # VAR 3 : y = F + W*eps (W = sqrt(theta....)) (EPS FIX 1)
        # VAR 4 : y = F + W*eps (W = F*theta + theta) (EPS FIX 1)
        # ELSE conversion might be incorrect.
        terms = sympy.Add.make_args(self.expr)
        # Assert that it follows the above set of format rules
        assert len(terms) <= 3

        errors = []
        for term in terms:
            error_term = False
            full_term = full_expression(term, self.model)
            for factor in sympy.Mul.make_args(term):
                full_factor = full_expression(factor, self.model)
                all_symbols = full_factor.free_symbols.union(factor.free_symbols)
                for symbol in all_symbols:
                    if str(symbol) in self.model.random_variables.epsilons.names:
                        sigma = convert_eps_to_sigma(symbol, self.model)
                        if self.model.parameters[str(sigma)].init == 1.0:
                            if self.model.parameters[str(sigma)].fix:
                                term = term.subs(factor, 1)
                        if factor != symbol:
                            sigma_alias = factor
                        else:
                            sigma_alias = None
                        error_term = True

            if error_term:
                errors.append(
                    {
                        "term": term,
                        "full_term": full_term,
                        "sigma": sigma,
                        "sigma_alias": sigma_alias,
                    }
                )
            else:
                if self.res is None:
                    self.res = term
                    self.create_res_alias()
                else:
                    # FIXME : Should this be allowed??
                    self.res += term
                    self.create_res_alias()

        if self.res is None:
            print("No resulting term found")
            exit
        elif len(errors) > 2:
            print("Too many error terms found. Will try to translate either way.")

        for t in errors:
            prop = False
            ali_removed = False
            term = t["term"]
            full_term = t["full_term"]
            for symbol in full_term.free_symbols.union(term.free_symbols):
                for ali in find_aliases(symbol, self.model):
                    if ali in self.res_alias:
                        prop = True
                        # Remove the resulting symbol from the error term
                        term = convert_eps_to_sigma(term, self.model)
                        if ali in term.free_symbols:
                            term = term.subs(ali, 1)
                            ali_removed = True
            if prop is True:
                if not ali_removed:
                    term = term / self.res
                if self.prop.expr == 0:
                    self.prop = error(
                        self.model,
                        term,
                        t["sigma"],
                        sigma_alias=t["sigma_alias"],
                        prop=True,
                    )
                else:
                    self.prop.expr = self.prop.expr + term
            else:
                term = convert_eps_to_sigma(term, self.model)
                if self.add.expr == 0:
                    self.add = error(
                        self.model, term, t["sigma"], sigma_alias=t["sigma_alias"], add=True
                    )
                else:
                    self.add.expr = self.add.expr + term

    def is_only_piecewise(self):
        dv = list(self.model.dependent_variables.keys())[0]
        for s in reversed(self.model.statements.after_odes):
            if s.symbol == dv:
                if not s.expression.is_Piecewise:
                    self.only_piecewise = False
                    break

        if self.only_piecewise is None:
            self.only_piecewise = True

    def __str__(self):
        s = ""
        s += str(f"add : {self.add.expr}\n")
        s += str(f"add_sigma : {self.add.sigma}\n")
        s += str(f"prop : {self.prop.expr}\n")
        s += str(f"prop_sigma : {self.prop.sigma}\n")
        s += str(f"Only piecewise : {self.only_piecewise}\n")
        return s


class error:
    def __init__(self, model=None, expr=0, sigma=None, add=False, sigma_alias=None, prop=False):
        self.model = model
        self.expr = expr
        self.sigma = sigma
        self.sigma_alias = sigma_alias
        self.sigma_fix = self.is_sigma_fix()
        self.add = add
        self.prop = prop
        self.dependencies = set()
        self.check_dependecies()

    def is_sigma_fix(self):
        if self.model is not None:
            if self.model.parameters[str(self.sigma)].init == 1.0:
                if self.model.parameters[str(self.sigma)].fix:
                    return True
        else:
            return False

    def check_dependecies(self):
        if (
            self.model is not None
            and self.expr is not None
            and self.model.parameters[self.sigma].init == 1
            and self.sigma_fix
        ):
            accepted_symbols = set([self.sigma, self.sigma_alias])
            thetas = get_thetas(self.model).symbols
            accepted_symbols.update(thetas)
            etas = [sympy.Symbol(i) for i in self.model.random_variables.etas.names]
            accepted_symbols.update(etas)
            for symbol in self.expr.free_symbols:
                if not any([i in accepted_symbols for i in find_aliases(symbol, self.model)]):
                    if is_number(symbol, self.model):
                        accepted_symbols.update([symbol])
                    else:
                        self.dependencies.add(symbol)


def is_number(symbol: sympy.Expr, model: pharmpy.model.Model) -> bool:
    alias = find_aliases(symbol, model)
    for a in alias:
        if a not in model.random_variables.free_symbols:
            a_assign = model.statements.find_assignment(a)
            if a_assign.expression.is_number:
                return True
    return False


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
        # If RES = ALI
        if symbol == expr.symbol and isinstance(expr.expression, sympy.Symbol):
            if expr.expression not in aliases:
                aliases.union(find_aliases(expr.expression, model, aliases))

        # If RES = PIECEWISE or PIECEWISE = RES
        if expr.expression.is_Piecewise:
            for e, c in expr.expression.args:
                if symbol == expr.symbol and isinstance(e, sympy.Symbol):
                    if e not in aliases:
                        aliases.union(find_aliases(e, model, aliases))
                elif symbol == e:
                    if expr.symbol not in aliases:
                        aliases.union(find_aliases(expr.symbol, model, aliases))

        # If ALI = RES
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
    # eps_to_sigma = {
    #    sympy.Symbol(eps.names[0]): sympy.Symbol(str(eps.variance))
    #    for eps in model.random_variables.epsilons
    # }
    eps_to_sigma = {}
    for dist in model.random_variables.epsilons:
        sigma = dist.variance
        if len(dist.names) == 1:
            eps_to_sigma[dist.names[0]] = sigma
        else:
            for row, col, eps in zip(range(sigma.rows), range(sigma.rows + 1), dist.names):
                eps_to_sigma[eps] = sigma[row, col]
    return expr.subs(eps_to_sigma)
