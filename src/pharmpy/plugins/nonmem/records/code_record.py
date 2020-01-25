"""
Generic NONMEM code record class.

"""

import lark
import sympy

from pharmpy.statements import Assignment, ModelStatements

from .record import Record


class ExpressionInterpreter(lark.visitors.Interpreter):
    def visit_children(self, tree):
        """Does not visit tokens
        """
        return [self.visit(child) for child in tree.children if isinstance(child, lark.Tree)]

    def expression(self, node):
        t = self.visit_children(node)
        if len(t) > 2:
            op = t[1]
            terms = t[0::2]
            if op == '+':
                expr = sympy.Add(*terms)
            elif op == '-':
                expr = terms[0]
                for term in terms[1:]:
                    expr -= term
            elif op == '*':
                expr = sympy.Mul(*terms)
            elif op == '/':
                expr = terms[0]
                for term in terms[1:]:
                    expr /= term
        else:
            expr = t[0]
        return expr

    def func(self, node):
        func, expr = self.visit_children(node)
        return func(expr)

    def func2(self, node):
        a, p = self.visit_children(node)
        return sympy.Mod(a, p)

    def intrinsic_func(self, node):
        name = str(node)
        if name == "EXP":
            return sympy.exp
        elif name == "LOG":
            return sympy.log
        elif name == "LOG10":
            return lambda x: sympy.log(x, 10)
        elif name == "SQRT":
            return sympy.sqrt
        elif name == "SIN":
            return sympy.sin
        elif name == "COS":
            return sympy.cos
        elif name == "ABS":
            return sympy.Abs
        elif name == "TAN":
            return sympy.tan
        elif name == "ASIN":
            return sympy.asin
        elif name == "ACOS":
            return sympy.acos
        elif name == "ATAN":
            return sympy.atan
        elif name == "INT":
            return lambda x: sympy.sign(x) * sympy.floor(sympy.Abs(x))
        elif name == "GAMLN":
            return sympy.loggamma

    def power(self, node):
        b, e = self.visit_children(node)
        return b**e

    def operator(self, node):
        return str(node)

    def number(self, node):
        try:
            return sympy.Integer(str(node))
        except ValueError:
            return sympy.Float(str(node))

    def symbol(self, node):
        return sympy.Symbol(str(node), real=True)


class CodeRecord(Record):
    @property
    def statements(self):
        s = []
        for node in self.root.all('assignment'):
            name = str(node.variable)
            expr = ExpressionInterpreter().visit(node.expression)
            ass = Assignment(name, expr)
            s.append(ass)
        return ModelStatements(s)
