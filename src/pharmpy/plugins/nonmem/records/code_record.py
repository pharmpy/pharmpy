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

    factor = expression
    term = expression

    def operator(self, node):
        return str(node)

    def number(self, node):
        return sympy.Float(str(node))

    def symbol(self, node):
        return sympy.Symbol(str(node))


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
