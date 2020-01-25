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
        if len(t) == 3:
            left, op, right = t
            if op == '+':
                expr = left + right
            elif op == '-':
                expr = left - right
        else:
            expr = t[0]
        return expr

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
