"""
Generic NONMEM code record class.

"""

import lark
import sympy

from pharmpy.data_structures import OrderedSet
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.plugins.nonmem.records.parsers import CodeRecordParser

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

    def logical_expression(self, node):
        t = self.visit_cildren(node)
        if len(t) > 2:
            left, op, right = self.visit_children(node)
            return op(left, right)
        elif len(t) == 2:
            op, expr = self.visit_children(node)
            return op(expr)
        else:
            return t[0]

    def logical_operator(self, node):
        name = str(node)
        if name == '==' or name == '.EQ.':
            return sympy.Eq
        elif name == '/=' or name == '.NE.':
            return sympy.Ne
        elif name == '<=' or name == '.LE.':
            return sympy.Le
        elif name == '>=' or name == '.GE.':
            return sympy.Ge
        elif name == '<' or name == '.LT.':
            return sympy.Lt
        elif name == '>' or name == '.GT.':
            return sympy.Gt
        elif name == '.AND.':
            return sympy.And
        elif name == '.OR.':
            return sympy.Or
        elif name == '.NOT.':
            return sympy.Not

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
    def __init__(self, content, parser_class):
        super().__init__(content, parser_class)
        self.nodes = []
        self._statements = None

    @property
    def statements(self):
        if self._statements is None:
            s = self.assign_statement(self)
            self._statements = ModelStatements(s)
        return self._statements

    @staticmethod
    def assign_statement(tree):
        s = []
        for statement in tree.root.all('statement'):
            node = statement.children[0]
            if node.rule == 'assignment':
                name = str(node.variable)
                expr = ExpressionInterpreter().visit(node.expression)
                ass = Assignment(name, expr)
                s.append(ass)
            elif node.rule == 'logical_if':
                logic_expr = ExpressionInterpreter().visit(node.logical_expression)
                name = str(node.assignment.variable)
                expr = ExpressionInterpreter().visit(node.assignment.expression)
                pw = sympy.Piecewise((expr, logic_expr))
                ass = Assignment(name, pw)
                s.append(ass)
            elif node.rule == 'block_if':
                interpreter = ExpressionInterpreter()
                blocks = []  # [(logic, [(symb1, expr1), ...]), ...]
                symbols = OrderedSet()

                first_logic = interpreter.visit(node.block_if_start.logical_expression)
                first_block = node.block_if_start
                first_symb_exprs = []
                for assign_node in first_block.all('assignment'):
                    name = str(assign_node.variable)
                    first_symb_exprs.append((name, interpreter.visit(assign_node.expression)))
                    symbols.add(name)
                blocks.append((first_logic, first_symb_exprs))

                else_if_blocks = node.all('block_if_elseif')
                for elseif in else_if_blocks:
                    logic = interpreter.visit(elseif.logical_expression)
                    elseif_symb_exprs = []
                    for assign_node in elseif.all('assignment'):
                        name = str(assign_node.variable)
                        elseif_symb_exprs.append((name, interpreter.visit(assign_node.expression)))
                        symbols.add(name)
                    blocks.append((logic, elseif_symb_exprs))

                else_block = node.find('block_if_else')
                if else_block:
                    else_symb_exprs = []
                    for assign_node in else_block.all('assignment'):
                        name = str(assign_node.variable)
                        else_symb_exprs.append((name, interpreter.visit(assign_node.expression)))
                        symbols.add(name)
                    blocks.append((True, else_symb_exprs))

                for symbol in symbols:
                    pairs = []
                    for block in blocks:
                        logic = block[0]
                        for cursymb, expr in block[1]:
                            if cursymb == symbol:
                                pairs.append((expr, logic))
                    pw = sympy.Piecewise(*pairs)
                    ass = Assignment(symbol, pw)
                    s.append(ass)
        return s

    @statements.setter
    def statements(self, statements_raw):
        # TODO: write standardization of statements_raw
        content_original = self._content.strip().split('[\t\n ]')
        content_input = statements_raw.strip().split('[\t\n ]')

        if content_original == content_input:
            print("New statements same as current, no changes made.")
        else:
            statements_old = self._statements

            tree_new = CodeRecordParser(f'\n{statements_raw}')
            s_new = self.assign_statement(tree_new)
            statements_new = ModelStatements(s_new)

            if statements_new == statements_old:  # TODO: write test
                print("New statements same as current, no changes made.")

            statements_updated = []
            for statement in statements_new:
                if statement in statements_old:
                    statements_updated.append(statement)
                    print('\t\tStatement exists.')
                else:
                    if statement.symbol in statements_old.free_symbols:
                        print('\t\tAssignment exists.')
                        statements_updated.append(statement)
                    else:
                        print('\t\tAssignment does not exist.')
                        statements_updated.append(statement)

            self._statements = statements_updated

