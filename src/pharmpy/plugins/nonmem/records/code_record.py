"""
Generic NONMEM code record class.

"""

import copy
import lark
import sympy

from pharmpy.data_structures import OrderedSet
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

    @staticmethod
    def logical_operator(node):
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

    @staticmethod
    def intrinsic_func(node):
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

    @staticmethod
    def operator(node):
        return str(node)

    @staticmethod
    def number(node):
        try:
            return sympy.Integer(str(node))
        except ValueError:
            return sympy.Float(str(node))

    @staticmethod
    def symbol(node):
        return sympy.Symbol(str(node), real=True)


class CodeRecord(Record):
    def __init__(self, content, parser_class):
        super().__init__(content, parser_class)
        self.nodes = []
        self._statements = None
        self._statements_updated = False

    @property
    def statements(self):
        if self._statements is None:
            self._statements = self.assign_statements()
        return self._statements

    @statements.setter
    def statements(self, statements_new):
        statements_original = copy.deepcopy(self._statements)
        statements_updated = copy.deepcopy(self._statements)

        if statements_new == statements_original:  # TODO: warn instead of print
            print("New statements same as current, no changes made.")

        index_original = 0

        for index_new, statement_new in enumerate(statements_new):
            try:
                if statement_new != statements_original[index_original]:
                    try:
                        index_found = statements_original.index(statement_new, index_original)
                    except ValueError:
                        index_found = None

                    if index_found is not None:
                        print("removing...")
                        statements_updated = self.remove_statements(statements_updated,
                                                                    index_original,
                                                                    index_found)
                        index_original = index_found

                index_original += 1
            except IndexError:
                pass

        self._statements = statements_updated
        self._statements_updated = True

    def remove_statements(self, statements_copy, index_start, index_end):
        for i in range(index_start, index_end):
            statement_to_remove = self._statements[i]
            statements_copy.remove(statement_to_remove)
            print(f'{statement_to_remove} has been removed!')

        return statements_copy

    def get_node(self, statement):
        for node in self.nodes:
            symbol = str(statement.symbol)
            expression = str(statement.expression).replace(' ', '')
            if str(node.eval) == f'{symbol}={expression}':
                return node
        return None

    def assign_statements(self):
        s = []
        for statement in self.root.all('statement'):
            node = statement.children[0]
            self.nodes.append(node)
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

        statements = ModelStatements(s)
        return statements
