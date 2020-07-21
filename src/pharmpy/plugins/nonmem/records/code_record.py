"""
Generic NONMEM code record class.

"""

import copy
import warnings

import lark
import sympy

from pharmpy.data_structures import OrderedSet
from pharmpy.plugins.nonmem.records.parsers import CodeRecordParser
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
        self.nodes_updated = []
        self._statements = None
        self._statements_updated = False

    @property
    def statements(self):
        if self._statements is not None:
            return self._statements

        statements = self._assign_statements()
        self._statements = statements

        return copy.deepcopy(statements)

    @statements.setter
    def statements(self, statements_new):
        statements_original = copy.deepcopy(self.statements)
        self.nodes_updated = copy.deepcopy(self.nodes)
        root_updated = copy.deepcopy(self.root)

        if statements_new == statements_original:
            warnings.warn('New statements same as current, no changes made.')
        else:
            index_original = 0

            for index_new, statement_new in enumerate(statements_new):
                if index_original == len(statements_original):
                    self._add_statement(root_updated, None, statement_new)
                elif statement_new != statements_original[index_original]:
                    if statement_new.symbol == statements_original[index_original].symbol or \
                            len(statements_original) == 1 and len(statements_new) == 1:
                        self._remove_statements(root_updated, index_original, index_original)

                    try:
                        index_statement = statements_original.index(statement_new,
                                                                    index_original)
                        index_last_removed = index_statement - 1
                    except ValueError:
                        index_last_removed = None

                    if index_last_removed is None:
                        self._add_statement(root_updated, index_new, statement_new)
                    else:
                        self._remove_statements(root_updated, index_original,
                                                index_last_removed)

                        index_original = index_last_removed

                elif index_new == len(statements_new) - 1:
                    self._remove_statements(root_updated, index_original+1,
                                            len(statements_original)-1)

                index_original += 1

        self.root = root_updated
        self._statements = statements_new
        self._statements_updated = True

    def _remove_statements(self, root_updated, index_remove_start, index_remove_end):
        for i in range(index_remove_start, index_remove_end+1):
            statement_to_remove = self.statements[i]
            node = self._get_node(statement_to_remove)
            self.nodes_updated.remove(node)
            root_updated.remove_node(node)

    # Creating node does not work for if-statements
    def _add_statement(self, root_updated, index_insert, statement):
        node_tree = CodeRecordParser(f'\n{str(statement).replace(":", "")}').root
        node = node_tree.all('statement')[0]

        if isinstance(index_insert, int) and index_insert >= len(self.nodes_updated):
            index_insert = None

        if index_insert is None:
            self.nodes_updated.append(node)
            root_updated.add_node(node)
        else:
            node_following = self.nodes_updated[index_insert]
            self.nodes_updated.insert(index_insert, node)
            root_updated.add_node(node, node_following)

    def _get_node(self, statement):
        index_statement = self.statements.index(statement)
        return self.nodes[index_statement]

    def _assign_statements(self):
        s = []
        for statement in self.root.all('statement'):
            node = statement.children[0]
            self.nodes.append(statement)
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
