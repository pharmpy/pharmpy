"""
Generic NONMEM code record class.

"""

import copy
import re

import lark
import sympy
from sympy import Piecewise

from pharmpy import data
from pharmpy.data_structures import OrderedSet
from pharmpy.parse_utils.generic import NoSuchRuleException
from pharmpy.plugins.nonmem.records.parsers import CodeRecordParser
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.symbols import real

from .record import Record


class ExpressionInterpreter(lark.visitors.Interpreter):
    def visit_children(self, tree):
        """Does not visit tokens
        """
        return [self.visit(child) for child in tree.children if isinstance(child, lark.Tree)]

    def expression(self, node):
        t = self.visit_children(node)

        if bool(node.find('UNARY_OP')) and str(node.tokens[0]) == '-':
            unary_factor = -1
        else:
            unary_factor = 1

        if len(t) > 2:
            ops = t[1::2]
            terms = t[2::2]
            expr = unary_factor * t[0]
            for op, term in zip(ops, terms):
                if op == '+':
                    expr += term
                elif op == '-':
                    expr -= term
                elif op == '*':
                    expr *= term
                elif op == '/':
                    expr /= term
        else:
            expr = unary_factor * t[0]
        return expr

    def logical_expression(self, node):
        t = self.visit_cildren(node)
        if len(t) > 2:
            ops = t[1::2]
            terms = t[2::2]
            expr = t[0]
            for op, term in zip(ops, terms):
                expr = op(expr, term)
            return expr
        elif len(t) == 2:
            op, expr = self.visit_children(node)
            return op(expr)
        else:
            return t[0]

    @staticmethod
    def logical_operator(node):
        name = str(node).upper()
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
        name = str(node).upper()
        if name == "EXP" or name == "DEXP":
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
        elif name == "PHI":
            return lambda x: (1 + sympy.erf(x) / sympy.sqrt(2)) / 2

    def power(self, node):
        b, e = self.visit_children(node)
        return b**e

    @staticmethod
    def operator(node):
        return str(node)

    @staticmethod
    def number(node):
        s = str(node)
        try:
            return sympy.Integer(s)
        except ValueError:
            s = s.replace('d', 'E')     # Fortran special format
            s = s.replace('D', 'E')
            return sympy.Float(s)

    @staticmethod
    def symbol(node):
        name = str(node).upper()
        if name.startswith('ERR('):
            name = 'EPS' + name[3:]
        symb = real(name)
        return symb


class CodeRecord(Record):
    def __init__(self, content, parser_class):
        super().__init__(content, parser_class)
        self.nodes = []
        self._nodes_updated = []
        self._root_updated = None
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
        statements_past = copy.deepcopy(self.statements)
        self._nodes_updated = copy.deepcopy(self.nodes)
        self._root_updated = copy.deepcopy(self.root)

        if statements_new != statements_past:
            index_past = 0
            last_index_past = len(statements_past) - 1
            last_index_new = len(statements_new) - 1

            for index_new, s_new in enumerate(statements_new):
                if index_past == len(statements_past):      # Add rest of new statements
                    if self._get_node(s_new) is None:
                        self._add_statement(index_past, s_new)
                    continue
                elif len(statements_past) == 1 and len(statements_new) == 1:
                    self._replace_statement(0, s_new)
                    break

                s_past = statements_past[index_past]
                if s_new != s_past:
                    if s_new.symbol == s_past.symbol:
                        self._replace_statement(index_past, s_new)
                    else:
                        index_to_remove = self._get_index_to_remove(s_new, index_past)

                        if index_to_remove is None:
                            self._add_statement(index_new, s_new)
                        else:
                            self._remove_statements(index_new, index_to_remove)
                            index_past = index_to_remove + 1

                elif index_new == last_index_new:          # Remove rest of original
                    self._remove_statements(index_new + 1, last_index_past)

                index_past += 1

        if self._root_updated.get_last_node().rule not in ['WS_ALL', 'NEWLINE']:
            self._root_updated.add_newline_node()

        self.nodes = copy.deepcopy(self._nodes_updated)
        self._nodes_updated = []
        self.root = copy.deepcopy(self._root_updated)
        self._root_updated = None

        self._statements = statements_new
        self._statements_updated = True

    def _replace_statement(self, index_replace, statement):
        self._remove_statements(index_replace, index_replace)
        self._add_statement(index_replace, statement)

    def _remove_statements(self, index_remove_start, index_remove_end):
        for i in range(index_remove_start, index_remove_end+1):
            statement_to_remove = self.statements[i]
            node = self._get_node(statement_to_remove)
            self._nodes_updated.remove(node)
            self._root_updated.remove_node(node)

    def _add_statement(self, index_insert, statement):
        if isinstance(statement.expression, Piecewise):
            statement_str = self._translate_sympy_piecewise(statement)
        else:
            statement_str = f'\n{repr(statement).replace(":", "")}'
        node_tree = CodeRecordParser(statement_str).root
        node = node_tree.all('statement')[0]

        if isinstance(index_insert, int) and index_insert >= len(self._nodes_updated):
            index_insert = None

        if index_insert is None:
            self._nodes_updated.append(node)
            self._root_updated.add_node(node)
        else:
            node_following = self._nodes_updated[index_insert]
            self._nodes_updated.insert(index_insert, node)
            self._root_updated.add_node(node, node_following)

    def _translate_sympy_piecewise(self, statement):
        expression = statement.expression.args
        symbol = statement.symbol

        if len(expression) == 1:
            value = expression[0][0]
            condition = expression[0][1]
            condition_translated = self._translate_condition(condition)

            statement_str = f'\nIF ({condition_translated}) {symbol} = {value}\n'
            return statement_str
        else:
            return self._translate_sympy_block(symbol, expression)

    def _translate_sympy_block(self, symbol, expression_block):
        statement_str = '\nIF '
        for i, expression in enumerate(expression_block):
            value = expression[0]
            condition = expression[1]

            condition_translated = self._translate_condition(condition)

            if condition_translated == 'True':
                statement_str = re.sub('ELSE IF ', 'ELSE', statement_str)
            else:
                statement_str += f'({condition_translated}) THEN'

            statement_str += f'\n{symbol} = {value}\n'

            if i < len(expression_block) - 1:
                statement_str += 'ELSE IF '
            else:
                statement_str += 'END IF\n'

        return statement_str

    @staticmethod
    def _translate_condition(c):
        sign_dict = {'>': '.GT.',
                     '<': '.LT.',
                     '>=': '.GE.',
                     '<=': '.LE.'}
        if str(c).startswith('Eq'):
            c_split = re.split('[(,) ]', str(c))
            c_clean = [item for item in c_split if item != '' and item != 'Eq']
            c_transl = '.EQ.'.join([c_clean[0], c_clean[1]])
        else:
            c_split = str(c).split(' ')
            c_transl = ''.join([sign_dict.get(symbol, symbol)
                                for symbol in c_split])
        return c_transl

    def _get_node(self, statement):
        try:
            index_statement = self.statements.index(statement)
            return self.nodes[index_statement]
        except ValueError:
            return None

    def _get_index_to_remove(self, statement, index_start):
        try:
            index_statement = self._statements.index(statement, index_start)
            return index_statement - 1
        except ValueError:
            return None

    def _assign_statements(self):
        s = []
        for statement in self.root.all('statement'):
            node = statement.children[0]
            self.nodes.append(statement)
            if node.rule == 'assignment':
                name = str(node.variable).upper()
                expr = ExpressionInterpreter().visit(node.expression)
                ass = Assignment(name, expr)
                s.append(ass)
            elif node.rule == 'logical_if':
                logic_expr = ExpressionInterpreter().visit(node.logical_expression)
                try:
                    assignment = node.assignment
                except NoSuchRuleException:
                    pass
                else:
                    name = str(assignment.variable).upper()
                    expr = ExpressionInterpreter().visit(assignment.expression)
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
                for ifstat in first_block.all('statement'):
                    for assign_node in ifstat.all('assignment'):
                        name = str(assign_node.variable).upper()
                        first_symb_exprs.append((name, interpreter.visit(assign_node.expression)))
                        symbols.add(name)
                blocks.append((first_logic, first_symb_exprs))

                else_if_blocks = node.all('block_if_elseif')
                for elseif in else_if_blocks:
                    logic = interpreter.visit(elseif.logical_expression)
                    elseif_symb_exprs = []
                    for elseifstat in elseif.all('statement'):
                        for assign_node in elseifstat.all('assignment'):
                            name = str(assign_node.variable).upper()
                            elseif_symb_exprs.append((name,
                                                      interpreter.visit(assign_node.expression)))
                            symbols.add(name)
                    blocks.append((logic, elseif_symb_exprs))

                else_block = node.find('block_if_else')
                if else_block:
                    else_symb_exprs = []
                    for elsestat in else_block.all('statement'):
                        for assign_node in elsestat.all('assignment'):
                            name = str(assign_node.variable).upper()
                            else_symb_exprs.append((name,
                                                    interpreter.visit(assign_node.expression)))
                            symbols.add(name)
                    piecewise_logic = True
                    if len(blocks[0][1]) == 0 and not else_if_blocks:    # Special case for empty if
                        piecewise_logic = sympy.Not(blocks[0][0])
                    blocks.append((piecewise_logic, else_symb_exprs))

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

    def update(self, nonmem_names):
        statements_updated = copy.deepcopy(self.statements)
        for key, value in nonmem_names.items():
            statements_updated.subs({key: value})
        statements_updated.subs({'NaN': int(data.conf.na_rep)})
        self.statements = statements_updated

    def from_odes(self, ode_system):
        """Set statements of record given an eplicit ode system
        """
        odes = ode_system.odes[:-1]    # Skip last ode as it is for the output compartment
        functions = [ode.lhs.args[0] for ode in odes]
        function_map = {f: real(f'A({i + 1})') for i, f in enumerate(functions)}
        statements = []
        for i, ode in enumerate(odes):
            symbol = real(f'DADT({i + 1})')
            expression = ode.rhs.subs(function_map)
            statements.append(Assignment(symbol, expression))
        self.statements = statements
