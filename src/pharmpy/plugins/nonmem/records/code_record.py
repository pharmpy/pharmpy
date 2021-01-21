"""
Generic NONMEM code record class.

"""

import re

import lark
import sympy
from sympy import Piecewise

import pharmpy.symbols as symbols
from pharmpy.data_structures import OrderedSet
from pharmpy.parse_utils.generic import AttrToken, NoSuchRuleException
from pharmpy.plugins.nonmem.records.parsers import CodeRecordParser
from pharmpy.statements import Assignment, ModelStatements

from .record import Record


class ExpressionInterpreter(lark.visitors.Interpreter):
    def visit_children(self, tree):
        """Does not visit tokens"""
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
                else:  # op == '/':
                    expr /= term
        else:
            expr = unary_factor * t[0]
        return expr

    def logical_expression(self, node):
        t = self.visit_children(node)
        if len(t) > 2:
            ops = t[1::2]
            terms = t[2::2]
            expr = t[0]
            for op, term in zip(ops, terms):
                expr = op(expr, term)
            return expr
        else:
            op, expr = self.visit_children(node)
            return op(expr)

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
        else:  # name == '.NOT.':
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
        else:  # name == "PHI":
            return lambda x: (1 + sympy.erf(x) / sympy.sqrt(2)) / 2

    def power(self, node):
        b, e = self.visit_children(node)
        return b ** e

    @staticmethod
    def operator(node):
        return str(node)

    @staticmethod
    def number(node):
        s = str(node)
        try:
            return sympy.Integer(s)
        except ValueError:
            s = s.replace('d', 'E')  # Fortran special format
            s = s.replace('D', 'E')
            return sympy.Float(s)

    @staticmethod
    def symbol(node):
        name = str(node).upper()
        if name.startswith('ERR('):
            name = 'EPS' + name[3:]
        symb = symbols.symbol(name)
        return symb


def lcslen(a, b):
    # generate matrix of length of longest common subsequence for sublists of both lists
    lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    return lengths


def lcsdiff(c, x, y, i, j):
    """Print the diff using LCS length matrix using backtracking"""
    if i < 0 and j < 0:
        return
    elif i < 0:
        yield from lcsdiff(c, x, y, i, j - 1)
        yield '+', y[j]
    elif j < 0:
        yield from lcsdiff(c, x, y, i - 1, j)
        yield '-', x[i]
    elif x[i] == y[j]:
        yield from lcsdiff(c, x, y, i - 1, j - 1)
        yield None, x[i]
    elif c[i][j - 1] >= c[i - 1][j]:
        yield from lcsdiff(c, x, y, i, j - 1)
        yield '+', y[j]
    else:
        yield from lcsdiff(c, x, y, i - 1, j)
        yield '-', x[i]


def diff(old, new):
    """Get diff between a and b in order for all elements

    Optimizes by first handling equal elements from the head and tail
    Each entry is a pair of operation (+, - or None) and the element
    """
    for i, (a, b) in enumerate(zip(old, new)):
        if a == b:
            yield (None, b)
        else:
            break
    else:
        if len(old) == 0 or len(new) == 0:
            i = 0
        else:
            i += 1

    rold = old[i:]
    rnew = new[i:]

    saved = []
    for a, b in zip(reversed(rold), reversed(rnew)):
        if a == b:
            saved.append((None, b))
        else:
            break

    rold = rold[: len(rold) - len(saved)]
    rnew = rnew[: len(rnew) - len(saved)]

    c = lcslen(rold, rnew)
    for op, val in lcsdiff(c, rold, rnew, len(rold) - 1, len(rnew) - 1):
        yield op, val

    for pair in reversed(saved):
        yield pair


class CodeRecord(Record):
    def __init__(self, content, parser_class):
        self.is_updated = False
        self.rvs, self.trans = None, None
        super().__init__(content, parser_class)

    @property
    def statements(self):
        statements = self._assign_statements()
        self._statements = statements
        return statements.copy()

    @statements.setter
    def statements(self, new):
        try:
            old = self._statements
        except AttributeError:
            old = self.statements
        if new == old:
            return
        old_index = 0
        node_index = 0
        kept = []
        new_nodes = []
        for op, s in diff(old, new):
            while (
                node_index < len(self.root.children)
                and old_index < len(self.nodes)
                and self.root.children[node_index] is not self.nodes[old_index]
            ):
                node = self.root.children[node_index]
                kept.append(node)
                node_index += 1
            if op == '+':
                if isinstance(s.expression, Piecewise):
                    statement_str = self._translate_sympy_piecewise(s)
                elif re.search('sign', str(s.expression)):
                    statement_str = self._translate_sympy_sign(s)
                else:
                    statement_str = s.print_custom(self.rvs, self.trans)
                node_tree = CodeRecordParser(statement_str).root
                for node in node_tree.all('statement'):
                    if node_index == 0:
                        node.children.insert(0, AttrToken('LF', '\n'))
                    if (
                        not node.all('LF')
                        and node_index != 0
                        or len(self.root.children) > 0
                        and self.root.children[0].rule != 'empty_line'
                    ):
                        node.children.append(AttrToken('LF', '\n'))
                    new_nodes.append(node)
                    kept.append(node)
            elif op == '-':
                node_index += 1
                old_index += 1
            else:
                kept.append(self.root.children[node_index])
                new_nodes.append(self.root.children[node_index])
                node_index += 1
                old_index += 1
        if node_index < len(self.root.children):  # Remaining non-statements
            kept.extend(self.root.children[node_index:])
        self.root.children = kept
        self.nodes = new_nodes
        self._statements = new.copy()

    def _translate_sympy_piecewise(self, statement):
        expression = statement.expression.args
        symbol = statement.symbol

        expressions, _ = zip(*expression)

        if len(expression) == 1:
            value = expression[0][0]
            condition = expression[0][1]
            condition_translated = self._translate_condition(condition)

            statement_str = f'IF ({condition_translated}) {symbol} = {value}'
            return statement_str
        elif all(len(e.args) == 0 for e in expressions):
            return self._translate_sympy_single(symbol, expression)
        else:
            return self._translate_sympy_block(symbol, expression)

    def _translate_sympy_single(self, symbol, expression):
        statement_str = ''
        for e in expression:
            value = e[0]
            condition = e[1]

            condition_translated = self._translate_condition(condition)

            statement_str += f'IF ({condition_translated}) {symbol} = {value}\n'

        return statement_str

    def _translate_sympy_block(self, symbol, expression):
        statement_str = 'IF '
        for i, e in enumerate(expression):
            value = e[0]
            condition = e[1]

            condition_translated = self._translate_condition(condition)

            if condition_translated == 'True':
                statement_str = re.sub('ELSE IF ', 'ELSE', statement_str)
            else:
                statement_str += f'({condition_translated}) THEN'

            statement_str += f'\n{symbol} = {value}\n'

            if i < len(expression) - 1:
                statement_str += 'ELSE IF '
            else:
                statement_str += 'END IF'

        return statement_str

    @staticmethod
    def _translate_condition(c):
        sign_dict = {'>': '.GT.', '<': '.LT.', '>=': '.GE.', '<=': '.LE.'}
        if str(c).startswith('Eq'):
            c_split = re.split('[(,) ]', str(c))
            c_clean = [item for item in c_split if item != '' and item != 'Eq']
            c_transl = '.EQ.'.join([c_clean[0], c_clean[1]])
        else:
            c_split = str(c).split(' ')
            c_transl = ''.join([sign_dict.get(symbol, symbol) for symbol in c_split])
        return c_transl

    @staticmethod
    def _translate_sympy_sign(s):
        args = s.expression.args

        subs_dict = dict()
        for arg in args:
            if str(arg).startswith('sign'):
                sign_arg = arg.args[0]
                subs_dict[arg] = abs(sign_arg) / sign_arg

        s.subs(subs_dict)

        return f'\n{repr(s).replace(":", "")}'

    def _assign_statements(self):
        s = []
        self.nodes = []
        for statement in self.root.all('statement'):
            for node in statement.children:
                if node.rule == 'assignment':
                    name = str(node.variable).upper()
                    expr = ExpressionInterpreter().visit(node.expression)
                    ass = Assignment(name, expr)
                    s.append(ass)
                    self.nodes.append(statement)
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
                    self.nodes.append(statement)
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
                            first_symb_exprs.append(
                                (name, interpreter.visit(assign_node.expression))
                            )
                            symbols.add(name)
                    blocks.append((first_logic, first_symb_exprs))

                    else_if_blocks = node.all('block_if_elseif')
                    for elseif in else_if_blocks:
                        logic = interpreter.visit(elseif.logical_expression)
                        elseif_symb_exprs = []
                        for elseifstat in elseif.all('statement'):
                            for assign_node in elseifstat.all('assignment'):
                                name = str(assign_node.variable).upper()
                                elseif_symb_exprs.append(
                                    (name, interpreter.visit(assign_node.expression))
                                )
                                symbols.add(name)
                        blocks.append((logic, elseif_symb_exprs))

                    else_block = node.find('block_if_else')
                    if else_block:
                        else_symb_exprs = []
                        for elsestat in else_block.all('statement'):
                            for assign_node in elsestat.all('assignment'):
                                name = str(assign_node.variable).upper()
                                else_symb_exprs.append(
                                    (name, interpreter.visit(assign_node.expression))
                                )
                                symbols.add(name)
                        piecewise_logic = True
                        if len(blocks[0][1]) == 0 and not else_if_blocks:
                            # Special case for empty if
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
                    self.nodes.append(statement)

        statements = ModelStatements(s)
        return statements

    def from_odes(self, ode_system):
        """Set statements of record given an eplicit ode system"""
        odes = ode_system.odes[:-1]  # Skip last ode as it is for the output compartment
        functions = [ode.lhs.args[0] for ode in odes]
        function_map = {f: symbols.symbol(f'A({i + 1})') for i, f in enumerate(functions)}
        statements = []
        for i, ode in enumerate(odes):
            # For now Piecewise signals zero-order infusions, which are handled with parameters
            ode = ode.replace(sympy.Piecewise, lambda a1, a2: 0)
            symbol = symbols.symbol(f'DADT({i + 1})')
            expression = ode.rhs.subs(function_map)
            statements.append(Assignment(symbol, expression))
        self.statements = statements

    def __str__(self):
        if self.is_updated:
            return self.raw_name + str(self.root).upper()
        return super(CodeRecord, self).__str__()
