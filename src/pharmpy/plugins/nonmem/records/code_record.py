"""
Generic NONMEM code record class.

"""

import re
from typing import Iterator, Literal, Sequence, Tuple

import lark

from pharmpy.deps import sympy, sympy_printing
from pharmpy.internals.ds.ordered_set import OrderedSet
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.parse import AttrToken, NoSuchRuleException
from pharmpy.internals.sequence.lcs import diff
from pharmpy.model import Assignment, RandomVariables, Statement, Statements
from pharmpy.plugins.nonmem.records.parsers import CodeRecordParser

from .record import Record


class MyPrinter(sympy_printing.str.StrPrinter):
    def _print_Add(self, expr):
        args = expr.args
        new = []
        for arg in args:
            new.append(self._print(arg))
        return super()._print_Add(sympy.Add(*args, evaluate=False), order='none')


class NMTranPrinter(sympy_printing.fortran.FCodePrinter):
    # Differences from FCodePrinter in sympy
    # 1. Upper case
    # 2. Use Fortran 77 names for relationals
    # 3. Use default kind for reals (which will be translated to double kind by NMTRAN)
    # All these could be submitted as options to the sympy printer
    _relationals = {
        '<=': '.LE.',
        '>=': '.GE.',
        '<': '.LT.',
        '>': '.GT.',
        '!=': '.NE.',
        '==': '.EQ.',
    }
    _operators = {
        'and': '.AND.',
        'or': '.OR.',
        'xor': '.NEQV.',
        'equivalent': '.EQV.',
        'not': '.NOT. ',
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._settings["standard"] = 95

    def _print_Float(self, expr):
        printed = sympy_printing.codeprinter.CodePrinter._print_Float(self, expr)
        return printed


def nmtran_assignment_string(assignment, defined_symbols, rvs, trans):
    if isinstance(assignment.expression, sympy.Piecewise):
        statement_str = _translate_sympy_piecewise(assignment, defined_symbols)
    elif re.search('sign', str(assignment.expression)):  # FIXME: Don't use re here
        statement_str = _translate_sympy_sign(assignment)
    else:
        statement_str = _print_custom(assignment, rvs, trans)
    return statement_str


def _translate_sympy_piecewise(statement, defined_symbols):
    expression = statement.expression.args
    symbol = statement.symbol
    # Did we (possibly) add the default in the piecewise with 0 or symbol?
    has_added_else = expression[-1][1] is sympy.true and (
        expression[-1][0] == symbol or (expression[-1][0] == 0 and symbol not in defined_symbols)
    )
    if has_added_else:
        expression = expression[0:-1]
    has_else = expression[-1][1] is sympy.true

    expressions, _ = zip(*expression)

    if len(expression) == 1:
        value = expression[0][0]
        condition = expression[0][1]
        condition_translated = _translate_condition(condition)

        statement_str = f'IF ({condition_translated}) {symbol} = {value}'
        return statement_str
    elif all(len(e.args) == 0 for e in expressions) and not has_else:
        return _translate_sympy_single(symbol, expression)
    else:
        return _translate_sympy_block(symbol, expression)


def _translate_sympy_single(symbol, expression):
    statement_str = ''
    for e in expression:
        value = e[0]
        condition = e[1]

        condition_translated = _translate_condition(condition)

        statement_str += f'IF ({condition_translated}) {symbol} = {value}\n'

    return statement_str


def _translate_sympy_block(symbol, expression):
    statement_str = ''
    for i, e in enumerate(expression):
        value, condition = e

        condition_translated = _translate_condition(condition)

        if i == 0:
            statement_str += f'IF ({condition_translated}) THEN\n'
        elif condition_translated == '.true.':
            statement_str += 'ELSE\n'
        else:
            statement_str += f'ELSE IF ({condition_translated}) THEN\n'

        statement_str += f'    {symbol} = {value}\n'

    statement_str += 'END IF'
    return statement_str


def _translate_condition(c):
    fprn = NMTranPrinter(settings={'source_format': 'free'})
    fortran = fprn.doprint(c).replace(' ', '')
    return fortran


def _translate_sympy_sign(s):
    args = s.expression.args

    subs_dict = {}
    for arg in args:
        if str(arg).startswith('sign'):
            sign_arg = arg.args[0]
            subs_dict[arg] = abs(sign_arg) / sign_arg

    s = s.subs(subs_dict)
    fprn = NMTranPrinter(settings={'source_format': 'free'})
    fortran = fprn.doprint(s.expression)
    expr_str = f'{s.symbol} = {fortran}'
    return expr_str


def _print_custom(assignment, rvs, trans):
    expr_ordered = _order_terms(assignment, rvs, trans)
    return f'{assignment.symbol} = {expr_ordered}'


def _order_terms(assignment: Assignment, rvs: RandomVariables, trans):
    """Order terms such that random variables are placed last. Currently only supports
    additions."""
    if not isinstance(assignment.expression, sympy.Add) or rvs is None:
        return assignment.expression

    rvs_names = set(rvs.names)

    trans_rvs = {str(v): str(k) for k, v in trans.items() if str(k) in rvs_names} if trans else None

    expr_args = assignment.expression.args
    terms_iiv_iov, terms_ruv, terms = [], [], []

    for arg in expr_args:
        arg_symbs = [s.name for s in arg.free_symbols]
        rvs_intersect = rvs_names.intersection(arg_symbs)

        if trans_rvs is not None:
            trans_intersect = set(trans_rvs.keys()).intersection(arg_symbs)
            rvs_intersect.update({trans_rvs[rv] for rv in trans_intersect})

        if rvs_intersect:
            if len(rvs_intersect) == 1:
                rv_name = list(rvs_intersect)[0]
                variability_level = rvs[rv_name].level
                if variability_level == 'RUV':
                    terms_ruv.append(arg)
                    continue
            terms_iiv_iov.append(arg)
        else:
            terms.append(arg)

    if not terms_iiv_iov and not terms_ruv:
        return assignment.expression

    def arg_len(symb):
        return len(symb.args)

    terms_iiv_iov.sort(reverse=True, key=arg_len)
    terms_ruv.sort(reverse=True, key=arg_len)
    terms += terms_iiv_iov + terms_ruv

    new_order = sympy.Add(*terms, evaluate=False)

    return MyPrinter().doprint(new_order)


class ExpressionInterpreter(lark.visitors.Interpreter):
    def visit_children(self, tree):
        """Does not visit tokens"""
        return [
            self.visit(child)
            for child in tree.children
            if isinstance(child, lark.Tree)  # pyright: ignore [reportPrivateImportUsage]
        ]

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
        smallz = 2.8e-103
        name = str(node).upper()
        if name == "EXP" or name == "DEXP":
            return sympy.exp
        if name == "PEXP":
            return lambda x: sympy.Piecewise((sympy.exp(100), x > 100), (sympy.exp(x), True))
        elif name == "LOG":
            return sympy.log
        elif name == "PLOG":
            return lambda x: sympy.Piecewise((sympy.log(smallz), x < smallz), (sympy.log(x), True))
        elif name == "LOG10":
            return lambda x: sympy.log(x, 10)
        elif name == "PLOG10":
            return lambda x: sympy.Piecewise(
                (sympy.log(smallz, 10), x < smallz), (sympy.log(x, 10), True)
            )
        elif name == "SQRT":
            return sympy.sqrt
        elif name == "PSQRT":
            return lambda x: sympy.Piecewise((0, x < 0), (sympy.sqrt(x), True))
        elif name == "SIN":
            return sympy.sin
        elif name == "COS":
            return sympy.cos
        elif name == "ABS":
            return sympy.Abs
        elif name == "TAN" or name == "PTAN":
            return sympy.tan
        elif name == "ASIN" or name == "PASIN":
            return sympy.asin
        elif name == "ACOS" or name == "PACOS":
            return sympy.acos
        elif name == "ATAN" or name == "PATAN":
            return sympy.atan
        elif name == "INT":
            return lambda x: sympy.sign(x) * sympy.floor(sympy.Abs(x))
        elif name == "GAMLN":
            return sympy.loggamma
        elif name == "PDZ":
            return lambda x: sympy.Piecewise((1 / smallz, abs(x) < smallz), (1 / x, True))
        elif name == "PZR":
            return lambda x: sympy.Piecewise((smallz, abs(x) < smallz), (x, True))
        elif name == "PNP":
            return lambda x: sympy.Piecewise((smallz, x < smallz), (x, True))
        elif name == "PHE":
            return lambda x: sympy.Piecewise((100, x > 100), (x, True))
        elif name == "PNG":
            return lambda x: sympy.Piecewise((0, x < 0), (x, True))
        else:  # name == "PHI":
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
            s = s.replace('d', 'E')  # Fortran special format
            s = s.replace('D', 'E')
            return sympy.Float(s)

    @staticmethod
    def symbol(node):
        name = str(node).upper()
        if name.startswith('ERR('):
            name = 'EPS' + name[3:]
        symb = sympy.Symbol(name)
        return symb


def _index_statements_diff(
    index: Sequence[Tuple[int, int, int, int]], it: Iterator[Tuple[Literal[-1, 0, 1], Statement]]
):
    """This function reorders and groups a diff of statements according to the
    given index mapping"""

    index_index = 0
    last_node_index = index[0][0] if index else 0

    while True:

        try:
            op, s = next(it)
        except StopIteration:
            break

        # NOTE We forward standalone insertions
        if op == 1:
            yield op, [s], last_node_index, last_node_index
            continue

        # NOTE We fetch the index entry for this group of statements
        assert index_index < len(index)
        ni, nj, si, sj = index[index_index]
        index_index += 1

        operations = [op]
        statements = [s]
        expected = sj - si - 1

        # NOTE We retrieve all statements for this index entry, as well as
        # interleaved insertion statements
        while expected > 0:

            op, s = next(it)  # NOTE We do not expect this to raise

            operations.append(op)
            statements.append(s)
            if op != 1:
                expected -= 1

        if all(op == 0 for op in operations):
            # NOTE If this group of statements contains no changes, we can keep
            # the associated nodes.
            yield 0, statements, ni, nj
        else:
            # NOTE Otherwise we remove all associated nodes
            yield -1, [s for s, op in zip(statements, operations) if op != 1], ni, nj
            # NOTE And generate new nodes for kept statements
            new_statements = [s for s, op in zip(statements, operations) if op != -1]
            if new_statements:
                yield 1, new_statements, nj, nj

        last_node_index = nj


class CodeRecord(Record):
    def __init__(self, content, parser_class):
        self.is_updated = False
        self.rvs, self.trans = None, None
        # NOTE self._index establishes a correspondance between self.root
        # nodes and self.statements statements. self._index consists of
        # (ni, nj, si, sj) tuples which maps the nodes
        # self.root.children[ni:nj] to the statements self.statements[si:sj]
        self._index = []
        super().__init__(content, parser_class)

    @property
    def statements(self):
        statements = self._assign_statements()
        self._statements = statements
        return statements

    @statements.setter
    def statements(self, new: Sequence[Statement]):
        try:
            old = self._statements
        except AttributeError:
            old = self.statements
        if new == old:
            return
        new_children = []
        last_node_index = 0
        new_index = []
        defined_symbols = set()  # NOTE Set of all defined symbols so far
        si = 0  # NOTE We keep track of progress in the "new" statements sequence
        for op, statements, ni, nj in _index_statements_diff(self._index, diff(old, new)):
            # NOTE We copy interleaved non-statement nodes
            new_children.extend(self.root.children[last_node_index:ni])
            if op == 1:
                for s in statements:
                    statement_nodes = self._statement_to_nodes(defined_symbols, ni, s)
                    # NOTE We insert the generated nodes just before the next
                    # existing statement node
                    insert_pos = len(new_children)
                    new_index.append((insert_pos, insert_pos + len(statement_nodes), si, si + 1))
                    si += 1
                    new_children.extend(statement_nodes)
                    if isinstance(s, Assignment):
                        defined_symbols.add(s.symbol)
            elif op == 0:
                # NOTE We keep the nodes but insert them at an updated position
                insert_pos = len(new_children)
                insert_len = nj - ni
                new_index.append((insert_pos, insert_pos + insert_len, si, si + len(statements)))
                si += len(statements)
                new_children.extend(self.root.children[ni:nj])
                for s in statements:
                    if isinstance(s, Assignment):
                        defined_symbols.add(s.symbol)
            last_node_index = nj
        # NOTE We copy any non-statement nodes that are remaining
        new_children.extend(self.root.children[last_node_index:])
        self.root.children = new_children
        self._index = new_index
        self._statements = new

    def _statement_to_nodes(self, defined_symbols, node_index, s):
        statement_str = nmtran_assignment_string(s, defined_symbols, self.rvs, self.trans)
        node_tree = CodeRecordParser(statement_str).root
        assert node_tree is not None
        statement_nodes = []
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
            statement_nodes.append(node)
        return statement_nodes

    def _assign_statements(self):
        s = []
        new_index = []
        for child_index, child in enumerate(self.root.children):
            if child.rule != 'statement':
                continue
            for node in child.children:
                # NOTE why does this iterate over the children?
                # Right now it looks like it could add the same statement
                # multiple times because there is no break on a match.
                if node.rule == 'assignment':
                    name = str(node.variable).upper()
                    expr = ExpressionInterpreter().visit(node.expression)
                    ass = Assignment(sympy.Symbol(name), expr)
                    s.append(ass)
                    new_index.append((child_index, child_index + 1, len(s) - 1, len(s)))
                elif node.rule == 'logical_if':
                    logic_expr = ExpressionInterpreter().visit(node.logical_expression)
                    try:
                        assignment = node.assignment
                    except NoSuchRuleException:
                        pass
                    else:
                        name = str(assignment.variable).upper()
                        expr = ExpressionInterpreter().visit(assignment.expression)
                        # Check if symbol was previously declared
                        else_val = (
                            sympy.Symbol(name)
                            if any(map(lambda x: x.symbol.name == name, s))
                            else sympy.Integer(0)
                        )
                        pw = sympy.Piecewise((expr, logic_expr), (else_val, True))
                        ass = Assignment(sympy.Symbol(name), pw)
                        s.append(ass)
                        new_index.append((child_index, child_index + 1, len(s) - 1, len(s)))
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

                    for name in symbols:
                        pairs = []
                        for block in blocks:
                            logic = block[0]
                            for cursymb, expr in block[1]:
                                if cursymb == name:
                                    pairs.append((expr, logic))

                        if pairs[-1][1] is not True:
                            else_val = (
                                sympy.Symbol(name)
                                if any(map(lambda x: x.symbol.name == name, s))
                                else sympy.Integer(0)
                            )
                            pairs.append((else_val, True))

                        pw = sympy.Piecewise(*pairs)
                        ass = Assignment(sympy.Symbol(name), pw)
                        s.append(ass)
                    new_index.append((child_index, child_index + 1, len(s) - len(symbols), len(s)))

        self._index = new_index
        statements = Statements(s)
        return statements

    def from_odes(self, ode_system):
        """Set statements of record given an explicit ode system"""
        odes = ode_system.odes[:-1]  # Skip last ode as it is for the output compartment
        functions = [ode.lhs.args[0] for ode in odes]
        function_map = {f: sympy.Symbol(f'A({i + 1})') for i, f in enumerate(functions)}
        statements = []
        for i, ode in enumerate(odes):
            # For now Piecewise signals zero-order infusions, which are handled with parameters
            ode = ode.replace(sympy.Piecewise, lambda a1, a2: 0)
            symbol = sympy.Symbol(f'DADT({i + 1})')
            expression = subs(ode.rhs, function_map, simultaneous=True)
            statements.append(Assignment(symbol, expression))
        self.statements = statements

    def __str__(self):
        if self.is_updated:
            s = str(self.root)
            newlines = []
            # FIXME: Workaround for upper casing all code but not comments.
            # should properly be handled in a custom printer
            for line in s.split('\n'):
                parts = line.split(';', 1)
                modline = parts[0].upper()
                if len(parts) == 2:
                    modline += ';' + parts[1]
                newlines.append(modline)
            assert self.raw_name is not None
            return self.raw_name + '\n'.join(newlines)
        return super(CodeRecord, self).__str__()
