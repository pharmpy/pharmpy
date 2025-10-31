"""
Generic NONMEM code record class.

"""

from __future__ import annotations

import re
from functools import lru_cache
from operator import neg, pos, sub, truediv
from typing import Iterator, Literal, Sequence

from pharmpy.basic import BooleanExpr, Expr
from pharmpy.deps import sympy, sympy_printing
from pharmpy.internals.ds.ordered_set import OrderedSet
from pharmpy.internals.expr.funcs import (
    INT,
    LOG10,
    PDZ,
    PEXP,
    PHE,
    PHI,
    PLOG,
    PLOG10,
    PNG,
    PNP,
    PSQRT,
    PZR,
)
from pharmpy.internals.parse import AttrTree, NoSuchRuleException
from pharmpy.internals.parse.tree import Interpreter
from pharmpy.internals.sequence.lcs import diff
from pharmpy.model import Assignment, Statement, Statements

from .parsers import CodeRecordParser
from .record import Record


class NMTranPrinter(sympy_printing.str.StrPrinter):
    _operators = {
        'not': '.NOT. ',
    }

    def __init__(self, rvs=None, trans=None, **kwargs):
        self.rvs = rvs
        self.trans = trans
        super().__init__(**kwargs)

    def _print_Add(self, expr, order=None):
        if self.rvs is None:
            return super()._print_Add(expr)
        else:
            rvs_names = set(self.rvs.names)

            trans_rvs = (
                {str(v): str(k) for k, v in self.trans.items() if str(k) in rvs_names}
                if self.trans
                else None
            )

            expr_args = expr.args
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
                        variability_level = self.rvs[rv_name].level
                        if variability_level == 'RUV':
                            terms_ruv.append(arg)
                            continue
                    terms_iiv_iov.append(arg)
                else:
                    terms.append(arg)

            if not terms_iiv_iov and not terms_ruv:
                return super()._print_Add(expr)
            else:

                def arg_len(symb):
                    return len(symb.args)

                terms_iiv_iov.sort(reverse=True, key=arg_len)
                terms_ruv.sort(reverse=True, key=arg_len)
                terms += terms_iiv_iov + terms_ruv

        # Put numeric constant at the end
        for i, term in enumerate(terms):
            if term.is_Number:
                terms = terms[0:i] + terms[i + 1 :] + terms[i : i + 1]
                break

        return super()._print_Add(sympy.Add(*terms, evaluate=False), order='none')

    def _print_Float(self, expr):
        printed = str(super()._print_Float(expr))
        return printed.upper()

    def _print_Integer(self, expr):
        return str(expr)

    def _print_Function(self, expr):
        try:
            name = expr.name
        except AttributeError:
            name = expr.__class__.__name__.upper()
        if name == "LOGGAMMA":
            name = "GAMLN"
        return f'{name}({super().doprint(expr.args[0])})'

    def _print_Pow(self, expr, rational=False):
        if expr.exp == sympy.Rational(1, 2):
            return f"SQRT({self.doprint(expr.base)})"
        elif expr.exp == -1:
            if (
                isinstance(expr.base, sympy.Expr)
                and not isinstance(expr.base, sympy.Symbol)
                and len(
                    expr.base.make_args(expr.base)  # pyright: ignore [reportAttributeAccessIssue]
                )
                > 1
            ):
                return f"1/({expr.base})"
            else:
                return f"1/{expr.base}"
        else:
            return super()._print_Pow(expr, rational=rational)

    def _do_infix(self, expr, op):
        return super()._print(expr.args[0]) + op + super()._print(expr.args[1])

    def _print_Equality(self, expr):
        return self._do_infix(expr, ".EQ.")

    def _print_Unequality(self, expr):
        return self._do_infix(expr, ".NE.")

    def _print_Or(self, expr):
        return self._do_infix(expr, ".OR.")

    def _print_And(self, expr):
        return self._do_infix(expr, ".AND.")

    def _print_LessThan(self, expr):
        return self._do_infix(expr, ".LE.")

    def _print_StrictLessThan(self, expr):
        return self._do_infix(expr, ".LT.")

    def _print_StrictGreaterThan(self, expr):
        return self._do_infix(expr, ".GT.")

    def _print_GreaterThan(self, expr):
        return self._do_infix(expr, ".GE.")

    def _print_Not(self, expr):
        return f".NOT. ({self._print(expr.args[0])})"

    def _print_Symbol(self, expr):
        s = str(expr)
        if s != "NaN":
            s = s.upper()
        return s


def expression_to_nmtran(expr, rvs=None, trans=None):
    printer = NMTranPrinter(rvs, trans)
    expr_str = printer.doprint(sympy.sympify(expr))
    return expr_str


def nmtran_assignment_string(assignment: Assignment, defined_symbols: set[Expr], rvs, trans):
    expr = assignment.expression
    if expr.is_piecewise():
        s = _translate_sympy_piecewise(assignment, defined_symbols, rvs, trans)
    elif re.search('sign', str(expr)):  # FIXME: Don't use re here
        s = _translate_sympy_sign(assignment)
    elif expr.is_function() and expr.name == 'forward':
        s = _translate_forward(assignment, defined_symbols, rvs, trans)
    else:
        s = f'{str(assignment.symbol).upper()} = {expression_to_nmtran(expr, rvs, trans)}'
    return s


def _translate_forward(assignment, defined_symbols, rvs, trans):
    symbol = assignment.symbol
    expr = assignment.expression
    value = expr.args[0]
    cond = expr.args[1]
    piecewise_expr = Expr.piecewise((value, cond))
    piecewise_assignment = Assignment.create(symbol, piecewise_expr)
    s = _translate_sympy_piecewise(piecewise_assignment, defined_symbols, rvs, trans)
    return s


def _translate_sympy_piecewise(statement: Assignment, defined_symbols: set[Expr], rvs, trans):
    expression = statement.expression._sympy_().args
    symbol = statement.symbol
    # Did we (possibly) add the default in the piecewise with 0 or symbol?
    has_added_else = expression[-1][1] is sympy.true and (  # pyright: ignore [reportIndexIssue]
        expression[-1][0] == symbol  # pyright: ignore [reportIndexIssue]
        or (
            expression[-1][0] == 0  # pyright: ignore [reportIndexIssue]
            and symbol not in defined_symbols
        )
    )
    if has_added_else:
        expression = expression[0:-1]
    has_else = expression[-1][1] is sympy.true  # pyright: ignore [reportIndexIssue]

    expressions, _ = zip(*expression)

    if len(expression) == 1:
        value = expression[0][0]  # pyright: ignore [reportIndexIssue]
        condition = expression[0][1]  # pyright: ignore [reportIndexIssue]
        condition_translated = _translate_condition(condition)

        statement_str = (
            f'IF ({condition_translated}) {symbol} = {expression_to_nmtran(value, rvs, trans)}'
        )
        return statement_str
    elif all(len(e.args) == 0 for e in expressions) and not has_else:
        return _translate_sympy_single(symbol, expression, rvs, trans)
    else:
        return _translate_sympy_block(symbol, expression, rvs, trans)


def _translate_sympy_single(symbol, expression, rvs, trans):
    statement_str = ''
    for e in expression:
        value = e[0]
        condition = e[1]

        condition_translated = _translate_condition(condition)

        statement_str += (
            f'IF ({condition_translated}) {symbol} = {expression_to_nmtran(value, rvs, trans)}\n'
        )

    return statement_str


def _translate_sympy_block(symbol, expression, rvs, trans):
    statement_str = ''
    for i, e in enumerate(expression):
        value, condition = e

        condition_translated = _translate_condition(condition)

        if i == 0:
            statement_str += f'IF ({condition_translated}) THEN\n'
        elif condition_translated == 'True':
            statement_str += 'ELSE\n'
        else:
            statement_str += f'ELSE IF ({condition_translated}) THEN\n'

        statement_str += f'    {symbol} = {expression_to_nmtran(value, rvs, trans)}\n'
        if symbol.name == 'F_FLAG':
            statement_str += f'    MDVRES = {value}\n'

    statement_str += 'END IF'
    return statement_str


def _translate_condition(c):
    fprn = NMTranPrinter()
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
    fprn = NMTranPrinter()
    fortran = fprn.doprint(sympy.sympify(s.expression))
    expr_str = f'{s.symbol} = {fortran}'
    return expr_str


class ExpressionInterpreter(Interpreter):
    def instruction_unary(self, node):
        f, x = self.visit_subtrees(node)
        return f(x)

    def instruction_infix(self, node):
        a, op, b = self.visit_subtrees(node)
        return op(a, b)

    def real_expr(self, node):
        t = self.visit_subtrees(node)
        return t[0]

    def neg_op(self, _):
        return neg

    def pos_op(self, _):
        return pos

    def add_op(self, _):
        return sympy.Add

    def sub_op(self, _):
        return sub

    def mul_op(self, _):
        return sympy.Mul

    def div_op(self, _):
        return truediv

    def pow_op(self, _):
        return sympy.Pow

    def bool_expr(self, node):
        t = self.visit_subtrees(node)
        return t[0]

    def land(self, _):
        return sympy.And

    def lor(self, _):
        return sympy.Or

    def lnot(self, _):
        return sympy.Not

    def eq(self, _):
        return sympy.Eq

    def ne(self, _):
        return sympy.Ne

    def le(self, _):
        return sympy.Le

    def lt(self, _):
        return sympy.Lt

    def ge(self, _):
        return sympy.Ge

    def gt(self, _):
        return sympy.Gt

    def img(self, node):
        fn, *parameters = self.visit_subtrees(node)
        return fn(*parameters)

    def exp(self, _):
        return sympy.exp

    def pexp(self, _):
        return PEXP

    def log(self, _):
        return sympy.log

    def plog(self, _):
        return PLOG

    def log10(self, _):
        return LOG10

    def plog10(self, _):
        return PLOG10

    def sqrt(self, _):
        return sympy.sqrt

    def psqrt(self, _):
        return PSQRT

    def sin(self, _):
        return sympy.sin

    def cos(self, _):
        return sympy.cos

    def tan(self, _):
        return sympy.tan

    def asin(self, _):
        return sympy.asin

    def acos(self, _):
        return sympy.acos

    def atan(self, _):
        return sympy.atan

    def abs(self, _):
        return sympy.Abs

    def int(self, _):
        return INT

    def loggamma(self, _):
        return sympy.loggamma

    def pdz(self, _):
        return PDZ

    def pzr(self, _):
        return PZR

    def pnp(self, _):
        return PNP

    def phe(self, _):
        return PHE

    def png(self, _):
        return PNG

    def phi(self, _):
        return PHI

    def mod(self, _):
        return sympy.Mod

    def number(self, node):
        s = str(node)
        try:
            return sympy.Integer(s)
        except ValueError:
            s = s.replace('d', 'E')  # Fortran special format
            s = s.replace('D', 'E')
            return sympy.Float(s)

    def symbol(self, node):
        t = self.visit_subtrees(node)
        return sympy.Symbol(t[0])

    def assignable(self, node):
        t = self.visit_subtrees(node)
        return sympy.Symbol(t[0])

    def parameter(self, node):
        name, *subscripts = self.visit_subtrees(node)
        return f'{name}({",".join(subscripts)})'

    def vector(self, node):
        name, *subscripts = self.visit_subtrees(node)
        return f'{name}({",".join(subscripts)})'

    def name(self, node):
        return str(node).upper()

    def index(self, node):
        return str(node)

    def array(self, node):
        name = str(node).upper()
        return 'EPS' if name == 'ERR' else name

    def matrix(self, node):
        return str(node).upper()


def _index_statements_diff(
    last_node_index: int,
    index: Sequence[tuple[int, int, int, int]],
    it: Iterator[tuple[Literal[-1, 0, 1], Statement]],
):
    """This function reorders and groups a diff of statements according to the
    given index mapping"""

    index_index = 0

    while True:
        try:
            op, s = next(it)
        except StopIteration:
            break

        # NOTE: We forward standalone insertions
        if op == 1:
            yield op, [s], last_node_index, last_node_index
            continue

        # NOTE: We fetch the index entry for this group of statements
        assert index_index < len(index)
        ni, nj, si, sj = index[index_index]
        index_index += 1

        operations = [op]
        statements = [s]
        expected = sj - si - 1

        # NOTE: We retrieve all statements for this index entry, as well as
        # interleaved insertion statements
        while expected > 0:
            op, s = next(it)  # NOTE: We do not expect this to raise

            operations.append(op)
            statements.append(s)
            if op != 1:
                expected -= 1

        if all(op == 0 for op in operations):
            # NOTE: If this group of statements contains no changes, we can keep
            # the associated nodes.
            yield 0, statements, ni, nj
        else:
            # NOTE: Otherwise we remove all associated nodes
            yield -1, [s for s, op in zip(statements, operations) if op != 1], ni, nj
            # NOTE: And generate new nodes for kept statements
            new_statements = [s for s, op in zip(statements, operations) if op != -1]
            if new_statements:
                yield 1, new_statements, nj, nj

        last_node_index = nj


class CodeRecord(Record):
    def __init__(self, name, raw_name, content, index=None, statements=None):
        # NOTE: self._index establishes a correspondance between self.root
        # nodes and self.statements statements. self._index consists of
        # (ni, nj, si, sj) tuples which maps the nodes
        # self.root.children[ni:nj] to the statements self.statements[si:sj]
        if index is None:
            self._index = []
        else:
            self._index = index
        if statements is None:
            index, statements = _parse_tree(content)
            self._index = index
            self._statements = statements
        else:
            self._statements = statements
        super().__init__(name, raw_name, content)

    @property
    def statements(self):
        return self._statements

    def update_statements(self, new: Sequence[Statement], rvs=None, trans=None):
        try:
            old = self._statements
        except AttributeError:
            old = self.statements
        if new == old:
            return self
        new_children = []
        last_node_index = 0
        new_index = []
        defined_symbols = set()  # NOTE: Set of all defined symbols so far
        si = 0  # NOTE: We keep track of progress in the "new" statements sequence

        first_statement_index = (
            self._index[0][0]  # NOTE: Start insertion just before the first statement
            if self._index
            else (
                first_verbatim  # NOTE: Start insertion just before the first verbatim
                if (first_verbatim := self.root.first_index('verbatim')) != -1
                else len(self.root.children)  # NOTE: Start insertion after all blanks
            )
        )
        for op, statements, ni, nj in _index_statements_diff(
            first_statement_index, self._index, diff(old, new)
        ):
            # NOTE: We copy interleaved non-statement nodes
            new_children.extend(self.root.children[last_node_index:ni])
            if op == 1:
                for s in statements:
                    assert isinstance(s, Assignment)
                    statement_nodes = self._statement_to_nodes(defined_symbols, s, rvs, trans)
                    # NOTE: We insert the generated nodes just before the next
                    # existing statement node
                    insert_pos = len(new_children)
                    new_index.append((insert_pos, insert_pos + len(statement_nodes), si, si + 1))
                    si += 1
                    new_children.extend(statement_nodes)
                    defined_symbols.add(s.symbol)
            elif op == 0:
                # NOTE: We keep the nodes but insert them at an updated position
                insert_pos = len(new_children)
                insert_len = nj - ni
                new_index.append((insert_pos, insert_pos + insert_len, si, si + len(statements)))
                si += len(statements)
                new_children.extend(self.root.children[ni:nj])
                for s in statements:
                    if isinstance(s, Assignment):
                        defined_symbols.add(s.symbol)
            last_node_index = nj
        # NOTE: We copy any non-statement nodes that are remaining
        new_children.extend(self.root.children[last_node_index:])
        new_root = AttrTree(self.root.rule, tuple(new_children))
        return CodeRecord(self.name, self.raw_name, new_root, index=new_index, statements=new)

    def _statement_to_nodes(self, defined_symbols: set, s: Assignment, rvs, trans):
        statement_str = nmtran_assignment_string(s, defined_symbols, rvs, trans) + '\n'
        node_tree = CodeRecordParser(statement_str).root
        assert node_tree is not None
        statement_nodes = list(node_tree.subtrees('statement'))
        return statement_nodes

    def from_odes(self, ode_system, extra):
        """Set statements of record given an explicit ode system
        extra statements are added to the top
        """
        odes = ode_system.eqs
        functions = [ode.lhs.args[0] for ode in odes]
        function_map = {f: sympy.Symbol(f'A({i + 1})') for i, f in enumerate(functions)}
        statements = [s.subs(function_map) for s in extra]
        for i, ode in enumerate(odes):
            # For now Piecewise signals zero-order infusions, which are handled with parameters
            ode = BooleanExpr(ode._sympy_().replace(sympy.Piecewise, lambda a1, a2: 0))
            symbol = Expr.symbol(f'DADT({i + 1})')
            expression = ode.rhs.subs(function_map)
            statements.append(Assignment(symbol, expression))
        return self.update_statements(statements)

    def update_extra_nodes(self, dvs, dvid_name):
        """Update AST nodes not part of the statements

        Currently the block IF for DVID is handled as extra nodes
        """
        # Want to find block_if node not in index
        if len(dvs) < 2:
            return self
        curn = 0
        found = None
        for ni, nj, _, _ in self._index:
            for n in range(curn, ni):
                node = self.root.children[n]
                statement_nodes = list(node.subtrees('statement'))
                for s_node in statement_nodes:
                    if s_node.find('block_if') or s_node.find('logical_if'):
                        found = n
                        break
            curn = nj
        if curn < len(self.root.children):
            for n in range(curn, len(self.root.children)):
                node = self.root.children[n]
                statement_nodes = list(node.subtrees('statement'))
                for s_node in statement_nodes:
                    if s_node.find('block_if') or s_node.find('logical_if'):
                        found = n
                        break

        if found is None:
            node = create_dvs_node(dvs, dvid_name)
            new_root = AttrTree(self.root.rule, self.root.children + (node,))
            rec = CodeRecord(
                self.name, self.raw_name, new_root, index=self._index, statements=self._statements
            )
            return rec
        else:
            return self


def create_dvs_node(dvs, dvid_name):
    """Create special dvs AST node"""
    s = ""
    for i, (dv, dvid) in enumerate(dvs.items()):
        s += f'IF ({dvid_name}.EQ.{dvid}) Y = {dv}\n'
    node = CodeRecordParser(s).root
    return node


@lru_cache(4096)
def _parse_tree(tree: AttrTree):
    s = []
    new_index = []
    interpreter = ExpressionInterpreter()
    for child_index, child in enumerate(tree.children):
        if not isinstance(child, AttrTree) or child.rule != 'statement':
            continue
        for node in child.children:
            # NOTE: Why does this iterate over the children?
            # Right now it looks like it could add the same statement
            # multiple times because there is no break on a match.
            if not isinstance(node, AttrTree):
                continue
            if node.rule == 'assignment':
                symbol = interpreter.visit(node.subtree('assignable'))
                expr = interpreter.visit(node.subtree('real_expr'))
                ass = Assignment.create(symbol, expr)
                s.append(ass)
                new_index.append((child_index, child_index + 1, len(s) - 1, len(s)))
            elif node.rule == 'logical_if':
                logic_expr = interpreter.visit(node.subtree('bool_expr'))
                try:
                    assignment = node.subtree('assignment')
                except NoSuchRuleException:
                    pass
                else:
                    symbol = interpreter.visit(assignment.subtree('assignable'))
                    expr = interpreter.visit(assignment.subtree('real_expr'))
                    # Check if symbol was previously declared
                    else_val = symbol if any(map(lambda x: x.symbol == symbol, s)) else None
                    if else_val is not None:
                        pw = sympy.Piecewise((expr, logic_expr), (else_val, True))
                    else:
                        if (
                            logic_expr == sympy.Ne(sympy.Symbol("NEWIND"), sympy.Integer(2))
                            and expr.is_Symbol
                        ):
                            pw = Expr.first(expr, "ID")
                        elif logic_expr == sympy.Gt(sympy.Symbol('AMT'), sympy.Integer(0)) and (
                            str(expr) == "AMT" or str(expr) == "TIME"
                        ):
                            pw = Expr.forward(expr, logic_expr)
                        else:
                            pw = sympy.Piecewise((expr, logic_expr))
                    ass = Assignment.create(symbol, pw)
                    s.append(ass)
                    new_index.append((child_index, child_index + 1, len(s) - 1, len(s)))
            elif node.rule == 'block_if':
                blocks = []  # [(logic, [(symb1, expr1), ...]), ...]
                symbols = OrderedSet()

                first_block = node.subtree('block_if_start')
                first_logic = interpreter.visit(first_block.subtree('bool_expr'))
                first_symb_exprs = []
                for ifstat in first_block.subtrees('statement'):
                    for assign_node in ifstat.subtrees('assignment'):
                        symbol = interpreter.visit(assign_node.subtree('assignable'))
                        first_symb_exprs.append(
                            (symbol, interpreter.visit(assign_node.subtree('real_expr')))
                        )
                        symbols.add(symbol)
                blocks.append((first_logic, first_symb_exprs))

                for elseif in node.subtrees('block_if_elseif'):
                    logic = interpreter.visit(elseif.subtree('bool_expr'))
                    elseif_symb_exprs = []
                    for elseifstat in elseif.subtrees('statement'):
                        for assign_node in elseifstat.subtrees('assignment'):
                            symbol = interpreter.visit(assign_node.subtree('assignable'))
                            elseif_symb_exprs.append(
                                (symbol, interpreter.visit(assign_node.subtree('real_expr')))
                            )
                            symbols.add(symbol)
                    blocks.append((logic, elseif_symb_exprs))

                else_block = node.find('block_if_else')
                if else_block:
                    assert isinstance(else_block, AttrTree)
                    else_symb_exprs = []
                    for elsestat in else_block.subtrees('statement'):
                        for assign_node in elsestat.subtrees('assignment'):
                            symbol = interpreter.visit(assign_node.subtree('assignable'))
                            else_symb_exprs.append(
                                (symbol, interpreter.visit(assign_node.subtree('real_expr')))
                            )
                            symbols.add(symbol)
                    piecewise_logic = True
                    if len(blocks) == 1 and len(blocks[0][1]) == 0:
                        # Special case for empty if
                        piecewise_logic = sympy.Not(blocks[0][0])
                    blocks.append((piecewise_logic, else_symb_exprs))
                s_block = []
                for symbol in symbols:
                    pairs = []
                    for block in blocks:
                        logic = block[0]
                        for cursymb, expr in block[1]:
                            if cursymb == symbol:
                                pairs.append((expr, logic))

                    if pairs[-1][1] is not True:
                        else_val = symbol if any(map(lambda x: x.symbol == symbol, s)) else None
                        if else_val is not None:
                            pairs.append((else_val, True))

                    pw = sympy.Piecewise(*pairs)
                    ass = Assignment.create(symbol, pw)
                    s_block.append(ass)

                s.extend(_reorder_block_statements(s_block))
                curind = (child_index, child_index + 1, len(s) - len(symbols), len(s))
                new_index.append(curind)

    statements = Statements(s).subs({Expr.symbol('NEWIND'): Expr.newind()})

    return new_index, statements


def _reorder_block_statements(s):
    piecewise = [ass for ass in s if isinstance(ass.expression, sympy.Piecewise)]
    assignments = [ass for ass in s if ass not in piecewise]
    return assignments + piecewise
