"""Simple Lexer class

Inspiration from:
    * http://pygments.org/docs/lexerdevelopment/
    * https://gist.github.com/eliben/5797351/
"""
from collections import namedtuple
from textwrap import dedent
import re
import sys

from .api_generic import ModelParsingError


TokenRule = namedtuple('TokenRule', 'regex type action')


class Token(object):
    def __init__(self, stack: list, content: str, pos: int):
        assert len(stack) > 0
        self.stack = stack
        self.content = content
        self.pos = pos

    @property
    def type(self):
        return self.__class__.__name__ + '.' + self.stack[-1]

    @property
    def state(self):
        if len(self.stack) > 2:
            return self.stack[-1]

    @property
    def path(self):
        return '.'.join([self.__class__.__name__] + self.stack)

    def __str__(self):
        return '%d %s %s' % (self.pos, self.type, repr(self.content))

    def __repr__(self):
        clsname = self.__class__.__name__
        clsattr = []
        for name in ['stack', 'content', 'pos']:
            attr = getattr(self, name)
            if isinstance(attr, int):
                attr = str(attr)
            else:
                attr = "'%s'" % (attr,)
            clsattr += ['%s=%s' % (name, attr)]
        return '%s(%s)' % (clsname, ', '.join(clsattr))


def TokenType(name):
    return type(name, (Token,), {})


class LexerError(ModelParsingError):
    def __init__(self, pos, msg='lexing error'):
        super().__init__('%s at position %d' % (msg, pos))
        self.pos = pos


class Lexer(object):
    def __init__(self, buf):
        if not getattr(self, 'root', None):
            cls = self.__class__.__name__
            idx = cls.rfind('Lexer')
            if idx:
                self.root = cls[0:idx] + 'Token' + cls[(idx+5):]
            else:
                self.root = cls + 'Token'
        self.Token = TokenType(self.root)
        if getattr(self, 'skip_whitespace', None):
            self.re_ws_skip = re.compile('\S')
        else:
            self.re_ws_skip = None
        self.debug = getattr(self, 'debug', None)

        self._token_rules = self._init_rules()
        self._state_stack = []

        self.pos = 0
        self.buf = buf

    def token(self):
        if self.pos >= len(self.buf):
            return None
        if self.re_ws_skip:
            m = self.re_ws_skip.search(self.buf, self.pos)
            if m:
                self.pos = m.start()
            else:
                return None

        self._debug()
        for rule in self._token_rules[self.state]:
            m = rule.regex.match(self.buf, self.pos)
            if m:
                stack = self.states + [rule.type]
                tok = self.Token(stack, m.group(), self.pos)
                self._debug('generated token: %s' % (tok,))
                if rule.action:
                    if rule.action == '#push':
                        self._state_stack.append(self.state)
                    elif rule.action == '#pop':
                        self._state_stack.pop()
                    else:
                        self._state_stack.append(rule.action)
                self.pos = m.end()
                return tok
        raise LexerError(self.pos, 'no rule match')

    @property
    def tokens(self):
        self.pos = 0
        while 1:
            tok = self.token()
            if tok is None: break
            yield tok

    @property
    def state(self):
        if len(self._state_stack) > 0:
            return self._state_stack[-1]
        else:
            return 'root'

    @property
    def states(self):
        return self._state_stack.copy()

    def _init_rules(self):
        try:
            rules = self.rules
        except AttributeError:
            self._invalid("self.rules missing or empty")
        try:
            rules['root']
        except KeyError:
            self._invalid("self.rules missing key 'root'")

        tokens = dict()
        for state, ruleset in rules.items():
            try:
                rules_state = list(ruleset)
            except TypeError:
                self._invalid("self.rules['%s'] not (iterable) rules" % state)
            actions = set(rules.keys()) - {state, 'root'}
            if state != 'root':
                if state.startswith('#'):
                    self._invalid("bad state name: self.rules['%s']" % state)
                actions.update(['#push', '#pop'])
            valid = []
            for rule in rules_state:
                try:
                    N = len(rule)
                except TypeError:
                    self._invalid_rule(state, i, 'not iterable')
                if N == 2:
                    pat, type = rule
                    action = None
                elif N == 3:
                    pat, type, action = rule
                    if action not in actions:
                        self._invalid_rule(state, i,
                            "action '%s' not legal: " % (action, actions))
                else:
                    self._invalid_rule(state, i, 'len not 2 or 3')
                try:
                    regex = re.compile(pat)
                except re.error:
                    self._invalid_rule(state, i, 'err re.compile: %s' % (pat,))
                valid += [TokenRule(regex, type, action)]
            tokens[state] = valid
        return tokens

    def _invalid_rule(self, name, i, msg):
        path = "self.rules['%s']" % (name,)
        self._invalid("invalid (%s) rule %s[%d]" % (msg, path, i))

    def _invalid(self, msg):
        clsname = self.__class__.__name__
        supname = self.super().__class__.__name__
        pre = "%s setup error (%s inherit is invalid)" % (supname, clsname)
        raise ValueError("%s: %s" % (pre, msg))

    def _debug(self, msg=None):
        if self.debug and msg:
            print(msg)
        elif self.debug:
            maxlen = 40
            beg, cur, end = 0, self.pos, len(self.buf)
            cutlen = max(0, end - beg - maxlen)
            if cutlen > 0:
                beg_offs = max(0, cur - beg - int(maxlen/2))
                end_offs = cutlen - beg_offs
                line = self.buf[(beg + beg_offs):(end - end_offs)]
                cur = cur - beg_offs
            else:
                line = self.buf
            cur_adj = cur
            lines = line.splitlines()
            lines_out = []
            for i, line in enumerate(lines):
                ll = len(line)
                if cur_adj in range(ll):
                    marker = '~'*cur_adj + '^'
                    marker += '~'*(ll - len(marker))
                else:
                    marker = '~'*ll
                pre = '%d: ' % (i,)
                lines_out += [pre + line, ' '*len(pre) + marker]
                cur_adj -= (1 + ll)
            out = '\n'.join(lines_out)
            print(out)

    def __repr__(self):
        pos_old = self.pos
        self.pos = 0
        tokens = [repr(tok) for tok in self.tokens]
        self.pos = pos_old
        return str(tokens)

    def __str__(self):
        maxlen = 20
        pos_old = self.pos
        self.pos = 0
        rows = []
        wcol = None
        for tok in self.tokens:
            content = repr(tok.content)
            if len(content) > maxlen:
                cut = len(content) - maxlen
                content = "%s..[%d]" % (content[0:maxlen], cut)
            row = ('%d' % (tok.pos,), '%s' % (tok.type,), content)
            width = tuple(len(x) for x in row)
            if not wcol:
                wcol = width
            else:
                wcol = tuple(max(a, b) for a, b in zip(wcol, width))
            rows += [row]
        self.pos = pos_old
        fmt = ' '.join(['%-{}s'.format(w) for w in wcol])
        return '\n'.join([fmt % tuple(r) for r in rows])

def test():
    class TestLexer(Lexer):
        name = 'Non-state example Lexer'
        rules = {
            'root': [
                ('\s+',             'WHITESPACE'),
                ('\d+',             'NUMBER'),
                ('[a-zA-Z_]\w+',    'IDENTIFIER'),
                ('\+',              'PLUS'),
                ('\-',              'MINUS'),
                ('\*',              'MULTIPLY'),
                ('\/',              'DIVIDE'),
                ('\(',              'LP'),
                ('\)',              'RP'),
                ('=',               'EQUALS'),
            ]
        }

    lex = TestLexer('erw = _abc + 12*(R4-623902)  ')
    print(str(lex))

    class TestLexer(Lexer):
        name = 'State example Lexer (.cpp comments)'
        rules = {
            'root': [
                (r'[^/]+',  'TEXT'),
                (r'/\*',    'COMMENT.MULTILINE', 'comment'),
                (r'//.*?$', 'COMMENT.SINGLELINE'),
                (r'/',      'TEXT')
            ],
            'comment': [
                (r'[^*/]+', 'COMMENT.MULTILINE'),
                (r'/\*',    'COMMENT.MULTILINE', '#push'),
                (r'\*/',    'COMMENT.MULTILINE', '#pop'),
                (r'[*/]',   'COMMENT.MULTILINE')
            ]
        }

    buf = dedent('''
        10/2
        a = 2 // comment
        // comment out next line\
        std::cout << std::endl;

        // b = 3
        /* comment */
        /*
           multiline comment
            /*
               nested multiline comment
            */
        */
    '''.strip())

    lex = TestLexer(buf)
    print(str(lex))
