from lark import Lark, InlineTransformer, Tree

from lark_pretty import prettyTree


'''Lexer for THETA records

Legal init forms:
    1. init [FIXED]
    2. ([low,] init [,up] [FIXED])
    3. ([low,] init [,up]) [FIXED]
    4. (low,,up) # have a hard time believing this one!
    5. (value)xn
'''

grammar = r'''
?root : unit*

?unit : param [whitespace] comment -> init
      | param                      -> init
      | [WS] comment NL            -> commentline
      | WS_MULTILINE               -> whitespace

?param : theta | thetas
theta  : init_value [fix]
       | LP [lower_bound] sep init_value ([fix] RP | RP [fix])
       | LP [lower_bound] sep [init_value] sep [upper_bound] ([fix] RP | RP [fix])
thetas : LP init_value RP [whitespace] n_thetas

init_value   : NUMERIC [whitespace]
lower_bound  : NUMERIC [whitespace]
upper_bound  : NUMERIC [whitespace]
sep          : [WS] COMMA [whitespace]
fix          : ("FIX" | "FIXED") [whitespace]
n_thetas : "x" [whitespace] INT [whitespace]

whitespace  : WS (CONTINUATION NL [WS])?
comment     : ";" [whitespace] text NL
            | ";" [whitespace] NL
            | ";" NL
            | /;[^\r\n]*/
text        : NOT_NL

COMMA        : ","
LP           : "("
RP           : ")"
CONTINUATION : /&\n/

DIGIT: "0".."9"
INT: DIGIT+
DECIMAL: INT "." [INT] | "." INT
SIGNED_INT: ["+"|"-"] INT

_EXP: "E" SIGNED_INT
FLOAT: INT _EXP | DECIMAL [_EXP]
SIGNED_FLOAT: ["+"|"-"] FLOAT

NUMBER: FLOAT | INT
SIGNED_NUMBER: ["+"|"-"] NUMBER
NUMERIC      : (NUMBER | SIGNED_NUMBER)

WS           : (" "|/\t/)+
NL           : /\r?\n/
NOT_NL       : /[^\r\n]/+
WS_MULTILINE : /[ \t\f\r\n]/+
'''
parser = Lark(grammar, start='root')

data = '''
  (0,0.00469307) ; CL
  (0,1.00916) ; V
  (-.99,.1)
'''
data = '''

  (0, 0.00469307) ; CL

  12.3 ; V
  ; comment
    ; comment
  (,.1)
  (3) x 5

'''

tree = parser.parse(data)
print(prettyTree(tree))
print(prettyTree(tree, verbose=False))
