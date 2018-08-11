// Legal forms:
//     1. init [FIXED]
//     2. ([low,] init [,up] [FIXED])
//     3. ([low,] init [,up]) [FIXED]
//     4. (low,,up) # have a hard time believing this one!
//     5. (value)xn
?root : unit*

?unit : _param [whitespace] comment -> parameter_def
      | _param                      -> parameter_def
      | [WS] comment NL             -> commentline
      | WS_MULTILINE                -> whitespace

_param : single | multiple
single : init [fix]
       | LP [lower_bound] sep init ([fix] RP | RP [fix])
       | LP [lower_bound] sep [init] sep [upper_bound] ([fix] RP | RP [fix])
multiple : LP init RP [whitespace] n_thetas

init        : NUMERIC [whitespace]
lower_bound : NUMERIC [whitespace]
upper_bound : NUMERIC [whitespace]
sep         : COMMA [whitespace]
fix         : ("FIX" | "FIXED") [whitespace]
n_thetas    : "x" [whitespace] INT [whitespace]

whitespace : WS (CONTINUATION NL [WS])?
comment    : ";" [whitespace] text NL
           | ";" [whitespace] NL
           | ";" NL
           | /;[^\r\n]*/
text       : NOT_NL

COMMA        : ","
LP           : "("
RP           : ")"
CONTINUATION : /&\n/

DIGIT: "0".."9"
INT: DIGIT+
DECIMAL: INT "." [INT] | "." INT
SIGNED_INT: ["+"|"-"] INT

EXP: "E" SIGNED_INT
FLOAT: INT EXP | DECIMAL [EXP]
SIGNED_FLOAT: ["+"|"-"] FLOAT

NUMBER: FLOAT | INT
SIGNED_NUMBER: ["+"|"-"] NUMBER
NUMERIC      : NUMBER | SIGNED_NUMBER

WS           : (" "|/\t/)+
NL           : /\r?\n/
NOT_NL       : /[^\r\n]/+
WS_MULTILINE : /[ \t\f\r\n]/+
