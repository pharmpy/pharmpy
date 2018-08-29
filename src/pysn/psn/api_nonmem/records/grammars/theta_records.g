// legal forms:
//   * init [FIXED]
//   * ([low,] init [,up] [FIXED])
//   * ([low,] init [,up]) [FIXED]
//   * (low,,up)
//   * (value)xn

root : [ws] (param | ws | comment)*

param : (single | multi)

single : init [fix]
       | LP [WS] [low] sep init (fix RP [WS] | RP [WS] [fix])
       | LP [WS] [low] sep [init] sep [up] (fix RP [WS] | RP [WS] [fix])
multi  : LP [WS] init RP [WS] n_thetas

init        : NUMERIC [WS]
low : NUMERIC [WS]
up : NUMERIC [WS]
sep         : COMMA [WS]
fix         : ("FIX" | "FIXED") [WS]
n_thetas    : "x" [WS] INT [WS]

COMMA: ","
LP: "("
RP: ")"

// common rules
ws      : WS_ALL
comment : ";" [WS] [TEXT]

// common terminals
TEXT: /\S.*(?<!\s)/
WS: (" " | /\t/)+
WS_ALL: /\s+/

DIGIT: "0".."9"
INT: DIGIT+
DECIMAL: (INT "." [INT] | "." INT)
SIGNED_INT: ["+" | "-"] INT

EXP: "E" SIGNED_INT
FLOAT: (INT EXP | DECIMAL [EXP])
SIGNED_FLOAT: ["+" | "-"] FLOAT

NUMBER: (FLOAT | INT)
SIGNED_NUMBER: ["+" | "-"] NUMBER
NUMERIC: (NUMBER | SIGNED_NUMBER)
