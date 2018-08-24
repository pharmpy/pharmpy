// A grammar for the filter options in NONMEM, i.e. IGNORE and ACCEPT in $DATA
root: ws path (ignore_accept | pair | ws | comment)*

path: SINGLE_QUOTED_STRING -> single_quoted_path
    | DOUBLE_QUOTED_STRING -> double_quoted_path
    | NON_QUOTED_STRING -> non_quoted_path

ignore_accept.2: ignore_accept_identifier [ws] [EQUALS] [ws] ignore_accept_list
ignore_accept_identifier: IGNORE | ACCEPT
ignore_accept_list: OPENPAREN (COMMA | filter_expression)+  CLOSEPAREN 
filter_expression: filter_column filter_operator filter_value 
filter_operator: OP_STRING_EQUALS | OP_STRING_NOT_EQUALS | OP_GREATER_THAN
    | OP_GREATER_THAN_OR_EQUAL | OP_LESS_THAN | OP_LESS_THAN_OR_EQUAL | OP_NOT_EQUALS | OP_EQUALS
filter_column: COLUMN_NAME
filter_value: STRING_UNTIL_COMMA_OR_CLOSEPAREN

pair: KEY [WS] EQUALS [WS] VALUE
      | VALUE

OP_STRING_EQUALS: /\.EQ\.|=|==/
OP_STRING_NOT_EQUALS: /\.NE\.|\/=/ 
OP_GREATER_THAN: /\.GT\.|>/
OP_GREATER_THAN_OR_EQUAL: /\.GE\.|>=/
OP_LESS_THAN: /\.LT\.|</
OP_LESS_THAN_OR_EQUAL: /\.LE\.|<=/
OP_NOT_EQUALS: /\.NEN\./
OP_EQUALS: /\.EQN\./
IGNORE: /IGNORE|IGNOR|IGNO|IGN/
ACCEPT: /ACC|ACCE|ACCEP|ACCEPT/
STRING_UNTIL_COMMA_OR_CLOSEPAREN: /[^,)]+/
VALUE: /[^\s=]+/
KEY: /\w+/

// common rules
ws      : WS_ALL
comment : ";" [WS] [TEXT]

// common terminals
TEXT: /\S.*(?<!\s)/
WS: (" " | /\t/)+
WS_ALL: /\s+/
EQUALS: "="
OPENPAREN: "("
CLOSEPAREN: ")"
COMMA: ","
SINGLE_QUOTE: "'"
SINGLE_QUOTED_STRING: /'[^']*'/
DOUBLE_QUOTED_STRING: /["][^"]*["]/
NON_QUOTED_STRING: /[^"'][^\s]*/
STRING: /[^\s]+/
COLUMN_NAME: /\w+/
