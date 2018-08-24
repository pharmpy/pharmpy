// A grammar for the filter options in NONMEM, i.e. IGNORE and ACCEPT in $DATA
root: [ws] path (pair | ws | comment)*

path: SINGLE_QUOTED_STRING -> single_quoted_path
    | DOUBLE_QUOTED_STRING -> double_quoted_path
    | STRING -> non_quoted_path


pair: KEY [WS] EQUALS [WS] VALUE
      | VALUE

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
SINGLE_QUOTE: "'"
SINGLE_QUOTED_STRING: /'[^']*'/
DOUBLE_QUOTED_STRING: /"[^']*"/
STRING: /[^\s]+/
