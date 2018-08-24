// A grammar for the filter options in NONMEM, i.e. IGNORE and ACCEPT in $DATA
root: (pair | ws | comment)*

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
