root: (option | ws | comment)*

option: KEY [WS] EQUALS [WS] VALUE
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
