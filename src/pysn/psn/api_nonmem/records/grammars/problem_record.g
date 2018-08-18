
root : [WS] text? (comment | ws)*

text    : TEXT

// common rules
ws      : WS_ALL
comment : ";" [WS] [TEXT] [WS]

// common terminals
TEXT: /\S.*(?<!\s)/
WS: (" "|/\t/)+
WS_ALL: /\s+/
