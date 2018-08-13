root : [WS] text? (comment | WS_ALL)*

text     : TEXT
comment  : ";" [WS] [TEXT] [WS]

TEXT     : /\S.*(?<!\s)/

WS       : (" "|/\t/)+
NL       : /\r?\n/
WS_ALL   : /\s+/
