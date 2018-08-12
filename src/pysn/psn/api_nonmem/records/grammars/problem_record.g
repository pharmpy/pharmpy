root : other* text? other*

text     : TEXT
other    : WS_ALL          -> whitespace
         | [WS] comment NL -> line

comment  : ";" [WS] [COMMENT] [WS]

TEXT     : /\S.*(?<!\s)/
COMMENT  : /\S[^;]*(?<!\s)/

WS       : (" "|/\t/)+
NL       : /\r?\n/
WS_ALL   : /\s+/
