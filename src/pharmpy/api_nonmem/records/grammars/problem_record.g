
root : [WS] text? (comment | ws)*

text : TEXT

// common rules
ws      : WS_ALL
comment : ";" [WS] [COMMENT]

// common misc terminals
WS: (" " | /\t/)+
WS_ALL: /\s+/

// common naked/enquoted text terminals
// TEXT: /[^"'][^,;()=\s]  // TODO: harmonize
TEXT: /\S.*(?<!\s)/
COMMENT : /\S.*(?<!\s)/  // no left/right whitespace padding
