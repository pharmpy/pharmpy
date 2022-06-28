# The modeling features language
# A high level language to describe model features and ranges of model features

# whitespace between tokens are ignored
# ; and \n separate features
# feature names are case insensitive
# FEATURE_NAME(options)     options is a comma separated list
#
# absorption(x)      - x can be FO, ZO, SEQ-ZO-FO, [FO, ZO] for multiple or * for all
# elimination(x)     - x can be FO, ZO, MM and MIX-FO-MM
# peripherals(n)     - n can be 0, 1, ... or a..b for a range (inclusive in both ends)
# transits(n[, d])   - n as above, d (optional) is DEPOT, NODEPOT, or *
# covariate(p, c, e) - p is a parameter name, a list of parameter names, or a reference
#                    - c is a covariate name, a list of covariate names, or a reference
#                    - e is an effect, a list of effects, or * for all continuous effects
#
# LET(variable_name, values) defines a new variable to be a value or list of values


grammar = r"""
%ignore " "
_SEPARATOR: /;|\n/
start: _statement (_SEPARATOR _statement)*
_statement: definition | _feature

definition: "LET"i "(" VARIABLE_NAME "," values ")"

_feature: absorption | elimination | peripherals | transits | lagtime | covariate

absorption: "ABSORPTION"i "(" (_absorption_option) ")"
elimination: "ELIMINATION"i "(" (_elimination_option) ")"
peripherals: "PERIPHERALS"i "(" (_counts) ")"
transits: "TRANSITS"i "(" _counts ["," depot_option] ")"
lagtime: "LAGTIME"i "(" ")"
covariate: "COVARIATE"i "(" parameter_option "," covariate_option "," fp_option ["," op_option] ")"

_absorption_option: absorption_modes | absorption_wildcard
absorption_modes: ABSORPTION_MODE | "[" [ABSORPTION_MODE ("," ABSORPTION_MODE)*] "]"
absorption_wildcard: WILDCARD
ABSORPTION_MODE: "FO"i | "ZO"i | "SEQ-ZO-FO"i
_elimination_option: elimination_modes | elimination_wildcard
elimination_modes: ELIMINATION_MODE | "[" [ELIMINATION_MODE ("," ELIMINATION_MODE)*] "]"
elimination_wildcard: WILDCARD
ELIMINATION_MODE: "FO"i | "ZO"i | "MM"i | "MIX-FO-MM"i
depot_option: DEPOT | NODEPOT | WILDCARD
DEPOT: "DEPOT"i
NODEPOT: "NODEPOT"i
parameter_option: values | ref | parameter_wildcard
covariate_option: values | ref | covariate_wildcard
fp_option: values | fp_wildcard
!op_option: "+" | "*"
ref: "@" VARIABLE_NAME
parameter_wildcard: WILDCARD
covariate_wildcard: WILDCARD
fp_wildcard: WILDCARD

WILDCARD: "*"

VARIABLE_NAME: /[a-zA-Z]+/

values: value | _value_array
_value_array: "[" [value ("," value)*] "]"
value: /[a-zA-Z0-9-]+/

_counts: count | count_array
count_array: "[" [number ("," number)*] "]"
count: number | range
range: number ".." number
number: /\d+/
"""
