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
# transits(N)        - a literal "N"
# covariate(p, c, e) - p is a parameter name, a list of parameter names, or a reference
#                    - c is a covariate name, a list of covariate names, or a reference
#                    - e is an effect, a list of effects, or * for all continuous effects
# allometry(c[, ref]) - c is a covariate name and ref is an optional reference value
#
# LET(variable_name, values) defines a new variable to be a value or list of values


grammar = r"""
%ignore " "
_SEPARATOR: /;|\n/
start: _statement (_SEPARATOR _statement)*
_statement: definition | _feature

definition: "LET"i "(" VARIABLE_NAME "," values ")"

_feature: absorption | elimination | peripherals | transits | lagtime
            | covariate | allometry | direct_effect | effect_comp | indirect_effect
            | metabolite | iiv | iov | covariance

absorption: "ABSORPTION"i "(" (_values_or_wildcard) ")"
elimination: "ELIMINATION"i "(" (_values_or_wildcard) ")"
peripherals: "PERIPHERALS"i "(" (_counts) ["," _values_or_wildcard] ")"
transits: "TRANSITS"i "(" (n | (_counts ["," _values_or_wildcard])) ")"
lagtime: "LAGTIME"i "(" (_values_or_wildcard) ")"
covariate: "COVARIATE"i [optional] "(" _values_or_ref_or_wildcard "," _values_or_ref_or_wildcard "," (_values_or_wildcard) ["," op_option] ")"  # noqa: E501
allometry: "ALLOMETRY"i "(" value ["," decimal] ")"
iiv: "IIV"i [optional] "(" _values_or_ref_or_wildcard "," (_values_or_wildcard) ")"
iov: "IOV"i [optional] "(" _values_or_ref_or_wildcard "," (_values_or_wildcard) ")"
covariance: "COVARIANCE"i [optional] "(" _values_or_wildcard "," (_values_or_ref_or_wildcard) ")"

direct_effect: "DIRECTEFFECT"i "(" (_values_or_wildcard) ")"
effect_comp: "EFFECTCOMP"i "(" (_values_or_wildcard) ")"
indirect_effect: "INDIRECTEFFECT"i "(" _values_or_wildcard "," _values_or_wildcard ")"

metabolite: "METABOLITE"i "(" (_values_or_wildcard) ")"

_values_or_wildcard: values | wildcard
_values_or_ref_or_wildcard: values | ref | wildcard

!op_option: "+" | "*"

ref: "@" VARIABLE_NAME
optional: OPTIONAL
wildcard: WILDCARD
WILDCARD: "*"
OPTIONAL: "?"

VARIABLE_NAME: /[a-zA-Z_]+/

values: value | _value_array
_value_array: "[" [value ("," value)*] "]"
value: /[a-zA-Z0-9-]+/

n: "n"i
_counts: count | count_array
count_array: "[" [number ("," number)*] "]"
count: number | range
range: number ".." number
number: /\d+/
decimal: /\d+(\.\d+)?/
"""
