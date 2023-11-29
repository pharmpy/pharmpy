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

_feature: absorption | elimination | peripherals | transits | lagtime
            | covariate | direct_effect | effect_comp | indirect_effect
            | metabolite

absorption: "ABSORPTION"i "(" (_absorption_option) ")"
elimination: "ELIMINATION"i "(" (_elimination_option) ")"
peripherals: "PERIPHERALS"i "(" (_counts) ["," _peripheral_comp] ")"
transits: "TRANSITS"i "(" _counts ["," _depot_option] ")"
lagtime: "LAGTIME"i "(" (_lagtime_option) ")"
covariate: "COVARIATE"i [optional_cov] "(" parameter_option "," covariate_option "," fp_option ["," op_option] ")"

direct_effect: "DIRECTEFFECT"i "(" (_pdtype_option) ")"
effect_comp: "EFFECTCOMP"i "(" (_pdtype_option) ")"
indirect_effect: "INDIRECTEFFECT"i "(" _pdtype_option "," _production_option ")"

metabolite: "METABOLITE"i "(" (_metabolite_option) ")"

_pdtype_option: pdtype_modes | pdtype_wildcard
pdtype_modes: PDTYPE_MODE | "[" [PDTYPE_MODE ("," PDTYPE_MODE)*] "]"
PDTYPE_MODE: "linear"i | "Emax"i | "sigmoid"i
pdtype_wildcard: WILDCARD

_production_option: production_modes | production_wildcard
production_modes: PRODUCTION_MODE
PRODUCTION_MODE: "production"i | "degradation"i
production_wildcard: WILDCARD

_metabolite_option: metabolite_modes | metabolite_wildcard
metabolite_modes: METABOLITE_MODE | "[" [METABOLITE_MODE ("," METABOLITE_MODE)*] "]"
METABOLITE_MODE: "basic"i | "psc"i
metabolite_wildcard: WILDCARD

_absorption_option: absorption_modes | absorption_wildcard
absorption_modes: ABSORPTION_MODE | "[" [ABSORPTION_MODE ("," ABSORPTION_MODE)*] "]"
absorption_wildcard: WILDCARD
ABSORPTION_MODE: "FO"i | "ZO"i | "SEQ-ZO-FO"i | "INST"i

_elimination_option: elimination_modes | elimination_wildcard
elimination_modes: ELIMINATION_MODE | "[" [ELIMINATION_MODE ("," ELIMINATION_MODE)*] "]"
elimination_wildcard: WILDCARD
ELIMINATION_MODE: "FO"i | "ZO"i | "MM"i | "MIX-FO-MM"i

_depot_option: depot_modes | depot_wildcard
depot_modes: DEPOT_MODE | "[" [DEPOT_MODE ("," DEPOT_MODE)*] "]"
DEPOT_MODE: "DEPOT"i |"NODEPOT"i
depot_wildcard: WILDCARD

_peripheral_comp: peripheral_modes | peripheral_wildcard
peripheral_modes: PERIPHERAL_MODE | "[" [PERIPHERAL_MODE ("," PERIPHERAL_MODE)*] "]"
PERIPHERAL_MODE: "DRUG"i | "MET"i
peripheral_wildcard: WILDCARD

_lagtime_option: lagtime_modes | lagtime_wildcard
lagtime_modes: LAGTIME_MODE | "[" [LAGTIME_MODE ("," LAGTIME_MODE)*] "]"
lagtime_wildcard: WILDCARD
LAGTIME_MODE: "ON"i | "OFF"i

parameter_option: values | ref | parameter_wildcard
covariate_option: values | ref | covariate_wildcard
fp_option: values | fp_wildcard
!op_option: "+" | "*"
optional_cov: OPTIONAL

ref: "@" VARIABLE_NAME
parameter_wildcard: WILDCARD
covariate_wildcard: WILDCARD
fp_wildcard: WILDCARD

WILDCARD: "*"
OPTIONAL: "?"

VARIABLE_NAME: /[a-zA-Z_]+/

values: value | _value_array
_value_array: "[" [value ("," value)*] "]"
value: /[a-zA-Z0-9-]+/

_counts: count | count_array
count_array: "[" [number ("," number)*] "]"
count: number | range
range: number ".." number
number: /\d+/
"""
