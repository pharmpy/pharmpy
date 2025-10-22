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
            | metabolite | iiv | iov

absorption: "ABSORPTION"i "(" (_absorption_option) ")"
elimination: "ELIMINATION"i "(" (_elimination_option) ")"
peripherals: "PERIPHERALS"i "(" (_counts) ["," _peripheral_comp] ")"
transits: "TRANSITS"i "(" (n | (_counts ["," _depot_option])) ")"
lagtime: "LAGTIME"i "(" (_lagtime_option) ")"
covariate: "COVARIATE"i [optional] "(" parameter_option "," covariate_option "," (_fp_options) ["," op_option] ")"
allometry: "ALLOMETRY"i "(" value ["," decimal] ")"
iiv: "IIV"i [optional] "(" parameter_option "," (_fp_var_options) ")"
iov: "IOV"i [optional] "(" parameter_option "," (_fp_var_options) ")"

direct_effect: "DIRECTEFFECT"i "(" (_pdtype_option) ")"
effect_comp: "EFFECTCOMP"i "(" (_pdtype_option) ")"
indirect_effect: "INDIRECTEFFECT"i "(" _pdtype_option "," _production_option ")"

metabolite: "METABOLITE"i "(" (_metabolite_option) ")"

_pdtype_option: pdtype_modes | wildcard
pdtype_modes: PDTYPE_MODE | "[" [PDTYPE_MODE ("," PDTYPE_MODE)*] "]"
PDTYPE_MODE: "linear"i | "Emax"i | "sigmoid"i | "step"i | "loglin"i

_production_option: production_modes | wildcard
production_modes: PRODUCTION_MODE | "[" [PRODUCTION_MODE ("," PRODUCTION_MODE)*] "]"
PRODUCTION_MODE: "production"i | "degradation"i

_metabolite_option: metabolite_modes | wildcard
metabolite_modes: METABOLITE_MODE | "[" [METABOLITE_MODE ("," METABOLITE_MODE)*] "]"
METABOLITE_MODE: "basic"i | "psc"i

_absorption_option: absorption_modes | wildcard
absorption_modes: ABSORPTION_MODE | "[" [ABSORPTION_MODE ("," ABSORPTION_MODE)*] "]"
ABSORPTION_MODE: "FO"i | "ZO"i | "SEQ-ZO-FO"i | "INST"i | "WEIBULL"i

_elimination_option: elimination_modes | wildcard
elimination_modes: ELIMINATION_MODE | "[" [ELIMINATION_MODE ("," ELIMINATION_MODE)*] "]"
ELIMINATION_MODE: "FO"i | "ZO"i | "MM"i | "MIX-FO-MM"i

_depot_option: depot_modes | wildcard
depot_modes: DEPOT_MODE | "[" [DEPOT_MODE ("," DEPOT_MODE)*] "]"
DEPOT_MODE: "DEPOT"i |"NODEPOT"i

_peripheral_comp: peripheral_modes | wildcard
peripheral_modes: PERIPHERAL_MODE | "[" [PERIPHERAL_MODE ("," PERIPHERAL_MODE)*] "]"
PERIPHERAL_MODE: "DRUG"i | "MET"i

_lagtime_option: lagtime_modes | wildcard
lagtime_modes: LAGTIME_MODE | "[" [LAGTIME_MODE ("," LAGTIME_MODE)*] "]"
LAGTIME_MODE: "ON"i | "OFF"i

parameter_option: values | ref | wildcard
covariate_option: values | ref | wildcard
_fp_options: fp_option | wildcard
!op_option: "+" | "*"

fp_option: FP_OP  | "[" [FP_OP ("," FP_OP)*] "]"
FP_OP: "LIN"i | "CAT"i | "CAT2"i | "PIECE_LIN"i | "EXP"i | "POW"i | "CUSTOM"i

_fp_var_options: fp_var_option | wildcard
fp_var_option: FP_VAR_OP  | "[" [FP_VAR_OP ("," FP_VAR_OP)*] "]"
FP_VAR_OP: "EXP"i | "ADD"i | "PROP"i | "LOG"i | "RE_LOG"i

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
