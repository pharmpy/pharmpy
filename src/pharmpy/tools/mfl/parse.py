import warnings
from collections import defaultdict
from itertools import product
from typing import List, Optional

from lark import Lark

from pharmpy.model import Model
from pharmpy.modeling.covariate_effect import get_covariates
from pharmpy.modeling.odes import (
    get_number_of_peripheral_compartments,
    get_number_of_transit_compartments,
    has_first_order_absorption,
    has_first_order_elimination,
    has_instantaneous_absorption,
    has_lag_time,
    has_michaelis_menten_elimination,
    has_mixed_mm_fo_elimination,
    has_seq_zo_fo_absorption,
    has_zero_order_absorption,
    has_zero_order_elimination,
)
from pharmpy.tools.mfl.feature.covariate import features as covariate_features
from pharmpy.tools.mfl.statement.definition import Let
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.symbols import Name
from pharmpy.tools.mfl.statement.feature.transits import Transits

from .grammar import grammar
from .helpers import all_funcs, funcs, modelsearch_features
from .interpreter import MFLInterpreter
from .statement.feature.covariate import Ref
from .statement.feature.symbols import Wildcard
from .statement.statement import Statement
from .stringify import stringify as mfl_stringify


def parse(code: str, mfl_class=False) -> List[Statement]:
    parser = Lark(
        grammar,
        start='start',
        parser='lalr',
        # lexer='standard',  # NOTE: This does not work because lexing for the
        #      MFL grammar is context-dependent
        propagate_positions=False,
        maybe_placeholders=False,
        debug=False,
        cache=True,
    )

    tree = parser.parse(code)

    mfl_statement_list = MFLInterpreter().interpret(tree)
    validate_mfl_list(mfl_statement_list)

    # TODO : only return class once it has been implemented everywhere
    if mfl_class:
        return ModelFeatures.create_from_mfl_statement_list(mfl_statement_list)
    else:
        return mfl_statement_list


def validate_mfl_list(mfl_statement_list):
    # TODO : Implement for other features as necessary
    optional_cov = set()
    mandatory_cov = set()
    # FIXME (?) : Allow for same exact cov effect to be forced by multiple explicit statements
    for s in mfl_statement_list:
        if isinstance(s, Covariate):
            if not s.optional.option and isinstance(s.fp, Wildcard):
                raise ValueError(
                    f"Error in {mfl_stringify([s])} :"
                    f" Mandatory effects need to be explicit (not '*')"
                )
            if not isinstance(s.parameter, Ref) and not isinstance(s.covariate, Ref):
                if s.optional.option:
                    optional_cov.update(product(s.parameter, s.covariate))
                else:
                    if error := [
                        e for e in product(s.parameter, s.covariate) if e in mandatory_cov
                    ]:
                        raise ValueError(
                            f"Covariate effect(s) {error} is being forced by"
                            f" multiple statements. Please force only once"
                        )
                    mandatory_cov.update(product(s.parameter, s.covariate))
    if error := [op for op in optional_cov if op in mandatory_cov]:
        raise ValueError(
            f"The covariate effect(s) {error} : are defined as both mandatory and optional"
        )


class ModelFeatures:
    def __init__(
        self,
        absorption=Absorption((Name('INST'),)),
        elimination=Elimination((Name('FO'),)),
        transits=(Transits((0,), (Name('DEPOT'),)),),  # NOTE : This is a tuple
        peripherals=Peripherals((0,)),
        lagtime=LagTime((Name('OFF'),)),
        covariate=tuple(),  # Note : Should always be tuple (empty meaning no covariates)
    ):
        self._absorption = absorption
        self._elimination = elimination
        self._transits = transits
        self._peripherals = peripherals
        self._lagtime = lagtime
        self._covariate = covariate

    @classmethod
    def create(
        cls,
        absorption=Absorption((Name('INST'),)),
        elimination=Elimination((Name('FO'),)),
        transits=(Transits((0,), (Name('DEPOT'),)),),
        peripherals=Peripherals((0,)),
        lagtime=LagTime((Name('OFF'),)),
        covariate=tuple(),
    ):
        # TODO : Check if allowed input value
        if not isinstance(absorption, Absorption):
            raise ValueError(f"Absorption : {absorption} is not suppoerted")

        if not isinstance(elimination, Elimination):
            raise ValueError(f"Elimination : {elimination} is not supported")

        if not isinstance(transits, tuple):
            raise ValueError("Transits need to be given within a tuple")
        if not all(isinstance(t, Transits) for t in transits):
            raise ValueError("All given elements of transits must be of type Transits")

        if not isinstance(peripherals, Peripherals):
            raise ValueError(f"Peripherals : {peripherals} is not supported")

        if not isinstance(lagtime, LagTime):
            raise ValueError(f"Lagtime : {lagtime} is not supported")

        if not covariate == tuple():
            if not isinstance(covariate, tuple):
                raise ValueError("Covariates need to be given within a tuple")
            if not all(isinstance(c, Covariate) for c in covariate):
                raise ValueError(f"Covariate : {covariate} is not supported")

        return cls(
            absorption=absorption,
            elimination=elimination,
            transits=transits,
            peripherals=peripherals,
            lagtime=lagtime,
            covariate=covariate,
        )

    @classmethod
    def create_from_mfl_statement_list(cls, mfl_list):
        m = ModelFeatures()
        absorption = m._default_values()['absorption']
        elimination = m._default_values()['elimination']
        transits = m._default_values()['transits']
        peripherals = m._default_values()['peripherals']
        lagtime = m._default_values()['lagtime']
        covariate = m._default_values()['covariate']
        let = {}
        for statement in mfl_list:
            if isinstance(statement, Absorption):
                absorption = statement
            elif isinstance(statement, Elimination):
                elimination = statement
            elif isinstance(statement, Transits):
                transits = (statement,)
            elif isinstance(statement, Peripherals):
                peripherals = statement
            elif isinstance(statement, LagTime):
                lagtime = statement
            elif isinstance(statement, Covariate):
                covariate += (statement,)
            elif isinstance(statement, Let):
                let[statement.name] = statement.value
            else:
                raise ValueError(f'Unknown ({type(statement)} statement ({statement}) given.')

        # Substitute all Let statements (if any)
        if len(let) != 0:
            # FIXME : Multiple let statements for the same reference value ?
            def _let_subs(cov, let):
                return Covariate(
                    parameter=cov.parameter
                    if not (isinstance(cov.parameter, Ref) and cov.parameter.name in let)
                    else let[cov.parameter.name],
                    covariate=cov.covariate
                    if not (isinstance(cov.covariate, Ref) and cov.covariate.name in let)
                    else let[cov.covariate.name],
                    fp=cov.fp,
                    op=cov.op,
                    optional=cov.optional,
                )

            covariate = tuple([_let_subs(cov, let) for cov in covariate])

        mfl = cls.create(
            absorption=absorption,
            elimination=elimination,
            transits=transits,
            peripherals=peripherals,
            lagtime=lagtime,
            covariate=covariate,
        )
        return mfl

    @classmethod
    def create_from_mfl_string(cls, mfl_string):
        return parse(mfl_string, mfl_class=True)

    def replace(self, **kwargs):
        absorption = kwargs.get("absorption", self._absorption)
        elimination = kwargs.get("elimination", self._elimination)
        transits = kwargs.get("transits", self._transits)
        peripherals = kwargs.get("peripherals", self._peripherals)
        lagtime = kwargs.get("lagtime", self._lagtime)
        covariate = kwargs.get("covariate", self._covariate)
        return ModelFeatures.create(
            absorption=absorption,
            elimination=elimination,
            transits=transits,
            peripherals=peripherals,
            lagtime=lagtime,
            covariate=covariate,
        )

    def _default_values(self):
        return {
            'absorption': Absorption((Name('INST'),)),
            'elimination': Elimination((Name('FO'),)),
            'transits': (Transits((0,), (Name('DEPOT'),)),),
            'peripherals': Peripherals((0,)),
            'lagtime': LagTime((Name('OFF'),)),
            'covariate': tuple(),
        }

    @property
    def absorption(self):
        return self._absorption

    @property
    def elimination(self):
        return self._elimination

    @property
    def transits(self):
        return self._transits

    @property
    def peripherals(self):
        return self._peripherals

    @property
    def lagtime(self):
        return self._lagtime

    @property
    def covariate(self):
        return self._covariate

    def expand(self, model):
        explicit_covariates = set(
            [
                p
                for c in self.covariate
                if (not isinstance(c.parameter, Ref) and not isinstance(c.covariate, Ref))
                for p in product(c.parameter, c.covariate)
            ]
        )  # Override @ reference with explicit value
        covariate = tuple(
            c for c in [c.eval(model, explicit_covariates) for c in self.covariate] if c is not None
        )

        param_cov = [
            p for c in covariate for p in product(c.parameter, c.covariate) if not c.optional.option
        ]
        counts = [(c, param_cov.count(c)) for c in param_cov]
        if any(c[1] > 1 for c in counts):
            error = set(c[0] for c in filter(lambda c: c[1] > 1, counts))
            raise ValueError(
                f"Covariate effect(s) {error} is forced by multiple reference statements."
                f" Please redefine the search space."
            )

        return ModelFeatures.create(
            absorption=self.absorption.eval,
            elimination=self.elimination.eval,
            transits=tuple([t.eval for t in self.transits]),
            peripherals=self.peripherals,
            lagtime=self.lagtime.eval,
            covariate=covariate,
        )

    def mfl_statement_list(self, attribute_type: Optional[List[str]] = []):
        """Add the repspective MFL attributes to a list"""

        # NOTE : This function is needed to be able to convert the classes to functions

        if not attribute_type:
            attribute_type = [
                "absorption",
                "elimination",
                "transits",
                "peripherals",
                "lagtime",
                "covariate",
            ]
        mfl_list = []
        if "absorption" in attribute_type:
            mfl_list.append(self.absorption)
        if "elimination" in attribute_type:
            mfl_list.append(self.elimination)
        if "transits" in attribute_type:
            for t in self.transits:
                mfl_list.append(t)
        if "peripherals" in attribute_type:
            mfl_list.append(self.peripherals)
        if "lagtime" in attribute_type:
            mfl_list.append(self.lagtime)
        if "covariate" in attribute_type:
            for c in self.covariate:
                mfl_list.append(c)

        return [m for m in mfl_list if m]

    def convert_to_funcs(
        self, attribute_type: Optional[List[str]] = None, model: Optional[Model] = None
    ):
        # The model argument is used for when extacting covariates.
        if not model:
            model = Model()
        # TODO : implement argument if we wish to use subset instead of all functions.
        if self.covariate != tuple():
            return all_funcs(model, self.mfl_statement_list(attribute_type))
        else:
            return funcs(model, self.mfl_statement_list(attribute_type), modelsearch_features)

    def contain_subset(self, mfl, model: Optional[Model] = None, tool: Optional[str] = None):
        """See if class contain specified subset"""
        transits = self._subset_transits(mfl)

        # FIXME : Add support for wildcard
        if (
            all([s in self.absorption.eval.modes for s in mfl.absorption.eval.modes])
            and all([s in self.elimination.eval.modes for s in mfl.elimination.eval.modes])
            and transits
            and all([s in self.peripherals.counts for s in mfl.peripherals.counts])
            and all([s in self.lagtime.eval.modes for s in mfl.lagtime.eval.modes])
        ):
            if tool is None or tool in ["modelsearch"]:
                return True
            else:
                if self.covariate != tuple() or mfl.covariate != tuple():
                    if model is None:
                        warnings.warn("Need argument 'model' in order to compare covariates")
                    else:
                        return True if self._subset_covariate(mfl, model) else False
        else:
            return False

    def _subset_transits(self, mfl):
        lhs_counts = set([c for t in self.transits for c in t.counts])
        lhs_depot = set([d for t in self.transits for d in t.eval.depot])

        rhs_counts = set([c for t in mfl.transits for c in t.counts])
        rhs_depot = set([d for t in mfl.transits for d in t.eval.depot])
        # FIXME : Need to compare counts per depot individually when comparing two
        # search spaces (Currenty working for model vs search space)
        return all([c in lhs_counts for c in rhs_counts]) and all(
            [d in lhs_depot for d in rhs_depot]
        )

    def _subset_covariates(self, mfl, model):
        lhs = defaultdict(list)
        rhs = defaultdict(list)

        for cov in self.covariate:
            cov_eval = cov.eval(model)
            for effect in cov_eval.fp:
                for op in cov_eval.op:
                    lhs[(effect, op)].append(product(cov_eval.parameter, cov_eval.covariate))
        for cov in mfl.covariate:
            cov_eval = cov.eval(model)
            for effect in cov_eval.fp:
                for op in cov_eval.op:
                    rhs[(effect, op)].append(product(cov_eval.parameter, cov_eval.covariate))

        for key in rhs.keys():
            if key not in lhs.keys():
                return False
            if all(p in lhs[key] for p in rhs[key]):
                continue
            else:
                return False
        return True

    def least_number_of_transformations(
        self, other, model: Optional[Model] = None, tool: Optional[str] = None
    ):
        """The smallest set of transformations to become part of other"""
        # Add more tools than "modelsearch" if support is needed
        lnt = {}
        if tool is None or tool in ["modelsearch"]:
            if not any(a in other.absorption.eval.modes for a in self.absorption.eval.modes):
                name, func = list(other.convert_to_funcs(["absorption"]).items())[0]
                lnt[name] = func

            if not any(e in other.elimination.eval.modes for e in self.elimination.eval.modes):
                name, func = list(other.convert_to_funcs(["elimination"]).items())[0]
                lnt[name] = func

            lnt = self._lnt_transits(other, lnt)

            if not any(p in other.peripherals.counts for p in self.peripherals.counts):
                name, func = list(other.convert_to_funcs(["peripherals"]).items())[0]
                lnt[name] = func

            if not any(lt in other.lagtime.eval.modes for lt in self.lagtime.eval.modes):
                name, func = list(other.convert_to_funcs(["lagtime"]).items())[0]
                lnt[name] = func

        # TODO : Use in covsearch instead of taking diff
        if tool is None:
            if model is not None:
                lnt = self._lnt_covariates(other, lnt, model)
            else:
                if self.covariate != tuple() or other.covariate != tuple():
                    warnings.warn("Need argument 'model' in order to compare covariates")

        return lnt

    def _lnt_transits(self, other, lnt):
        lhs_depot = [d for t in self.transits for d in t.eval.depot]
        rhs_depot = [d for t in other.transits for d in t.eval.depot]

        if not any(td in rhs_depot for td in lhs_depot):
            # DEPOT does not match -> Take first function regardless of counts
            name, func = list(other.convert_to_funcs(["transits"]).items())[0]
            lnt[name] = func
        else:
            # DEPOT does match -> Need to check counts for corresponding depot
            # First check counts of all matching depots
            for depot in lhs_depot:
                if depot in rhs_depot:
                    depot_counts = [
                        c for t in self.transits for c in t.counts if depot in t.eval.depot
                    ]
                    rhs_depot_counts = [
                        c for t in other.transits for c in t.counts if depot in t.eval.depot
                    ]
                    diff = [c for c in rhs_depot_counts if c not in depot_counts]
                    match = [c for c in rhs_depot_counts if c in depot_counts]

                    if len(match) != 0:
                        return lnt
                    # There is a difference in set of counts for the same depot argument
                    if len(diff) != 0:
                        func_dict = other.convert_to_funcs(["transits"])
                        lnt[('TRANSITS', diff[0], depot.name)] = func_dict[
                            ('TRANSITS', diff[0], depot.name)
                        ]

                        return lnt

            # else take first count value of non matching depot
            depot = next(d for d in rhs_depot if d not in lhs_depot)
            count = next(c for t in other.transits for c in t.counts if depot in t.eval.depot)
            func_dict = other.convert_to_funcs(["transits"])
            lnt[('TRANSITS', count, depot.name)] = func_dict[('TRANSITS', diff[0], depot.name)]

        return lnt

    def _lnt_covariates(self, other, lnt, model):
        lhs = self._extract_covariates()
        rhs = other._extract_covariates()

        remove_cov_dict = dict(covariate_features(model, self.covariate, remove=True))
        if not any(key in rhs.keys() for key in lhs.keys()):
            # Remove all covariates
            lnt.update(remove_cov_dict)
            if len(rhs) != 0:
                # No effect/operator combination matching
                func_dict = other.convert_to_funcs(["covariate"])
                key = list(func_dict.keys())[0]
                lnt[key] = func_dict[key]
        else:
            # One (or more) effect/operator are overlapping
            for key in lhs.keys():
                if key in rhs.keys():
                    if not any(p in rhs[key] for p in lhs[key]):
                        # take the first value with this key.
                        func_dict = other.convert_to_funcs(["covariate"], model)
                        func_dict = {
                            k: v
                            for k, v in func_dict.items()
                            if (k[3] == key[0] and k[4] == key[1])
                        }
                        key = list(func_dict.keys())[0]
                        lnt[key] = func_dict[key]
                    for p in [p for p in lhs[key] if p not in rhs[key]]:
                        lnt[(p[0], p[1], key[0], key[1])] = remove_cov_dict[
                            (p[0], p[1], key[0], key[1])
                        ]

        return lnt

    def __repr__(self):
        # TODO : Remove default values
        return mfl_stringify(self.mfl_statement_list())

    def __sub__(self, other):
        transits = self._add_sub_transits(other, add=False)
        covariates = self._add_sub_covariates(other, add=False)

        return ModelFeatures.create(
            absorption=self.absorption - other.absorption,
            elimination=self.elimination - other.elimination,
            transits=transits,
            peripherals=self.peripherals - other.peripherals,
            lagtime=self.lagtime - other.lagtime,
            covariate=covariates,
        )

    def __add__(self, other):
        transits = self._add_sub_transits(other, add=True)
        covariates = self._add_sub_covariates(other, add=True)

        return ModelFeatures.create(
            absorption=self.absorption + other.absorption,
            elimination=self.elimination + other.elimination,
            transits=transits,
            peripherals=self.peripherals + other.peripherals,
            lagtime=self.lagtime + other.lagtime,
            covariate=covariates,
        )

    def _add_sub_transits(self, other, add=True):
        """Apply logic for adding/subtracting mfl transits.
        Use add = False for subtraction"""

        # Need multiple Transit objects due to presence of both depot / nodepot
        # and different sets of counts
        lhs_counts_depot = [
            c for t in self.transits if Name("DEPOT") in t.eval.depot for c in t.counts
        ]
        lhs_counts_nodepot = [
            c for t in self.transits if Name("NODEPOT") in t.eval.depot for c in t.counts
        ]
        rhs_counts_depot = [
            c for t in other.transits if Name("DEPOT") in t.eval.depot for c in t.counts
        ]
        rhs_counts_nodepot = [
            c for t in other.transits if Name("NODEPOT") in t.eval.depot for c in t.counts
        ]

        if add:
            depot_counts = list(set(lhs_counts_depot + rhs_counts_depot))
            nodepot_counts = list(set(lhs_counts_nodepot + rhs_counts_nodepot))
        else:
            depot_counts = [c for c in lhs_counts_depot if c not in rhs_counts_depot]
            nodepot_counts = [c for c in lhs_counts_nodepot if c not in rhs_counts_nodepot]

        if len(depot_counts) == len(nodepot_counts) == 0:
            transits = (
                Transits(
                    (0,),
                    (Name("DEPOT"),),
                ),
            )
        else:
            both_depot = [d for d in depot_counts if d in nodepot_counts]
            nodepot_diff = [d for d in nodepot_counts if d not in depot_counts]
            depot_diff = [d for d in depot_counts if d not in nodepot_counts]

            transits = []
            if len(both_depot) != 0:
                transits.append(Transits(tuple(both_depot), (Name("DEPOT"), Name("NODEPOT"))))
            if len(depot_diff) != 0:
                transits.append(Transits(tuple(depot_diff), (Name("DEPOT"),)))
            if len(nodepot_diff) != 0:
                transits.append(Transits(tuple(nodepot_diff), (Name("NODEPOT"),)))

        return tuple(transits)

    def _add_sub_covariates(self, other, add=True):
        lhs = self._extract_covariates()
        rhs = other._extract_covariates()

        res = []
        if len(rhs) != 0 and len(lhs) != 0:
            # Find the unique products in both lists with matching expression/operator
            combined = lhs
            for effop, prod in rhs.items():
                if add:
                    combined[effop].update(prod)
                else:
                    combined[effop].difference_update(prod)
            combined = {k: v for k, v in combined.items() if v != set()}
            res = combined
        elif len(rhs) != 0 and len(lhs) == 0 and add:
            res = rhs
        elif len(lhs) != 0 and len(rhs) == 0:
            res = lhs
        else:
            return tuple()

        # Clean and combine all effect/operators with the same set of parameter and covariates
        clean_res = defaultdict(list)
        for effop, prod in res.items():
            prod_key = tuple(sorted(prod))
            if existing_effop := clean_res[prod_key]:
                if effop[0] not in existing_effop[0]:
                    existing_effop[0].append(effop[0])
                if effop[1] not in existing_effop[1]:
                    existing_effop[1].append(effop[1])
                clean_res[prod_key] = existing_effop
            else:
                clean_res[prod_key] = [[effop[0]], [effop[1]]]

        cov_res = []
        for prod, effop in clean_res.items():
            parameter = tuple(set([p[0] for p in prod]))
            covariate = tuple(set([p[1] for p in prod]))
            cov_res.append(Covariate(parameter, covariate, tuple(effop[0]), tuple(effop[1])))

        if all(len(c.parameter) == 0 for c in cov_res):
            return tuple()
        else:
            return tuple(cov_res)

    def __eq__(self, other):
        transits = self._eq_transits(other)
        return (
            self.absorption == other.absorption
            and self.elimination == other.elimination
            and transits
            and self.peripherals == other.peripherals
            and self.lagtime == other.lagtime
            and self._eq_covariate(other)
        )

    def _eq_transits(self, other):
        lhs_counts_depot = [
            c for t in self.transits if Name("DEPOT") in t.eval.depot for c in t.counts
        ]
        lhs_counts_nodepot = [
            c for t in self.transits if Name("NODEPOT") in t.eval.depot for c in t.counts
        ]
        rhs_counts_depot = [
            c for t in other.transits if Name("DEPOT") in t.eval.depot for c in t.counts
        ]
        rhs_counts_nodepot = [
            c for t in other.transits if Name("NODEPOT") in t.eval.depot for c in t.counts
        ]
        return set(lhs_counts_depot) == set(rhs_counts_depot) and set(lhs_counts_nodepot) == set(
            rhs_counts_nodepot
        )

    def _eq_covariate(self, other):
        lhs = self._extract_covariates()
        rhs = other._extract_covariates()

        if all(key in rhs.keys() for key in lhs.keys()):
            for key in lhs.keys():
                if lhs[key] != rhs[key]:
                    return False
            return True
        return False

    def _extract_covariates(self):
        lhs = defaultdict(set)
        lhs_ref = []
        for cov in self.covariate:
            if isinstance(cov.parameter, Ref) or isinstance(cov.covariate, Ref):
                lhs_ref.append(cov)
                continue
            for effect in cov.eval().fp:
                for op in cov.op:
                    # TODO : Ignore using PROD and simply use tuple with one value each
                    lhs[(effect.lower(), op)].update(set(product(cov.parameter, cov.covariate)))

        if len(lhs_ref) != 0:
            raise ValueError(
                'Cannot be performed with reference value. Try using .expand(model) first.'
            )

        return lhs


def get_model_features(model: Model, supress_warnings: bool = False) -> str:
    """Create an MFL representation of an input model

    Given an input model. Create a model feature language (MFL) string
    representation. Can currently extract absorption, elimination, transits,
    peripherals and lagtime.

    Parameters
    ----------
    model : Model
        Model to extract features from.
    supress_warnings : TYPE, optional
        Choose to supress warnings if absorption/elimination type cannot be
        determined. The default is False.

    Returns
    -------
    str
        A MFL string representation of the input model.

    """
    # ABSORPTION
    absorption = None
    if has_seq_zo_fo_absorption(model):
        absorption = "SEQ-ZO-FO"
    elif has_zero_order_absorption(model):
        absorption = "ZO"
    elif has_first_order_absorption(model):
        absorption = "FO"
    elif has_instantaneous_absorption(model):
        absorption = "INST"

    if not supress_warnings:
        if absorption is None:
            warnings.warn("Could not determine absorption of model.")

    # ElIMINATION
    elimination = None
    if has_mixed_mm_fo_elimination(model):
        elimination = "MIX-FO-MM"
    elif has_zero_order_elimination(model):
        elimination = "ZO"
    elif has_first_order_elimination(model):
        elimination = "FO"
    elif has_michaelis_menten_elimination(model):
        elimination = "MM"

    if not supress_warnings:
        if elimination is None:
            warnings.warn("Could not determine elimination of model.")

    # ABSORPTION DELAY (TRANSIT AND LAGTIME)
    # TRANSITS
    transits = get_number_of_transit_compartments(model)

    # TODO : DEPOT
    if not model.statements.ode_system.find_depot(model.statements):
        depot = "NODEPOT"
    else:
        depot = "DEPOT"

    lagtime = has_lag_time(model)
    if not lagtime:
        lagtime = None

    # DISTRIBUTION (PERIPHERALS)
    peripherals = get_number_of_peripheral_compartments(model)

    # COVARIATES
    covariates = get_covariates(model)

    if absorption:
        absorption = f'ABSORPTION({absorption})'
    if elimination:
        elimination = f'ELIMINATION({elimination})'
    if lagtime:
        lagtime = "LAGTIME(ON)"
    if transits != 0:
        transits = f'TRANSITS({transits}{","+depot})'
    if peripherals != 0:
        peripherals = f'PERIPHERALS({peripherals})'
    if len(covariates) != 0:
        # FIXME : More extensive cleanup
        clean_cov = defaultdict(list)
        for key, value in covariates.items():
            clean_cov[(key[1], value[0][0], value[0][1])].append(key[0])

        cov_list = [
            f'COVARIATE({value},{key[0]},{key[1]},{key[2]})' for key, value in clean_cov.items()
        ]

        covariates = ';'.join(cov_list)
        # Remove quotes from parameter names
        covariates = covariates.replace("'", '')
    else:
        covariates = None

    # TODO : Implement IIV
    return ";".join(
        [
            e
            for e in [absorption, elimination, lagtime, transits, peripherals, covariates]
            if (e is not None and e != 0)
        ]
    )
