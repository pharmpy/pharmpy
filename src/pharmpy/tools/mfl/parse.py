import warnings
from collections import defaultdict
from itertools import product
from typing import List, Optional

from lark import Lark

from pharmpy.model import Model
from pharmpy.modeling.covariate_effect import get_covariate_effects
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
from pharmpy.tools.mfl.statement.feature.direct_effect import DirectEffect
from pharmpy.tools.mfl.statement.feature.effect_comp import EffectComp
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.indirect_effect import IndirectEffect
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.metabolite import Metabolite
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.symbols import Name
from pharmpy.tools.mfl.statement.feature.transits import Transits

from .grammar import grammar
from .helpers import (
    all_funcs,
    funcs,
    modelsearch_features,
    structsearch_metabolite_features,
    structsearch_pd_features,
)
from .interpreter import MFLInterpreter
from .statement.feature.covariate import Ref
from .statement.feature.symbols import Option, Wildcard
from .statement.statement import Statement
from .stringify import stringify as mfl_stringify


def parse(code: str, mfl_class=False) -> List[Statement]:
    mfl_statement_list = _parse(code)

    # TODO : only return class once it has been implemented everywhere
    if mfl_class:
        return ModelFeatures.create_from_mfl_statement_list(mfl_statement_list)
    else:
        return mfl_statement_list


def _parse(code: str):
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


class ModelFeatures:
    def __init__(
        self,
        absorption=None,
        elimination=None,
        transits=tuple(),  # NOTE : This is a tuple
        peripherals=tuple(),
        lagtime=None,
        covariate=tuple(),  # Note : Should always be tuple (empty meaning no covariates)
        direct_effect=None,
        effect_comp=None,
        indirect_effect=tuple(),
        metabolite=None,
    ):
        self._absorption = absorption
        self._elimination = elimination
        self._transits = transits
        self._peripherals = peripherals
        self._lagtime = lagtime
        self._covariate = covariate
        self._direct_effect = direct_effect
        self._effect_comp = effect_comp
        self._indirect_effect = indirect_effect
        self._metabolite = metabolite

    @classmethod
    def create(
        cls,
        absorption=None,
        elimination=None,
        transits=tuple(),
        peripherals=tuple(),
        lagtime=None,
        covariate=tuple(),
        direct_effect=None,
        effect_comp=None,
        indirect_effect=tuple(),
        metabolite=None,
    ):
        # TODO : Check if allowed input value
        if absorption is not None and not isinstance(absorption, Absorption):
            raise ValueError(f"Absorption : {absorption} is not suppoerted")

        if elimination is not None and not isinstance(elimination, Elimination):
            raise ValueError(f"Elimination : {elimination} is not supported")

        if not isinstance(transits, tuple):
            raise ValueError("Transits need to be given within a tuple")
        if not all(isinstance(t, Transits) for t in transits):
            raise ValueError("All given elements of transits must be of type Transits")

        if not isinstance(peripherals, tuple):
            raise ValueError("Peripherals need to be given within a tuple")
        if not all(isinstance(p, Peripherals) for p in peripherals):
            raise ValueError(f"Peripherals : {peripherals} is not supported")

        if lagtime is not None and not isinstance(lagtime, LagTime):
            raise ValueError(f"Lagtime : {lagtime} is not supported")

        if not isinstance(covariate, tuple):
            raise ValueError("Covariates need to be given within a tuple")
        if not all(isinstance(c, Covariate) for c in covariate):
            raise ValueError(f"Covariate : {covariate} is not supported")

        if direct_effect is not None and not isinstance(direct_effect, DirectEffect):
            raise ValueError(f"DirectEffect : {direct_effect} is not supported")

        if effect_comp is not None and not isinstance(effect_comp, EffectComp):
            raise ValueError(f"EffectComp : {effect_comp} is not supported")

        if not isinstance(indirect_effect, tuple):
            raise ValueError("IndirectEffect(s) need to be given within a tuple")
        if not all(isinstance(i, IndirectEffect) for i in indirect_effect):
            raise ValueError("All given elements of indirect_effect must be of type IndirectEffect")

        if metabolite is not None and not isinstance(metabolite, Metabolite):
            raise ValueError(f"Metabolite : {metabolite} is not supported")

        # Indicate that we have a PK model, need default features
        if any(x for x in [absorption, elimination, transits, peripherals, lagtime, metabolite]):
            if absorption is None:
                absorption = Absorption((Name('INST'),))
            if elimination is None:
                elimination = Elimination((Name('FO'),))
            if transits == tuple():
                transits = (Transits((0,), (Name('DEPOT'),)),)
            if peripherals == tuple():
                peripherals += (Peripherals((0,)),)
            if lagtime is None:
                lagtime = LagTime((Name('OFF'),))

        return cls(
            absorption=absorption,
            elimination=elimination,
            transits=transits,
            peripherals=peripherals,
            lagtime=lagtime,
            covariate=covariate,
            direct_effect=direct_effect,
            effect_comp=effect_comp,
            indirect_effect=indirect_effect,
            metabolite=metabolite,
        )

    @classmethod
    def create_from_mfl_statement_list(cls, mfl_list):
        absorption = None
        elimination = None
        transits = tuple()
        peripherals = tuple()
        lagtime = None
        covariate = tuple()
        direct_effect = None
        effect_comp = None
        indirect_effect = tuple()
        metabolite = None
        let = {}

        for statement in mfl_list:
            if isinstance(statement, Absorption):
                absorption = absorption + statement if absorption else statement
            elif isinstance(statement, Elimination):
                elimination = elimination + statement if elimination else statement
            elif isinstance(statement, Transits):
                transits += (statement,)
            elif isinstance(statement, Peripherals):
                peripherals += (statement,)
            elif isinstance(statement, LagTime):
                lagtime = lagtime + statement if lagtime else statement
            elif isinstance(statement, Covariate):
                covariate += (statement,)
            elif isinstance(statement, Let):
                let[statement.name] = statement.value
            elif isinstance(statement, DirectEffect):
                direct_effect = direct_effect + statement if direct_effect else statement
            elif isinstance(statement, EffectComp):
                effect_comp = effect_comp + statement if effect_comp else statement
            elif isinstance(statement, IndirectEffect):
                indirect_effect += (statement,)
            elif isinstance(statement, Metabolite):
                metabolite = metabolite + statement if metabolite else statement
            else:
                raise ValueError(f'Unknown ({type(statement)} statement ({statement}) given.')

        # Substitute all Let statements (if any)
        if len(let) != 0:
            # FIXME : Multiple let statements for the same reference value ?
            def _let_subs(cov, let):
                return Covariate(
                    parameter=(
                        cov.parameter
                        if not (isinstance(cov.parameter, Ref) and cov.parameter.name in let)
                        else let[cov.parameter.name]
                    ),
                    covariate=(
                        cov.covariate
                        if not (isinstance(cov.covariate, Ref) and cov.covariate.name in let)
                        else let[cov.covariate.name]
                    ),
                    fp=cov.fp,
                    op=cov.op,
                    optional=cov.optional,
                )

            # Add other attributes as necessary
            covariate = tuple([_let_subs(cov, let) for cov in covariate])

        mfl = cls.create(
            absorption=absorption,
            elimination=elimination,
            transits=transits,
            peripherals=peripherals,
            lagtime=lagtime,
            covariate=covariate,
            direct_effect=direct_effect,
            effect_comp=effect_comp,
            indirect_effect=indirect_effect,
            metabolite=metabolite,
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
        direct_effect = kwargs.get("direct_effect", self._direct_effect)
        effect_comp = kwargs.get("effect_comp", self._effect_comp)
        indirect_effect = kwargs.get("indirect_effect", self._indirect_effect)
        metabolite = kwargs.get("metabolite", self._metabolite)
        return ModelFeatures.create(
            absorption=absorption,
            elimination=elimination,
            transits=transits,
            peripherals=peripherals,
            lagtime=lagtime,
            covariate=covariate,
            direct_effect=direct_effect,
            effect_comp=effect_comp,
            indirect_effect=indirect_effect,
            metabolite=metabolite,
        )

    def replace_features(self, mfl_str):
        key_dict = {
            Absorption: 'absorption',
            Elimination: 'elimination',
            Transits: 'transits',
            Peripherals: 'peripherals',
            LagTime: 'lagtime',
            Covariate: 'covariate',
            DirectEffect: 'direct_effect',
            EffectComp: 'effect_comp',
            IndirectEffect: 'indirect_effect',
            Metabolite: 'metabolite',
        }
        mfl_list = _parse(mfl_str)
        kwargs = {}
        for statement in mfl_list:
            key = key_dict[type(statement)]
            if type(statement) in (Transits, Peripherals, Covariate, IndirectEffect):
                value = (statement,)
            else:
                value = statement
            if key in kwargs.keys():
                kwargs[key] += value
            else:
                kwargs[key] = value
        return self.replace(**kwargs)

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

    @property
    def direct_effect(self):
        return self._direct_effect

    @property
    def effect_comp(self):
        return self._effect_comp

    @property
    def indirect_effect(self):
        return self._indirect_effect

    @property
    def metabolite(self):
        return self._metabolite

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

        # Overwrite optional covariates if forced
        if covariate:
            covariate_combinations = set()
            for cov in covariate:
                covariate_combinations.update(
                    set(
                        product(
                            cov.parameter, cov.covariate, cov.eval().fp, (cov.op,), (cov.optional,)
                        )
                    )
                )
            for combination in covariate_combinations.copy():
                if not combination[4].option:
                    opposite = combination[:-1] + (Option(True),)
                    covariate_combinations.discard(opposite)

            covariate_combinations = [tuple({x} for x in e) for e in covariate_combinations]
            covariate_combinations = _reduce_covariate(covariate_combinations)
            covariate = tuple()
            for param, cov, fp, op, opt in covariate_combinations:
                covariate += (
                    Covariate(tuple(param), tuple(cov), tuple(fp), list(op)[0], list(opt)[0]),
                )

        return ModelFeatures.create(
            absorption=self.absorption.eval if self.absorption else None,
            elimination=self.elimination.eval if self.elimination else None,
            transits=tuple([t.eval for t in self.transits]),
            peripherals=tuple([p.eval for p in self.peripherals]),
            lagtime=self.lagtime.eval if self.lagtime else None,
            covariate=covariate,
            direct_effect=self.direct_effect.eval if self.direct_effect else None,
            effect_comp=self.effect_comp.eval if self.effect_comp else None,
            indirect_effect=tuple([i.eval for i in self.indirect_effect]),
            metabolite=self.metabolite.eval if self.metabolite else None,
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
                "direct_effect",
                "effect_comp",
                "indirect_effect",
                "metabolite",
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
            for p in self.peripherals:
                mfl_list.append(p)
        if "lagtime" in attribute_type:
            mfl_list.append(self.lagtime)
        if "covariate" in attribute_type:
            for c in self.covariate:
                mfl_list.append(c)
        if "direct_effect" in attribute_type:
            mfl_list.append(self.direct_effect)
        if "effect_comp" in attribute_type:
            mfl_list.append(self.effect_comp)
        if "indirect_effect" in attribute_type:
            for i in self.indirect_effect:
                mfl_list.append(i)
        if "metabolite" in attribute_type:
            mfl_list.append(self.metabolite)

        return [m for m in mfl_list if m is not None]

    def filter(self, subset):
        if subset == "pk":
            peripherals = self._extract_peripherals()
            if peripherals["DRUG"]:
                peripherals = (Peripherals(tuple(peripherals["DRUG"]), (Name("DRUG"),)),)
            else:
                peripherals = tuple()
            return ModelFeatures.create(
                absorption=self.absorption,
                elimination=self.elimination,
                transits=self.transits,
                peripherals=peripherals,
                lagtime=self.lagtime,
            )
        elif subset == "pd":
            return ModelFeatures.create(
                direct_effect=self.direct_effect,
                effect_comp=self.effect_comp,
                indirect_effect=self.indirect_effect,
            )
        elif subset == "metabolite":
            peripherals = self._extract_peripherals()
            if peripherals["MET"]:
                peripherals = (Peripherals(tuple(peripherals["MET"]), (Name("MET"),)),)
            else:
                peripherals = tuple()
            return ModelFeatures.create(peripherals=peripherals, metabolite=self.metabolite)
        else:
            raise ValueError(f"Unknown subset {subset}")

    def convert_to_funcs(
        self,
        attribute_type: Optional[List[str]] = None,
        model: Optional[Model] = None,
        subset_features=None,
    ):
        if subset_features == "pk":
            filtered_mfl = self.filter(subset_features)
            return funcs(
                model, filtered_mfl.mfl_statement_list(attribute_type), modelsearch_features
            )
        elif subset_features == "pd":
            filtered_mfl = self.filter(subset_features)
            return funcs(
                model, filtered_mfl.mfl_statement_list(attribute_type), structsearch_pd_features
            )
        elif subset_features == "metabolite":
            filtered_mfl = self.filter(subset_features)
            return funcs(
                model,
                filtered_mfl.mfl_statement_list(attribute_type),
                structsearch_metabolite_features,
            )
        else:
            # The model argument is used for when extacting covariates.
            if not model:
                model = Model()
            return all_funcs(model, self.mfl_statement_list(attribute_type))

    def contain_subset(self, mfl, model: Optional[Model] = None, tool: Optional[str] = None):
        """See if class contain specified subset"""
        transits = self._subset_transits(mfl)
        peripheral_lhs = self._extract_peripherals()
        peripheral_rhs = mfl._extract_peripherals()

        if (
            all([s in self.absorption.eval.modes for s in mfl.absorption.eval.modes])
            and all([s in self.elimination.eval.modes for s in mfl.elimination.eval.modes])
            and transits
            and all([s in self.lagtime.eval.modes for s in mfl.lagtime.eval.modes])
        ):
            if tool is None or tool in ["modelsearch"]:
                return all(p in peripheral_lhs["DRUG"] for p in list(peripheral_rhs["DRUG"]))
            else:
                if not (
                    all(p in peripheral_lhs["DRUG"] for p in list(peripheral_rhs["DRUG"]))
                    and all(p in peripheral_lhs["MET"] for p in list(peripheral_rhs["MET"]))
                ):
                    return False
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

        def _lnt_helper(lhs, rhs, mfl, name, lnt):
            if lhs is None and rhs is not None or lhs is not None and rhs is None:
                raise ValueError(
                    f"{name} : is only part of one of the MFLs" " and therefore cannot be compared"
                )
            if lhs is None and rhs is None:
                return lnt
            if not any(x in rhs.eval.modes for x in lhs.eval.modes):
                name, func = list(mfl.convert_to_funcs([name]).items())[0]
                lnt[name] = func
            return lnt

        # Add more tools than "modelsearch" if support is needed
        lnt = {}
        if tool is None or tool in ["modelsearch"]:
            lnt = _lnt_helper(self.absorption, other.absorption, other, "absorption", lnt)

            lnt = _lnt_helper(self.elimination, other.elimination, other, "elimination", lnt)

            lnt = self._lnt_transits(other, lnt)

            lnt = self._lnt_peripherals(other, lnt, "pk")

            lnt = _lnt_helper(self.lagtime, other.lagtime, other, "lagtime", lnt)

        # TODO : Use in covsearch instead of taking diff
        if tool is None:
            if model is not None:
                lnt = self._lnt_covariates(other, lnt, model)
            else:
                if self.covariate != tuple() or other.covariate != tuple():
                    warnings.warn("Need argument 'model' in order to compare covariates")

            lnt = self._lnt_peripherals(other, lnt, "metabolite")

            lnt = _lnt_helper(self.direct_effect, other.direct_effect, other, "direct_effect", lnt)

            lnt = _lnt_helper(self.effect_comp, other.effect_comp, other, "effect_comp", lnt)

            lnt = self._lnt_indirect_effect(other, lnt)

            lnt = _lnt_helper(self.metabolite, other.metabolite, other, "metabolite", lnt)

        return lnt

    def _lnt_indirect_effect(self, other, lnt):
        lhs, rhs, combine = _add_helper(
            self.indirect_effect, other.indirect_effect, "modes", "production"
        )
        if not combine and rhs:
            # No shared attribute
            func_dict = other.convert_to_funcs(["indirect_effect"])
            for key in lhs.keys():
                if key in rhs.keys():
                    lnt[('INDIRECT', rhs[key][0], key.name)] = func_dict[
                        ('INDIRECT', rhs[key][0], key.name)
                    ]
                    return lnt
            # No key is matching
            key = next(iter(rhs))
            lnt[('INDIRECT', rhs[key][0], key.name)] = func_dict[
                ('INDIRECT', rhs[key][0], key.name)
            ]
            return lnt
        return lnt

    def _lnt_transits(self, other, lnt):
        lhs, rhs, combine = _add_helper(self.transits, other.transits, "counts", "depot")
        if not combine and rhs:
            # No shared attribute
            func_dict = other.convert_to_funcs(["transits"])
            for key in lhs.keys():
                if key in rhs.keys():
                    lnt[('TRANSITS', rhs[key][0], key.name)] = func_dict[
                        ('TRANSITS', rhs[key][0], key.name)
                    ]
                    return lnt
            # No key is matching
            key = next(iter(rhs))
            lnt[('TRANSITS', rhs[key][0], key.name)] = func_dict[
                ('TRANSITS', rhs[key][0], key.name)
            ]
            return lnt
        return lnt

    def _lnt_peripherals(self, other, lnt, subset):
        if subset == "pk":
            keys = ["DRUG"]
        if subset == "metabolite":
            keys = ["MET"]
        else:
            keys = ["DRUG", "MET"]
        lhs = self._extract_peripherals()
        rhs = other._extract_peripherals()
        func_dict = other.convert_to_funcs(["peripherals"])
        for key in keys:
            if not any(c in rhs[key] for c in lhs[key]):
                if key == "DRUG":
                    if rhs[key]:
                        lnt[("PERIPHERALS", min(rhs[key]))] = func_dict[
                            ("PERIPHERALS", min(rhs[key]))
                        ]
                elif key == "MET":
                    if rhs[key]:
                        lnt[("PERIPHERALS", min(rhs[key]), "METABOLITE")] = func_dict[
                            ("PERIPHERALS", min(rhs[key]), "METABOLITE")
                        ]
        return lnt

    def _lnt_covariates(self, other, lnt, model):
        lhs = self._extract_covariates()
        lhs = [c for c in lhs if not c[4].option]  # Check only FORCED
        rhs = other._extract_covariates()
        rhs = [c for c in rhs if not c[4].option]  # Check only FORCED

        def convert_to_covariate(combinations):
            cov_list = []
            for param, cov, fp, op, opt in combinations:
                cov_list.append(Covariate((param,), (cov,), (fp,), op, opt))
            return cov_list

        # Remove all unqiue to LHS
        lhs_unique = [c for c in lhs if c not in rhs]
        if lhs_unique:
            cov_list = convert_to_covariate(lhs_unique)
            lhs_mfl = ModelFeatures.create_from_mfl_statement_list(cov_list)
            remove_cov_dict = dict(covariate_features(model, lhs_mfl.covariate, remove=True))
            lnt.update(remove_cov_dict)

        # Add all unique to RHS
        rhs_unique = [c for c in rhs if c not in lhs]
        if rhs_unique:
            cov_list = convert_to_covariate(rhs_unique)
            lhs_mfl = ModelFeatures.create_from_mfl_statement_list(cov_list)
            add_cov_dict = dict(covariate_features(model, lhs_mfl.covariate, remove=False))
            lnt.update(add_cov_dict)

        return lnt

    def __repr__(self):
        # TODO : Remove default values
        return mfl_stringify(self.mfl_statement_list())

    def __sub__(self, other):
        def sub(lhs, rhs):
            if lhs:
                if rhs:
                    if lhs == rhs:
                        return None
                    else:
                        return lhs - rhs
                else:
                    return lhs
            else:
                return lhs

        transits = self._add_sub_transits(other, add=False)
        peripherals = self._add_sub_peripherals(other, add=False)
        covariates = self._add_sub_covariates(other, add=False)
        indirect_effect = self._add_sub_indirect_effect(other, add=False)

        return ModelFeatures.create(
            absorption=sub(self.absorption, other.absorption),
            elimination=sub(self.elimination, other.elimination),
            transits=transits,
            peripherals=peripherals,
            lagtime=sub(self.lagtime, other.lagtime),
            covariate=covariates,
            direct_effect=sub(self.direct_effect, other.direct_effect),
            effect_comp=sub(self.effect_comp, other.effect_comp),
            indirect_effect=indirect_effect,
            metabolite=sub(self.metabolite, other.metabolite),
        )

    def __add__(self, other):
        def add(lhs, rhs):
            if lhs:
                if rhs:
                    return lhs + rhs
                else:
                    return lhs
            elif rhs:
                return rhs
            else:
                return lhs

        transits = self._add_sub_transits(other, add=True)
        peripherals = self._add_sub_peripherals(other, add=True)
        covariates = self._add_sub_covariates(other, add=True)
        indirect_effect = self._add_sub_indirect_effect(other, add=True)

        return ModelFeatures.create(
            absorption=add(self.absorption, other.absorption),
            elimination=add(self.elimination, other.elimination),
            transits=transits,
            peripherals=peripherals,
            lagtime=add(self.lagtime, other.lagtime),
            covariate=covariates,
            direct_effect=add(self.direct_effect, other.direct_effect),
            effect_comp=add(self.effect_comp, other.effect_comp),
            indirect_effect=indirect_effect,
            metabolite=add(self.metabolite, other.metabolite),
        )

    def _add_sub_peripherals(self, other, add=True):
        lhs = self._extract_peripherals()
        rhs = other._extract_peripherals()

        combined = {"MET": tuple(), "DRUG": tuple()}

        for key in combined.keys():
            if add:
                combined[key] = tuple(lhs[key].union(rhs[key]))
            else:
                combined[key] = tuple(lhs[key].difference(rhs[key]))

        peripherals = []
        for k, v in combined.items():
            if v:  # Not an empty tuple
                peripherals.append(Peripherals(v, (Name(k),)))
        if peripherals:
            return tuple(peripherals)
        else:
            return tuple()

    def _add_sub_indirect_effect(self, other, add=True):
        """Apply logic for adding/subtracting mfl IndirectEffect(s).
        Use add = False for subtraction"""
        # TODO : combine with _add_sub_transits()
        lhs, rhs, combined = _add_helper(
            self.indirect_effect, other.indirect_effect, "modes", "production"
        )

        def convert_to_indirect_effect(d):
            indirect_effects = []
            for k, v in d.items():
                indirect_effects.append(IndirectEffect(v, (k,)))
            return indirect_effects

        lhs_indirect_effects = convert_to_indirect_effect(lhs)
        rhs_indirect_effects = convert_to_indirect_effect(rhs)
        combined_indirect_effects = convert_to_indirect_effect(combined)

        # TODO : Cleanup and combine all possible statements
        if add:
            return tuple(lhs_indirect_effects + rhs_indirect_effects + combined_indirect_effects)
        else:
            if lhs_indirect_effects:
                return tuple(lhs_indirect_effects)
            else:
                return tuple()

    def _add_sub_transits(self, other, add=True):
        """Apply logic for adding/subtracting mfl transits.
        Use add = False for subtraction"""
        lhs, rhs, combined = _add_helper(self.transits, other.transits, "counts", "depot")

        def convert_to_transits(d):
            transits = []
            for k, v in d.items():
                transits.append(Transits(v, (k,)))
            return transits

        lhs_transits = convert_to_transits(lhs)
        rhs_transits = convert_to_transits(rhs)
        combined_transits = convert_to_transits(combined)

        # TODO : Cleanup and combine all possible statements
        if add:
            return tuple(lhs_transits + rhs_transits + combined_transits)
        else:
            if lhs_transits:
                return tuple(lhs_transits)
            else:
                return tuple()

    def _add_sub_covariates(self, other, add=True):
        lhs = self._extract_covariates()
        rhs = other._extract_covariates()

        res = []
        if len(rhs) != 0 and len(lhs) != 0:
            # Find the unique products in both lists with matching expression/operator
            combined = lhs.copy()
            if add:
                combined.update(rhs)
                for i in combined.copy():
                    if i[4].option:
                        opposite = list(i)
                        opposite[4] = Option(False)
                        opposite = tuple(opposite)
                        combined.discard(opposite)
            else:
                combined.difference_update(rhs)
                for i in rhs:
                    opposite = list(i)
                    opposite[4] = Option(False if i[4].option else True)
                    opposite = tuple(opposite)
                    if opposite in combined:
                        combined.discard(opposite)
                        combined.discard(i)  # Unnecessary ?
            res = combined
        elif len(rhs) != 0 and len(lhs) == 0 and add:
            res = rhs
        elif len(lhs) != 0 and len(rhs) == 0:
            res = lhs
        else:
            return tuple()

        # Convert all elements to SETS before using reduce
        res = [tuple({x} for x in e) for e in res]
        res = _reduce_covariate(res)
        cov_res = []
        for param, cov, fp, op, opt in res:
            cov_res.append(
                Covariate(tuple(param), tuple(cov), tuple(fp), list(op)[0], list(opt)[0])
            )

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
        # TODO : Use add helper and check all in "combined"

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
        # Should OPTIONAL be ignored?
        return all(c in rhs for c in lhs)

    def _extract_peripherals(self):
        peripheral_dict = {"MET": set(), "DRUG": set()}
        for p in self.peripherals:
            for m in p.modes:
                peripheral_dict[m.name] = peripheral_dict[m.name].union(set(p.counts))

        return peripheral_dict

    def _extract_covariates(self):
        lhs = set()
        lhs_ref = []
        for cov in self.covariate:
            if isinstance(cov.parameter, Ref) or isinstance(cov.covariate, Ref):
                lhs_ref.append(cov)
                continue
            else:
                lhs.update(
                    set(
                        product(
                            cov.parameter, cov.covariate, cov.eval().fp, (cov.op,), (cov.optional,)
                        )
                    )
                )

        if len(lhs_ref) != 0:
            raise ValueError(
                'Cannot be performed with reference value. Try using .expand(model) first.'
            )

        return lhs

    def get_number_of_features(self, model=None):
        no_of_features = 0
        for key, attr in vars(self).items():
            if attr is None:
                continue
            if isinstance(attr, tuple):
                if key == '_covariate':
                    no_of_features += sum(feat.get_length(model) for feat in attr)
                else:
                    no_of_features += sum(len(feat) for feat in attr)
            else:
                no_of_features += len(attr)

        return no_of_features


def _add_helper(s1, s2, value_name, join_name):
    s1_join_name_dict = defaultdict(list)
    for s in s1:
        s = s.eval
        attribute_names = getattr(s, join_name)
        attribute_values = getattr(s, value_name)
        for a in attribute_names:
            s1_join_name_dict[a].extend(attribute_values)
    s2_join_name_dict = defaultdict(list)
    for s in s2:
        s = s.eval
        attribute_names = getattr(s, join_name)
        attribute_values = getattr(s, value_name)
        for a in attribute_names:
            s2_join_name_dict[a].extend(attribute_values)
    s1_unique = {
        k: tuple(set(s1_join_name_dict[k]) - set(s2_join_name_dict[k]))
        for k in s1_join_name_dict.keys()
    }
    s2_unique = {
        k: tuple(set(s2_join_name_dict[k]) - set(s1_join_name_dict[k]))
        for k in s2_join_name_dict.keys()
    }
    s2_unique = {k: v for k, v in s2_unique.items() if v}
    s12_joined = {
        k: tuple(set(s1_join_name_dict[k]).intersection(set(s2_join_name_dict[k])))
        for k in s1_join_name_dict.keys()
    }
    s12_joined = {k: v for k, v in s12_joined.items() if v}

    def remove_empty(d):
        return {k: v for k, v in d.items() if v}

    return (remove_empty(s1_unique), remove_empty(s2_unique), remove_empty(s12_joined))


def _reduce_covariate(c):
    c = _reduce(c, 2)
    c = _reduce(c, 1)
    c = _reduce(c, 0)
    return c


def _reduce(s, n):
    """Reduce list of tuples of sets based on the n:th element in each tuple"""
    clean_s = []
    checked_keys = []
    for i in s:
        key = i[:n] + i[n + 1 :]
        if key in checked_keys:
            pass
        else:
            checked_keys.append(key)
            attr_set = set()
            for e in s:
                if key == e[:n] + e[n + 1 :]:
                    attr_set.update(e[n])
            clean_s.append(i[:n] + (attr_set,) + i[n + 1 :])
    return clean_s


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
    covariates = get_covariate_effects(model)

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

    # TODO : Implement IIV, PKPD, METABOLITE, TMDD(?)
    return ";".join(
        [
            e
            for e in [absorption, elimination, lagtime, transits, peripherals, covariates]
            if (e is not None and e != 0)
        ]
    )
