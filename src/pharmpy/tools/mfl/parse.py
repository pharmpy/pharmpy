import warnings
from typing import List, Optional

from lark import Lark

from pharmpy.model import Model
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
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.symbols import Name
from pharmpy.tools.mfl.statement.feature.transits import Transits

from .grammar import grammar
from .helpers import funcs, modelsearch_features
from .interpreter import MFLInterpreter
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

    # TODO : only return class once it has been implemented everywhere
    if mfl_class:
        return ModelFeatures.create_from_mfl_statement_list(MFLInterpreter().interpret(tree))
    else:
        return MFLInterpreter().interpret(tree)


class ModelFeatures:
    def __init__(
        self,
        absorption=Absorption((Name('INST'),)),
        elimination=Elimination((Name('FO'),)),
        transits=(Transits((0,), (Name('DEPOT'),)),),  # NOTE : This is a tuple
        peripherals=Peripherals((0,)),
        lagtime=LagTime((Name('OFF'),)),
        covariate=tuple(),  # TODO : Covariates (Should be represented by tuple similar to transits)
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
        covariate=tuple(),  # TODO : Covariates
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

        # TODO : Covariates

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
                covariate = statement
            else:
                raise ValueError(f'Unknown statement ({statement}) given.')
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
            # TODO : COVARIATES
            mfl_list.append(self.covariate)

        return [m for m in mfl_list if m]

    def convert_to_funcs(
        self, attribute_type: Optional[List[str]] = None, model: Optional[Model] = None
    ):
        # The model argument is used for when extacting covariates.
        if not model:
            model = Model()
        return funcs(model, self.mfl_statement_list(attribute_type), modelsearch_features)

    def contain_subset(self, mfl):
        transits = self._subset_transits(mfl)

        if (
            all([s in self.absorption.modes for s in mfl.absorption.eval.modes])
            and all([s in self.elimination.modes for s in mfl.elimination.eval.modes])
            and transits
            and all([s in self.peripherals.counts for s in mfl.peripherals.counts])
            and all([s in self.lagtime.modes for s in mfl.lagtime.eval.modes])
            # TODO : COVARIATES --> Require model (optional argument?)
        ):
            return True
        else:
            return False

    def _subset_transits(self, mfl):
        lhs_counts = set([c for t in self.transits for c in t.counts])
        lhs_depot = set([d for t in self.transits for d in t.eval.depot])

        rhs_counts = set([c for t in mfl.transits for c in t.counts])
        rhs_depot = set([d for t in mfl.transits for d in t.eval.depot])

        return all([c in lhs_counts for c in rhs_counts]) and all(
            [d in lhs_depot for d in rhs_depot]
        )

    def least_number_of_transformations(self, other):
        """The smallest set of transformations to become part of other"""
        lnt = {}
        if not any(a in other.absorption.eval.modes for a in self.absorption.eval.modes):
            name, func = list(other.convert_to_funcs(["absorption"]).items())[0]
            lnt[name] = func

        if not any(e in other.elimination.eval.modes for e in self.elimination.eval.modes):
            name, func = list(other.convert_to_funcs(["elimination"]).items())[0]
            lnt[name] = func

        # FIXME : ! Currently not working
        lnt = self._lnt_transits(other, lnt)

        if not any(p in other.peripherals.counts for p in self.peripherals.counts):
            name, func = list(other.convert_to_funcs(["peripherals"]).items())[0]
            lnt[name] = func

        if not any(lt in other.lagtime.eval.modes for lt in self.lagtime.eval.modes):
            name, func = list(other.convert_to_funcs(["lagtime"]).items())[0]
            lnt[name] = func

        # TODO : Covariates

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
                        break

            # else take first count value of non matching depot
            depot = next(d for d in rhs_depot if d not in lhs_depot)
            count = next(c for t in other.transits for c in t.counts if depot in t.eval.depot)
            func_dict = other.convert_to_funcs(["transits"])
            lnt[('TRANSITS', count, depot.name)] = func_dict[('TRANSITS', diff[0], depot.name)]

        return lnt

    def __repr__(self):
        # TODO : Remove default values
        return mfl_stringify(self.mfl_statement_list())

    def __sub__(self, other):
        transits = self._add_sub_transits(other, add=False)

        return ModelFeatures.create(
            absorption=self.absorption - other.absorption,
            elimination=self.elimination - other.elimination,
            transits=transits,
            peripherals=self.peripherals - other.peripherals,
            lagtime=self.lagtime - other.lagtime,
            # TODO : Covariate
        )

    def __add__(self, other):
        transits = self._add_sub_transits(other, add=True)

        return ModelFeatures.create(
            absorption=self.absorption + other.absorption,
            elimination=self.elimination + other.elimination,
            transits=transits,
            peripherals=self.peripherals + other.peripherals,
            lagtime=self.lagtime + other.lagtime,
            covariate=self.covariate,  # TODO : Covariate
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

    def __eq__(self, other):
        transits = self._eq_transits(other)
        return (
            self.absorption == other.absorption
            and self.elimination == other.elimination
            and transits
            and self.peripherals == other.peripherals
            and self.lagtime == other.lagtime
            and self.covariate == other.covariate
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

    # TODO : Implement more features such as covariates and IIV

    return ";".join(
        [
            e
            for e in [absorption, elimination, lagtime, transits, peripherals]
            if (e is not None and e != 0)
        ]
    )
